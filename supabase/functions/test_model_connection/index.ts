// supabase/functions/test_model_connection/index.ts
//
// PURPOSE
//   Admin-only connectivity test for a given provider+model.
//   - Verifies caller is admin (via public.user_profiles)
//   - Decrypts provider's API key from private.provider_secrets (AES-GCM; same as File 12)
//   - Performs a low-cost OpenAI-compatible request based on the model's route
//   - Returns { ok, status, latency_ms, route, url, sample }
//
// INPUT (POST JSON)
//   {
//     "providerId": "<uuid>",
//     "modelKey": "gpt-4o" | "claude-3-5-sonnet" | ...,
//     "testType": "ping" | "sample_prompt",
//     "samplePrompt": "optional user text",
//     "routeOverride": "responses|chat_completions|completions|embeddings|images"  // optional
//   }
//
// OUTPUT (200)
//   {
//     ok: true,
//     status: 200,
//     latency_ms: 123,
//     provider_key: "openai|anthropic|...",
//     route: "chat_completions",
//     url: "https://.../v1/chat/completions",
//     sample: { id: "...", created: 123, ... }   // redacted subset if available
//   }
//
// ENV (project secrets)
//   SUPABASE_URL
//   SUPABASE_ANON_KEY
//   SUPABASE_SERVICE_ROLE_KEY
//   PROVIDER_SECRET_PASSPHRASE   <-- same passphrase as save_provider_secret
//
// DEPLOY
//   supabase functions deploy test_model_connection
//

import { serve } from "https://deno.land/std@0.224.0/http/server.ts";
import { createClient } from "https://esm.sh/@supabase/supabase-js@2";

const corsHeaders = {
  "Access-Control-Allow-Origin": "*",
  "Access-Control-Allow-Headers": "authorization, x-client-info, apikey, content-type",
  "Access-Control-Allow-Methods": "POST, OPTIONS",
};
const jsonHeaders = { ...corsHeaders, "Content-Type": "application/json" };

// ---------- crypto helpers (AES-GCM + PBKDF2)  ----------
const ITER = 200_000;

function b64urlToBytes(b64url: string): Uint8Array {
  const b64 = b64url.replace(/-/g, "+").replace(/_/g, "/") + "===".slice((b64url.length + 3) % 4);
  const bin = atob(b64);
  const out = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i);
  return out;
}
function fromUtf8(s: string) { return new TextEncoder().encode(s); }
function toUtf8(ab: ArrayBuffer) { return new TextDecoder().decode(ab); }

async function deriveKey(passphrase: string, salt: Uint8Array): Promise<CryptoKey> {
  const baseKey = await crypto.subtle.importKey("raw", fromUtf8(passphrase), { name: "PBKDF2" }, false, ["deriveKey"]);
  return crypto.subtle.deriveKey(
    { name: "PBKDF2", salt, iterations: ITER, hash: "SHA-256" },
    baseKey,
    { name: "AES-GCM", length: 256 },
    false,
    ["encrypt", "decrypt"],
  );
}
async function decryptCiphertext(passphrase: string, payloadJson: string): Promise<string> {
  const payload = JSON.parse(payloadJson);
  // shape produced by File 12
  const salt = b64urlToBytes(payload.salt);
  const iv = b64urlToBytes(payload.iv);
  const ct = b64urlToBytes(payload.ct);
  const key = await deriveKey(passphrase, salt);
  const pt = await crypto.subtle.decrypt({ name: "AES-GCM", iv }, key, ct);
  return toUtf8(pt);
}

// ---------- provider/model helpers ----------
type Route = "responses"|"chat_completions"|"completions"|"embeddings"|"images";

function normalizeBaseUrl(base: string): string {
  return base.replace(/\/+$/, ""); // trim trailing slashes
}
function defaultPathForRoute(route: Route): string {
  switch (route) {
    case "responses": return "/v1/responses";
    case "chat_completions": return "/v1/chat/completions";
    case "completions": return "/v1/completions";
    case "embeddings": return "/v1/embeddings";
    case "images": return "/v1/images/generations";
    default: return "/v1/responses";
  }
}
function headersForProvider(providerKey: string, apiKey: string, extra: Record<string, any>): HeadersInit {
  const h: Record<string, string> = { "Content-Type": "application/json" };
  const pk = (providerKey || "").toLowerCase();
  if (pk.includes("anthropic")) {
    h["x-api-key"] = apiKey;
    // choose a sane default; can be overridden via extra headers if provided
    if (!("anthropic-version" in extra)) h["anthropic-version"] = "2023-06-01";
  } else {
    h["Authorization"] = `Bearer ${apiKey}`;
  }
  // merge static headers from provider row if any
  for (const [k, v] of Object.entries(extra || {})) {
    if (typeof v === "string") h[k] = v;
  }
  return h;
}

function makeBody(route: Route, modelKey: string, testType: string, samplePrompt?: string): any {
  const prompt = (samplePrompt && String(samplePrompt).trim()) || "ping";
  switch (route) {
    case "responses":
      // OpenAI Responses: cheap 1 token
      return { model: modelKey, input: prompt, max_output_tokens: 1 };
    case "chat_completions":
      return { model: modelKey, messages: [{ role: "user", content: prompt }], max_tokens: 1 };
    case "completions":
      return { model: modelKey, prompt, max_tokens: 1 };
    case "embeddings":
      return { model: modelKey, input: prompt };
    case "images":
      // Avoid cost explosions for image generation in a "test" call:
      throw new Error("images route not supported in test; choose embeddings or chat for ping.");
    default:
      return { model: modelKey, input: prompt, max_output_tokens: 1 };
  }
}

serve(async (req: Request) => {
  if (req.method === "OPTIONS") return new Response(null, { headers: corsHeaders });

  try {
    if (req.method !== "POST") {
      return new Response(JSON.stringify({ error: "Method not allowed" }), { status: 405, headers: jsonHeaders });
    }

    const supabaseUrl = Deno.env.get("SUPABASE_URL")!;
    const anonKey = Deno.env.get("SUPABASE_ANON_KEY")!;
    const serviceKey = Deno.env.get("SUPABASE_SERVICE_ROLE_KEY");
    const passphrase = Deno.env.get("PROVIDER_SECRET_PASSPHRASE");
    if (!supabaseUrl || !anonKey || !serviceKey || !passphrase) {
      return new Response(JSON.stringify({ error: "Missing Supabase env vars" }), { status: 500, headers: jsonHeaders });
    }

    const body = await req.json().catch(() => ({}));
    const providerId = (body.providerId ?? "").toString().trim();
    const modelKey = (body.modelKey ?? "").toString().trim();
    const testType = (body.testType ?? "ping").toString();
    const samplePrompt = body.samplePrompt ? String(body.samplePrompt) : undefined;
    const routeOverride = body.routeOverride ? String(body.routeOverride) as Route : undefined;

    if (!providerId || !modelKey) {
      return new Response(JSON.stringify({ error: "providerId and modelKey are required" }), { status: 400, headers: jsonHeaders });
    }

    // Caller must be authenticated + admin
    const userClient = createClient(supabaseUrl, anonKey, {
      global: { headers: { Authorization: req.headers.get("Authorization") ?? "" } },
    });
    const { data: { user: caller } } = await userClient.auth.getUser();
    if (!caller) return new Response(JSON.stringify({ error: "Unauthorized" }), { status: 401, headers: jsonHeaders });
    const { data: prof, error: profErr } = await userClient.from("user_profiles").select("role").eq("user_id", caller.id).single();
    if (profErr || !prof || prof.role !== "admin") {
      return new Response(JSON.stringify({ error: "Forbidden (admin only)" }), { status: 403, headers: jsonHeaders });
    }

    // Service-role client to fetch provider/model + secret (bypass RLS)
    const svc = createClient(supabaseUrl, serviceKey);

    const { data: provider, error: pErr } = await svc
      .from("model_providers")
      .select("id, name, provider_key, base_url, headers")
      .eq("id", providerId)
      .single();
    if (pErr || !provider) {
      return new Response(JSON.stringify({ error: "Unknown providerId" }), { status: 404, headers: jsonHeaders });
    }

    // Determine route from model row (unless override provided)
    let route: Route = "responses";
    if (routeOverride) {
      route = routeOverride;
    } else {
      const { data: model, error: mErr } = await svc
        .from("models")
        .select("route")
        .eq("provider_id", providerId)
        .eq("model_key", modelKey)
        .single();
      if (mErr || !model) {
        return new Response(JSON.stringify({ error: "Model not found for this provider" }), { status: 404, headers: jsonHeaders });
      }
      route = (model.route as Route) || "responses";
    }

    // Load and decrypt provider secret
    const { data: secretRow, error: sErr } = await svc
      .from("private.provider_secrets")
      .select("api_key_ciphertext")
      .eq("provider_id", providerId)
      .maybeSingle();

    if (sErr) {
      return new Response(JSON.stringify({ error: sErr.message }), { status: 500, headers: jsonHeaders });
    }
    if (!secretRow?.api_key_ciphertext) {
      return new Response(JSON.stringify({ error: "No provider secret set. Add a key in Admin → Model Configuration." }), {
        status: 400, headers: jsonHeaders,
      });
    }
    const apiKey = await decryptCiphertext(passphrase, secretRow.api_key_ciphertext);

    // Build request
    const baseUrl = normalizeBaseUrl(provider.base_url);
    const path = defaultPathForRoute(route);
    const url = `${baseUrl}${path}`;
    const headers = headersForProvider(provider.provider_key || "", apiKey, provider.headers || {});
    const method = "POST";
    let payload: any;

    try {
      payload = makeBody(route, modelKey, testType, samplePrompt);
    } catch (err) {
      return new Response(JSON.stringify({ error: (err as Error).message }), { status: 400, headers: jsonHeaders });
    }

    // Fire request and time it
    const t0 = Date.now();
    const resp = await fetch(url, { method, headers, body: JSON.stringify(payload) });
    const latency = Date.now() - t0;

    // Try to parse a small subset of response for diagnostics
    let sample: any = null;
    try {
      const j = await resp.json();
      // Redact large text; keep only small metadata
      if (j && typeof j === "object") {
        sample = {
          id: j.id ?? null,
          created: j.created ?? null,
          object: j.object ?? null,
          model: j.model ?? null,
          status: j.status ?? null,
          // Return a 1-line excerpt of text if present
          excerpt:
            (Array.isArray(j.choices) && j.choices[0]?.message?.content)
              ? String(j.choices[0].message.content).slice(0, 120)
              : (j.output && Array.isArray(j.output) && typeof j.output[0]?.content === "string")
                ? String(j.output[0].content).slice(0, 120)
                : undefined,
        };
      }
    } catch {
      // ignore JSON parse errors (some endpoints may return non-JSON on error)
      sample = null;
    }

    const out = {
      ok: resp.ok,
      status: resp.status,
      latency_ms: latency,
      provider_key: provider.provider_key,
      route,
      url,
      sample,
    };
    return new Response(JSON.stringify(out), { status: 200, headers: jsonHeaders });

  } catch (e) {
    return new Response(JSON.stringify({ error: String(e) }), { status: 500, headers: jsonHeaders });
  }
});
