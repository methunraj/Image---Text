// supabase/functions/save_provider_secret/index.ts
//
// PURPOSE
//   Admin-only edge function to securely store a provider API key.
//   - Verifies the caller is authenticated AND role='admin' in public.user_profiles
//   - Encrypts plaintext apiKey using AES-GCM (WebCrypto) with a server-side passphrase
//   - Upserts { provider_id, api_key_ciphertext, meta } into private.provider_secrets
//
// INPUT (POST JSON)
//   { "providerId": "<uuid>", "apiKey": "<plaintext string>", "meta": { ... } }
//
// OUTPUT (200)
//   { ok: true, provider_id: "...", has_key: true }
//
// ENV (set as project secrets)
//   SUPABASE_URL
//   SUPABASE_ANON_KEY
//   SUPABASE_SERVICE_ROLE_KEY     <-- required for privileged writes
//   PROVIDER_SECRET_PASSPHRASE    <-- strong passphrase for AES-GCM key derivation
//
// DEPLOY
//   supabase secrets set SUPABASE_SERVICE_ROLE_KEY=<service_role_key>
//   supabase secrets set PROVIDER_SECRET_PASSPHRASE='<long_random_phrase>'
//   supabase functions deploy save_provider_secret
//
// NOTE
//   - Ciphertext format (serialized as JSON string in api_key_ciphertext):
//       {
//         "v": 1,
//         "alg": "AES-GCM",
//         "kdf": "PBKDF2-SHA256",
//         "iter": 200000,
//         "salt": "<b64url>",
//         "iv": "<b64url>",
//         "ct": "<b64url>"
//       }
//   - File 13 (test_model_connection) will use the SAME scheme to decrypt.
//

import { serve } from "https://deno.land/std@0.224.0/http/server.ts";
import { createClient } from "https://esm.sh/@supabase/supabase-js@2";

const corsHeaders = {
  "Access-Control-Allow-Origin": "*",
  "Access-Control-Allow-Headers": "authorization, x-client-info, apikey, content-type",
  "Access-Control-Allow-Methods": "POST, OPTIONS",
};
const jsonHeaders = { ...corsHeaders, "Content-Type": "application/json" };

// --- crypto helpers (AES-GCM + PBKDF2) ---------------------------------

const ITER = 200_000;
const SALT_BYTES = 16;
const IV_BYTES = 12;

function b64url(bytes: ArrayBuffer): string {
  const bin = String.fromCharCode(...new Uint8Array(bytes));
  const b64 = btoa(bin).replace(/\+/g, "-").replace(/\//g, "_").replace(/=+$/g, "");
  return b64;
}

function fromUtf8(s: string): Uint8Array {
  return new TextEncoder().encode(s);
}

function toUtf8(ab: ArrayBuffer): string {
  return new TextDecoder().decode(ab);
}

async function deriveKey(passphrase: string, salt: Uint8Array): Promise<CryptoKey> {
  const baseKey = await crypto.subtle.importKey(
    "raw",
    fromUtf8(passphrase),
    { name: "PBKDF2" },
    false,
    ["deriveKey"]
  );
  return crypto.subtle.deriveKey(
    { name: "PBKDF2", salt, iterations: ITER, hash: "SHA-256" },
    baseKey,
    { name: "AES-GCM", length: 256 },
    false,
    ["encrypt", "decrypt"]
  );
}

async function encryptApiKey(passphrase: string, plaintext: string) {
  const salt = crypto.getRandomValues(new Uint8Array(SALT_BYTES));
  const iv = crypto.getRandomValues(new Uint8Array(IV_BYTES));
  const key = await deriveKey(passphrase, salt);
  const ct = await crypto.subtle.encrypt({ name: "AES-GCM", iv }, key, fromUtf8(plaintext));
  const payload = {
    v: 1,
    alg: "AES-GCM",
    kdf: "PBKDF2-SHA256",
    iter: ITER,
    salt: b64url(salt.buffer),
    iv: b64url(iv.buffer),
    ct: b64url(ct),
  };
  return JSON.stringify(payload);
}

// ----------------------------------------------------------------------

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
      return new Response(JSON.stringify({ error: "Missing env: SUPABASE_URL/ANON_KEY/SERVICE_ROLE_KEY/PASSPHRASE" }), {
        status: 500, headers: jsonHeaders,
      });
    }

    // Parse input
    const body = await req.json().catch(() => ({}));
    const providerId = (body.providerId ?? "").toString().trim();
    const apiKey = (body.apiKey ?? "").toString();
    const meta = (body.meta && typeof body.meta === "object") ? body.meta : {};

    if (!providerId || !apiKey) {
      return new Response(JSON.stringify({ error: "providerId and apiKey are required" }), { status: 400, headers: jsonHeaders });
    }

    // Client with caller's JWT (to verify admin role)
    const userClient = createClient(supabaseUrl, anonKey, {
      global: { headers: { Authorization: req.headers.get("Authorization") ?? "" } },
    });

    // Caller must be authenticated
    const { data: { user: caller }, error: callerErr } = await userClient.auth.getUser();
    if (callerErr || !caller) {
      return new Response(JSON.stringify({ error: "Unauthorized" }), { status: 401, headers: jsonHeaders });
    }

    // Caller must be admin
    const { data: prof, error: profErr } = await userClient
      .from("user_profiles")
      .select("role")
      .eq("user_id", caller.id)
      .single();

    if (profErr || !prof || prof.role !== "admin") {
      return new Response(JSON.stringify({ error: "Forbidden (admin only)" }), { status: 403, headers: jsonHeaders });
    }

    // Service-role client for privileged writes (bypass RLS)
    const serviceClient = createClient(supabaseUrl, serviceKey);

    // Verify provider exists
    const { data: prov, error: provErr } = await serviceClient
      .from("model_providers")
      .select("id")
      .eq("id", providerId)
      .single();

    if (provErr || !prov) {
      return new Response(JSON.stringify({ error: "Unknown providerId" }), { status: 404, headers: jsonHeaders });
    }

    // Encrypt on server
    const ciphertext = await encryptApiKey(passphrase, apiKey);

    // Upsert into private.provider_secrets
    const upsertPayload = {
      provider_id: providerId,
      api_key_ciphertext: ciphertext,
      meta: meta || {},
      created_by: caller.id,
    };

    const { data: row, error: upErr } = await serviceClient
      .from("private.provider_secrets")
      .upsert(upsertPayload, { onConflict: "provider_id" })
      .select("provider_id")
      .single();

    if (upErr) {
      return new Response(JSON.stringify({ error: upErr.message }), { status: 500, headers: jsonHeaders });
    }

    return new Response(JSON.stringify({ ok: true, provider_id: row.provider_id, has_key: true }), {
      status: 200,
      headers: jsonHeaders,
    });

  } catch (e) {
    return new Response(JSON.stringify({ error: String(e) }), { status: 500, headers: jsonHeaders });
  }
});
