// supabase/functions/import_config_xlsx/index.ts
//
// PURPOSE
//   One-time (or occasional) importer to migrate a legacy config.xlsx into Supabase metadata.
//   Admin-only. Uses service role for privileged upserts, but verifies the caller is admin.
//
// ACCEPTED INPUT
//   POST multipart/form-data with fields:
//     - file:   the Excel workbook (required)
//     - mode:   "dry-run" | "commit" (default: dry-run)
//   (Optional JSON body is also supported: { fileBase64: "<base64xlsx>", mode: "dry-run"|"commit" })
//
// EXPECTED SHEETS (case-insensitive)
//   - Providers: columns => name, provider_key, base_url, status, headers, timeouts, retry
//                headers/timeouts/retry may contain JSON
//   - Models:    provider_key, model_key, display_name, route, context_window, max_output_tokens,
//                default_temperature, default_top_p, force_json_mode, prefer_tools,
//                capabilities, compatibility, pricing, reasoning, show_in_ui,
//                allow_frontend_override_temperature, allow_frontend_override_reasoning, status
//   - Templates: name, purpose, description
//   - TemplateVersions: template_name, version, system_prompt, user_prompt, schema_json, variables_json, is_active
//   - Projects:  name, description, is_archived
//   - ProjectTemplates: project_name, template_name
//
// OUTPUT
//   200 OK: {
//     mode: "dry-run"|"commit",
//     counts: { providers, models, templates, template_versions, projects, project_templates },
//     warnings: string[],
//     examples: { providers?: any[], models?: any[] } // small sample (dry-run only)
//   }
//
// SECURITY / ENV
//   SUPABASE_URL
//   SUPABASE_ANON_KEY
//   SUPABASE_SERVICE_ROLE_KEY
//
// DEPLOY
//   supabase functions deploy import_config_xlsx
//

import { serve } from "https://deno.land/std@0.224.0/http/server.ts";
import { createClient } from "https://esm.sh/@supabase/supabase-js@2";
import * as XLSX from "https://esm.sh/xlsx@0.18.5?target=deno";

const corsHeaders = {
  "Access-Control-Allow-Origin": "*",
  "Access-Control-Allow-Headers": "authorization, x-client-info, apikey, content-type",
  "Access-Control-Allow-Methods": "POST, OPTIONS",
};
const jsonHeaders = { ...corsHeaders, "Content-Type": "application/json" };

type Row = Record<string, any>;
type LowerRow = Record<string, any>;

serve(async (req: Request) => {
  if (req.method === "OPTIONS") return new Response(null, { headers: corsHeaders });

  try {
    if (req.method !== "POST") {
      return new Response(JSON.stringify({ error: "Method not allowed" }), { status: 405, headers: jsonHeaders });
    }

    const supabaseUrl = Deno.env.get("SUPABASE_URL")!;
    const anonKey = Deno.env.get("SUPABASE_ANON_KEY")!;
    const serviceKey = Deno.env.get("SUPABASE_SERVICE_ROLE_KEY");
    if (!supabaseUrl || !anonKey || !serviceKey) {
      return new Response(JSON.stringify({ error: "Missing Supabase env vars" }), { status: 500, headers: jsonHeaders });
    }

    // Verify caller is authenticated admin
    const userClient = createClient(supabaseUrl, anonKey, {
      global: { headers: { Authorization: req.headers.get("Authorization") ?? "" } },
    });
    const { data: { user: caller } } = await userClient.auth.getUser();
    if (!caller) return new Response(JSON.stringify({ error: "Unauthorized" }), { status: 401, headers: jsonHeaders });
    const { data: prof } = await userClient.from("user_profiles").select("role").eq("user_id", caller.id).single();
    if (!prof || prof.role !== "admin") {
      return new Response(JSON.stringify({ error: "Forbidden (admin only)" }), { status: 403, headers: jsonHeaders });
    }

    // Read input (multipart or JSON)
    let mode: "dry-run" | "commit" = "dry-run";
    let wb: XLSX.WorkBook | null = null;

    const ctype = req.headers.get("content-type") || "";
    if (ctype.includes("multipart/form-data")) {
      const form = await req.formData();
      const modeField = String(form.get("mode") || "dry-run").toLowerCase();
      if (modeField === "commit") mode = "commit";
      const file = form.get("file") as File | null;
      if (!file) return new Response(JSON.stringify({ error: "file is required" }), { status: 400, headers: jsonHeaders });
      const buf = new Uint8Array(await file.arrayBuffer());
      wb = XLSX.read(buf, { type: "array" });
    } else {
      const body = await req.json().catch(() => ({}));
      const modeField = String(body.mode || "dry-run").toLowerCase();
      if (modeField === "commit") mode = "commit";
      const fileBase64 = body.fileBase64 ? String(body.fileBase64) : "";
      if (!fileBase64) return new Response(JSON.stringify({ error: "fileBase64 is required for JSON body" }), { status: 400, headers: jsonHeaders });
      const bin = Uint8Array.from(atob(fileBase64), c => c.charCodeAt(0));
      wb = XLSX.read(bin, { type: "array" });
    }

    if (!wb) return new Response(JSON.stringify({ error: "Failed to parse workbook" }), { status: 400, headers: jsonHeaders });

    const svc = createClient(supabaseUrl, serviceKey);

    // Helpers
    const warnings: string[] = [];
    const counts = {
      providers: 0,
      models: 0,
      templates: 0,
      template_versions: 0,
      projects: 0,
      project_templates: 0,
    };

    function sheet(name: string): Row[] {
      const ws = wb!.Sheets[findSheet(name)];
      if (!ws) return [];
      const rows = XLSX.utils.sheet_to_json<Row>(ws, { defval: null, raw: true });
      return rows;
    }
    function findSheet(name: string): string {
      const names = wb!.SheetNames;
      const target = name.toLowerCase();
      for (const n of names) if (n.toLowerCase() === target) return n;
      return name; // let it fail to undefined if truly missing
    }
    function lowerKeys(r: Row): LowerRow {
      const o: LowerRow = {};
      for (const k of Object.keys(r)) o[k.toLowerCase().trim()] = r[k];
      return o;
    }
    function parseJSONMaybe(val: any, label: string): any {
      if (val == null || val === "") return {};
      if (typeof val === "object") return val;
      try {
        return JSON.parse(String(val));
      } catch {
        warnings.push(`Failed to parse JSON in ${label}: ${val}`);
        return {};
      }
    }
    function toBool(v: any): boolean {
      if (typeof v === "boolean") return v;
      const s = String(v ?? "").trim().toLowerCase();
      return ["1","true","yes","y","on"].includes(s);
    }
    function toNumOrNull(v: any): number | null {
      const n = Number(v);
      return Number.isFinite(n) ? n : null;
    }

    // ---- Parse each sheet into arrays ----
    const rowsProviders = sheet("Providers").map(lowerKeys);
    const rowsModels = sheet("Models").map(lowerKeys);
    const rowsTemplates = sheet("Templates").map(lowerKeys);
    const rowsTemplateVersions = sheet("TemplateVersions").map(lowerKeys);
    const rowsProjects = sheet("Projects").map(lowerKeys);
    const rowsProjTemplates = sheet("ProjectTemplates").map(lowerKeys);

    // ---- Dry-run preview payloads (and eventual commit payloads) ----
    const providersPayload = rowsProviders.map((r) => ({
      name: String(r.name || "").trim(),
      provider_key: String(r.provider_key || "").trim(),
      base_url: String(r.base_url || "").trim(),
      status: (String(r.status || "active").trim().toLowerCase() === "inactive") ? "inactive" : "active",
      headers: parseJSONMaybe(r.headers, "Providers.headers"),
      timeouts: parseJSONMaybe(r.timeouts, "Providers.timeouts"),
      retry: parseJSONMaybe(r.retry, "Providers.retry"),
    })).filter(p => p.name && p.provider_key && p.base_url);

    const modelsPayload = rowsModels.map((r) => ({
      // provider_id will be resolved later by provider_key
      provider_key: String(r.provider_key || "").trim(),
      model_key: String(r.model_key || "").trim(),
      display_name: String(r.display_name || "").trim(),
      route: (String(r.route || "responses").trim() as any),
      context_window: toNumOrNull(r.context_window),
      max_output_tokens: toNumOrNull(r.max_output_tokens),
      max_temperature: null, // can be extended
      default_temperature: toNumOrNull(r.default_temperature),
      default_top_p: toNumOrNull(r.default_top_p),
      force_json_mode: toBool(r.force_json_mode),
      prefer_tools: toBool(r.prefer_tools),
      capabilities: parseJSONMaybe(r.capabilities, "Models.capabilities"),
      compatibility: parseJSONMaybe(r.compatibility, "Models.compatibility"),
      pricing: parseJSONMaybe(r.pricing, "Models.pricing"),
      reasoning: parseJSONMaybe(r.reasoning, "Models.reasoning"),
      show_in_ui: r.show_in_ui == null ? true : toBool(r.show_in_ui),
      allow_frontend_override_temperature: r.allow_frontend_override_temperature == null ? true : toBool(r.allow_frontend_override_temperature),
      allow_frontend_override_reasoning: r.allow_frontend_override_reasoning == null ? true : toBool(r.allow_frontend_override_reasoning),
      status: (String(r.status || "active").trim().toLowerCase() === "inactive") ? "inactive" : "active",
    })).filter(m => m.provider_key && m.model_key && m.display_name);

    const templatesPayload = rowsTemplates.map((r) => ({
      name: String(r.name || "").trim(),
      purpose: String(r.purpose || "chat").trim(),
      description: r.description ? String(r.description) : null,
    })).filter(t => t.name);

    const templateVersionsPayload = rowsTemplateVersions.map((r) => ({
      template_name: String(r.template_name || "").trim(),
      version: Number(r.version ?? 0) || 1,
      system_prompt: r.system_prompt ? String(r.system_prompt) : "",
      user_prompt: r.user_prompt ? String(r.user_prompt) : "",
      schema_json: parseJSONMaybe(r.schema_json, "TemplateVersions.schema_json"),
      variables: parseJSONMaybe(r.variables_json ?? r.variables, "TemplateVersions.variables"),
      is_active: r.is_active == null ? true : toBool(r.is_active),
    })).filter(v => v.template_name);

    const projectsPayload = rowsProjects.map((r) => ({
      name: String(r.name || "").trim(),
      description: r.description ? String(r.description) : null,
      is_archived: toBool(r.is_archived),
    })).filter(p => p.name);

    const projectTemplatesPayload = rowsProjTemplates.map((r) => ({
      project_name: String(r.project_name || "").trim(),
      template_name: String(r.template_name || "").trim(),
    })).filter(x => x.project_name && x.template_name);

    // ---- DRY RUN reply ----
    if (mode === "dry-run") {
      return new Response(JSON.stringify({
        mode,
        counts: {
          providers: providersPayload.length,
          models: modelsPayload.length,
          templates: templatesPayload.length,
          template_versions: templateVersionsPayload.length,
          projects: projectsPayload.length,
          project_templates: projectTemplatesPayload.length,
        },
        warnings,
        examples: {
          providers: providersPayload.slice(0, 3),
          models: modelsPayload.slice(0, 3),
        },
      }), { status: 200, headers: jsonHeaders });
    }

    // ---- COMMIT (transaction-like best-effort with ordered steps) ----

    // 1) Providers
    let provResult: any[] = [];
    if (providersPayload.length) {
      const { data, error } = await svc.from("model_providers")
        .upsert(providersPayload, { onConflict: "provider_key" })
        .select();
      if (error) return resp500(error, "providers upsert");
      provResult = data || [];
    }
    counts.providers = provResult.length;

    // Map provider_key -> id
    const providerKeyToId = new Map<string, string>();
    for (const p of provResult) providerKeyToId.set(p.provider_key, p.id);

    // 2) Models (resolve provider_id)
    const modelsRows = modelsPayload.map(m => {
      const provider_id = providerKeyToId.get(m.provider_key);
      if (!provider_id) {
        warnings.push(`Model with model_key='${m.model_key}' references unknown provider_key='${m.provider_key}' (skipped).`);
        return null;
      }
      const { provider_key, ...rest } = m;
      return { ...rest, provider_id };
    }).filter(Boolean) as Row[];

    let modelsResult: any[] = [];
    if (modelsRows.length) {
      const { data, error } = await svc.from("models")
        .upsert(modelsRows, { onConflict: "provider_id,model_key" })
        .select();
      if (error) return resp500(error, "models upsert");
      modelsResult = data || [];
    }
    counts.models = modelsResult.length;

    // 3) Templates
    let templResult: any[] = [];
    if (templatesPayload.length) {
      const { data, error } = await svc.from("prompt_templates")
        .upsert(templatesPayload, { onConflict: "name" })
        .select();
      if (error) return resp500(error, "templates upsert");
      templResult = data || [];
    }
    counts.templates = templResult.length;

    // Map template_name -> id
    const nameToTemplateId = new Map<string, string>();
    for (const t of templResult) nameToTemplateId.set((t.name as string).trim(), t.id);

    // 4) Template Versions
    //    Insert per group (template) in ascending version order
    let tvCount = 0;
    if (templateVersionsPayload.length) {
      // group by template_name
      const byTpl = new Map<string, Row[]>();
      for (const v of templateVersionsPayload) {
        const arr = byTpl.get(v.template_name) || [];
        arr.push(v);
        byTpl.set(v.template_name, arr);
      }

      for (const [tname, versions] of byTpl) {
        const tplId = nameToTemplateId.get(tname);
        if (!tplId) {
          warnings.push(`TemplateVersions references unknown template_name='${tname}' (skipped).`);
          continue;
        }
        versions.sort((a, b) => (a.version - b.version));
        const rows = versions.map(v => ({
          template_id: tplId,
          version: v.version,
          system_prompt: v.system_prompt,
          user_prompt: v.user_prompt,
          schema_json: v.schema_json,
          variables: v.variables,
          is_active: v.is_active,
        }));
        const { data, error } = await svc.from("prompt_template_versions").insert(rows).select();
        if (error) return resp500(error, `template_versions insert for ${tname}`);
        tvCount += (data?.length || 0);
      }
    }
    counts.template_versions = tvCount;

    // 5) Projects
    let projResult: any[] = [];
    if (projectsPayload.length) {
      const { data, error } = await svc.from("projects")
        .upsert(projectsPayload, { onConflict: "name" })
        .select();
      if (error) return resp500(error, "projects upsert");
      projResult = data || [];
    }
    counts.projects = projResult.length;

    // Map project_name -> id
    const nameToProjectId = new Map<string, string>();
    for (const p of projResult) nameToProjectId.set((p.name as string).trim(), p.id);

    // 6) Project-Template links
    let ptCount = 0;
    if (projectTemplatesPayload.length) {
      const rows = projectTemplatesPayload.map(x => {
        const pid = nameToProjectId.get(x.project_name);
        const tid = nameToTemplateId.get(x.template_name);
        if (!pid || !tid) {
          warnings.push(`ProjectTemplates row skipped: (${x.project_name}, ${x.template_name})`);
          return null;
        }
        return { project_id: pid, template_id: tid };
      }).filter(Boolean) as Row[];
      if (rows.length) {
        const { data, error } = await svc.from("project_template")
          .upsert(rows, { onConflict: "project_id,template_id" })
          .select();
        if (error) return resp500(error, "project_template upsert");
        ptCount = data?.length || 0;
      }
    }
    counts.project_templates = ptCount;

    // Done
    return new Response(JSON.stringify({ mode, counts, warnings }), { status: 200, headers: jsonHeaders });

  } catch (e) {
    return new Response(JSON.stringify({ error: String(e) }), { status: 500, headers: jsonHeaders });
  }
});

// ---- helpers ----
function resp500(error: any, ctx: string) {
  const msg = (error && error.message) ? error.message : String(error);
  return new Response(JSON.stringify({ error: `${ctx}: ${msg}` }), { status: 500, headers: jsonHeaders });
}
