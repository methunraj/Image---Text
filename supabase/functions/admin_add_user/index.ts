// supabase/functions/admin_add_user/index.ts
//
// PURPOSE
//   Admin-only edge function to add a user:
//     - Verifies caller is authenticated AND role='admin' in public.user_profiles
//     - Creates or invites the target user via Auth Admin API
//     - Upserts (user_id, email, display_name, role) into public.user_profiles
//
// INPUT (POST JSON)
//   {
//     "email": "someone@company.com",
//     "role": "admin" | "editor" | "user",
//     "displayName": "Someone",
//     "sendInvite": true      // default true; if false => createUser without email invite
//   }
//
// OUTPUT (200)
//   {
//     ok: true,
//     user_id: "...",
//     email: "...",
//     role: "editor",
//     invited: true,          // true if invite flow used
//     created: false,         // true if createUser used and a new user was created
//     profile: { ... }        // row from public.user_profiles
//   }
//
// ENV (set in project secrets):
//   SUPABASE_URL
//   SUPABASE_ANON_KEY
//   SUPABASE_SERVICE_ROLE_KEY   <-- required for auth.admin.* and RLS bypass on writes
//
// DEPLOY
//   supabase functions deploy admin_add_user
//   supabase secrets set SUPABASE_SERVICE_ROLE_KEY=<service_role_key>
//
// NOTE
//   This function uses two clients:
//     1) userClient  (anon key + caller's JWT) to verify the caller & role
//     2) serviceClient (service role) to perform admin ops + upsert profile
//

import { serve } from "https://deno.land/std@0.224.0/http/server.ts";
import { createClient } from "https://esm.sh/@supabase/supabase-js@2";

const corsHeaders = {
  "Access-Control-Allow-Origin": "*",
  "Access-Control-Allow-Headers":
    "authorization, x-client-info, apikey, content-type",
  "Access-Control-Allow-Methods": "POST, OPTIONS",
};
const jsonHeaders = { ...corsHeaders, "Content-Type": "application/json" };

serve(async (req: Request) => {
  if (req.method === "OPTIONS") return new Response(null, { headers: corsHeaders });

  try {
    if (req.method !== "POST") {
      return new Response(JSON.stringify({ error: "Method not allowed" }), {
        status: 405,
        headers: jsonHeaders,
      });
    }

    const supabaseUrl = Deno.env.get("SUPABASE_URL")!;
    const anonKey = Deno.env.get("SUPABASE_ANON_KEY")!;
    const serviceKey = Deno.env.get("SUPABASE_SERVICE_ROLE_KEY");
    if (!supabaseUrl || !anonKey || !serviceKey) {
      return new Response(JSON.stringify({ error: "Missing Supabase env vars" }), {
        status: 500,
        headers: jsonHeaders,
      });
    }

    // Parse body
    const body = await req.json().catch(() => ({}));
    const emailRaw = (body.email ?? "").toString().trim().toLowerCase();
    const role = (body.role ?? "user").toString();
    const displayName =
      (body.displayName ?? body.display_name ?? "").toString().trim() || null;
    const sendInvite = body.sendInvite !== false; // default true

    if (!emailRaw) {
      return new Response(JSON.stringify({ error: "email is required" }), {
        status: 400,
        headers: jsonHeaders,
      });
    }
    if (!["admin", "editor", "user"].includes(role)) {
      return new Response(JSON.stringify({ error: "invalid role" }), {
        status: 400,
        headers: jsonHeaders,
      });
    }

    // Client bound to caller's JWT (for verifying admin role)
    const userClient = createClient(supabaseUrl, anonKey, {
      global: { headers: { Authorization: req.headers.get("Authorization") ?? "" } },
    });

    // Caller must be authenticated
    const {
      data: { user: caller },
      error: callerErr,
    } = await userClient.auth.getUser();
    if (callerErr || !caller) {
      return new Response(JSON.stringify({ error: "Unauthorized" }), {
        status: 401,
        headers: jsonHeaders,
      });
    }

    // Caller must be admin
    const { data: profile, error: profErr } = await userClient
      .from("user_profiles")
      .select("role")
      .eq("user_id", caller.id)
      .single();

    if (profErr || !profile || profile.role !== "admin") {
      return new Response(JSON.stringify({ error: "Forbidden (admin only)" }), {
        status: 403,
        headers: jsonHeaders,
      });
    }

    // Service-role client for privileged ops
    const serviceClient = createClient(supabaseUrl, serviceKey);

    // Create/invite the target user
    let targetUser: any = null;
    let invited = false;
    let created = false;

    if (sendInvite) {
      const { data, error } = await serviceClient.auth.admin.inviteUserByEmail(
        emailRaw,
      );
      if (!error && data?.user) {
        targetUser = data.user;
        invited = true;
      } else {
        // If invite failed (e.g., already exists), try to resolve user by email
        const { data: list, error: listErr } =
          await serviceClient.auth.admin.listUsers({ page: 1, perPage: 1000 });
        if (!listErr && list?.users?.length) {
          targetUser =
            list.users.find((u: any) =>
              (u.email ?? "").toLowerCase() === emailRaw
            ) ?? null;
        }
        if (!targetUser) {
          return new Response(
            JSON.stringify({ error: error?.message || "Failed to invite user" }),
            { status: 400, headers: jsonHeaders },
          );
        }
      }
    } else {
      const { data, error } = await serviceClient.auth.admin.createUser({
        email: emailRaw,
        email_confirm: false,
      });
      if (error) {
        // If already exists, locate by email
        const { data: list, error: listErr } =
          await serviceClient.auth.admin.listUsers({ page: 1, perPage: 1000 });
        if (!listErr && list?.users?.length) {
          targetUser =
            list.users.find((u: any) =>
              (u.email ?? "").toLowerCase() === emailRaw
            ) ?? null;
        }
        if (!targetUser) {
          return new Response(JSON.stringify({ error: error.message }), {
            status: 400,
            headers: jsonHeaders,
          });
        }
      } else {
        targetUser = data.user;
        created = true;
      }
    }

    // Upsert into public.user_profiles (service role bypasses RLS)
    const upsertPayload = {
      user_id: targetUser.id,
      email: emailRaw,
      display_name: displayName,
      role,
    };

    const { data: upserted, error: upErr } = await serviceClient
      .from("user_profiles")
      .upsert(upsertPayload, { onConflict: "user_id" })
      .select()
      .single();

    if (upErr) {
      return new Response(JSON.stringify({ error: upErr.message }), {
        status: 500,
        headers: jsonHeaders,
      });
    }

    return new Response(
      JSON.stringify({
        ok: true,
        user_id: targetUser.id,
        email: emailRaw,
        role,
        invited,
        created,
        profile: upserted,
      }),
      { status: 200, headers: jsonHeaders },
    );
  } catch (e) {
    return new Response(JSON.stringify({ error: String(e) }), {
      status: 500,
      headers: jsonHeaders,
    });
  }
});
