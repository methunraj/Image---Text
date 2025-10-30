-- =========================================================
-- Supabase schema for Desktop OCR Admin + Model Config
-- =========================================================

-- Extensions usually enabled by default on Supabase
create extension if not exists pgcrypto;
create extension if not exists uuid-ossp;

-- ---------------------------------------------------------
-- 1) Users & roles
-- ---------------------------------------------------------
create table if not exists public.user_profiles (
  user_id uuid primary key references auth.users(id) on delete cascade,
  email text not null unique,
  display_name text,
  role text not null check (role in ('admin','editor','user')),
  created_at timestamptz default now(),
  updated_at timestamptz default now()
);

create index if not exists ix_user_profiles_email on public.user_profiles(email);

-- helpers
create or replace function public.is_admin(uid uuid)
returns boolean language sql stable as $$
  select exists (
    select 1 from public.user_profiles p
    where p.user_id = uid and p.role = 'admin'
  );
$$;

create or replace function public.is_editor(uid uuid)
returns boolean language sql stable as $$
  select exists (
    select 1 from public.user_profiles p
    where p.user_id = uid and p.role in ('editor','admin')
  );
$$;

-- ---------------------------------------------------------
-- 2) Projects & assignments
-- ---------------------------------------------------------
create table if not exists public.projects (
  id uuid primary key default gen_random_uuid(),
  name text not null unique,
  description text,
  is_archived boolean default false,
  created_at timestamptz default now(),
  updated_at timestamptz default now()
);

create table if not exists public.user_project (
  user_id uuid not null references auth.users(id) on delete cascade,
  project_id uuid not null references public.projects(id) on delete cascade,
  primary key (user_id, project_id),
  created_at timestamptz default now()
);

-- ---------------------------------------------------------
-- 3) Providers / Models / Parameters
-- (replaces config.xlsx model rows and provider rows)
-- ---------------------------------------------------------
create table if not exists public.model_providers (
  id uuid primary key default gen_random_uuid(),
  name text not null,                 -- "OpenAI", "Anthropic", "Google", "Ollama", etc.
  provider_key text not null,         -- stable id: "openai", "anthropic", "google", "ollama", etc. (unique)
  base_url text not null,
  status text not null default 'active',  -- 'active'|'inactive'
  headers jsonb default '{}'::jsonb,     -- optional static headers
  timeouts jsonb default '{}'::jsonb,    -- {"connect_s":5,"read_s":60,"total_s":120}
  retry jsonb default '{}'::jsonb,       -- {"max_retries":2,"backoff_s":0.5,"retry_on":[429,500,502,503,504]}
  created_by uuid references auth.users(id),
  created_at timestamptz default now(),
  updated_at timestamptz default now(),
  unique(provider_key)
);

create table if not exists public.models (
  id uuid primary key default gen_random_uuid(),
  provider_id uuid not null references public.model_providers(id) on delete cascade,
  model_key text not null,             -- e.g. "gpt-4o", "claude-3-5-sonnet", "gemini-1.5-pro"
  display_name text not null,          -- pretty name
  route text not null,                 -- 'chat_completions'|'responses'|'completions'|'embeddings'|'images'
  context_window int,
  max_output_tokens int,
  max_temperature numeric default 1.0,
  default_temperature numeric default 0.7,
  default_top_p numeric,
  force_json_mode boolean default false,
  prefer_tools boolean default false,
  capabilities jsonb default '{}'::jsonb,     -- {vision,tools,streaming,...}
  compatibility jsonb default '{}'::jsonb,    -- {image_part_key,max_tokens_param,extra_headers,...}
  pricing jsonb default '{}'::jsonb,          -- {input_per_1k,output_per_1k,...}
  reasoning jsonb default '{}'::jsonb,        -- {provider,effort_default,include_thoughts_default,allow_override}
  show_in_ui boolean default true,
  allow_frontend_override_temperature boolean default true,
  allow_frontend_override_reasoning boolean default true,
  status text not null default 'active',
  created_by uuid references auth.users(id),
  created_at timestamptz default now(),
  updated_at timestamptz default now(),
  unique(provider_id, model_key)
);

-- ---------------------------------------------------------
-- 4) Templates & versions (Streamlit Settings manages these)
-- ---------------------------------------------------------
create table if not exists public.prompt_templates (
  id uuid primary key default gen_random_uuid(),
  name text not null unique,
  purpose text not null default 'chat',
  description text,
  current_version_id uuid,                      -- set via trigger when inserting version
  created_by uuid references auth.users(id),
  created_at timestamptz default now(),
  updated_at timestamptz default now()
);

create table if not exists public.prompt_template_versions (
  id uuid primary key default gen_random_uuid(),
  template_id uuid not null references public.prompt_templates(id) on delete cascade,
  version int not null,
  system_prompt text default '',
  user_prompt text default '',
  schema_json jsonb default '{}'::jsonb,
  variables jsonb default '{}'::jsonb,         -- optional variable docs
  is_active boolean default true,
  created_by uuid references auth.users(id),
  created_at timestamptz default now(),
  unique(template_id, version)
);

-- when a new version is inserted as active, update template.current_version_id
create or replace function public._ptv_set_current()
returns trigger language plpgsql as $$
begin
  if new.is_active then
    update public.prompt_templates
    set current_version_id = new.id, updated_at = now()
    where id = new.template_id;
  end if;
  return new;
end
$$;

drop trigger if exists trg_ptv_set_current on public.prompt_template_versions;
create trigger trg_ptv_set_current
after insert or update on public.prompt_template_versions
for each row execute function public._ptv_set_current();

-- ---------------------------------------------------------
-- 5) Project <> Templates, Project-scoped API keys
-- ---------------------------------------------------------
create table if not exists public.project_template (
  project_id uuid not null references public.projects(id) on delete cascade,
  template_id uuid not null references public.prompt_templates(id) on delete cascade,
  primary key (project_id, template_id)
);

create table if not exists public.project_api_keys (
  project_id uuid not null references public.projects(id) on delete cascade,
  provider_id uuid not null references public.model_providers(id) on delete cascade,
  api_key_ciphertext text not null,       -- Fernet ciphertext from desktop app
  key_storage text not null check (key_storage in ('session','encrypted')) default 'encrypted',
  created_by uuid references auth.users(id),
  created_at timestamptz default now(),
  primary key (project_id, provider_id)
);

-- ---------------------------------------------------------
-- 6) Usage analytics (tiny, prompt/outputs NOT stored)
-- ---------------------------------------------------------
create table if not exists public.usage_analytics (
  id uuid primary key default gen_random_uuid(),
  ts timestamptz not null default now(),
  user_id uuid not null references auth.users(id) on delete cascade,
  project_id uuid not null references public.projects(id) on delete cascade,
  template_id uuid references public.prompt_templates(id) on delete set null,
  model_id uuid references public.models(id) on delete set null,
  meta jsonb default '{}'::jsonb
);

-- ---------------------------------------------------------
-- 7) Provider secrets (optional, admin-only)
-- If you prefer storing provider keys centrally (not required if you only use per-project keys)
-- ---------------------------------------------------------
create schema if not exists private;

create table if not exists private.provider_secrets (
  provider_id uuid primary key references public.model_providers(id) on delete cascade,
  api_key_ciphertext text not null,
  meta jsonb default '{}'::jsonb,
  created_by uuid references auth.users(id),
  created_at timestamptz default now(),
  updated_at timestamptz default now()
);

-- ---------------------------------------------------------
-- 8) Audit log
-- ---------------------------------------------------------
create table if not exists public.audit_log (
  id bigserial primary key,
  actor uuid,
  table_name text not null,
  row_pk text not null,
  action text not null check (action in ('INSERT','UPDATE','DELETE')),
  old_data jsonb,
  new_data jsonb,
  created_at timestamptz default now()
);

create or replace function public.log_audit()
returns trigger language plpgsql security definer as $$
declare
  actor_id uuid := auth.uid();
begin
  insert into public.audit_log(actor, table_name, row_pk, action, old_data, new_data)
  values (
    actor_id,
    tg_table_name,
    coalesce(new.id::text, old.id::text, coalesce(new.project_id::text, old.project_id::text, 'unknown')),
    tg_op,
    to_jsonb(old),
    to_jsonb(new)
  );
  return coalesce(new, old);
end
$$;

-- attach audit to key config tables
do $$
begin
  perform 1 from pg_trigger where tgname = 'audit_models';
  if not found then
    create trigger audit_models after insert or update or delete on public.models
    for each row execute function public.log_audit();
  end if;

  perform 1 from pg_trigger where tgname = 'audit_model_providers';
  if not found then
    create trigger audit_model_providers after insert or update or delete on public.model_providers
    for each row execute function public.log_audit();
  end if;

  perform 1 from pg_trigger where tgname = 'audit_templates';
  if not found then
    create trigger audit_templates after insert or update or delete on public.prompt_templates
    for each row execute function public.log_audit();
  end if;

  perform 1 from pg_trigger where tgname = 'audit_template_versions';
  if not found then
    create trigger audit_template_versions after insert or update or delete on public.prompt_template_versions
    for each row execute function public.log_audit();
  end if;

  perform 1 from pg_trigger where tgname = 'audit_projects';
  if not found then
    create trigger audit_projects after insert or update or delete on public.projects
    for each row execute function public.log_audit();
  end if;

  perform 1 from pg_trigger where tgname = 'audit_project_keys';
  if not found then
    create trigger audit_project_keys after insert or update or delete on public.project_api_keys
    for each row execute function public.log_audit();
  end if;
end $$;

-- ---------------------------------------------------------
-- 9) Row-Level Security (RLS)
-- ---------------------------------------------------------
alter table public.user_profiles enable row level security;
alter table public.projects enable row level security;
alter table public.user_project enable row level security;
alter table public.model_providers enable row level security;
alter table public.models enable row level security;
alter table public.prompt_templates enable row level security;
alter table public.prompt_template_versions enable row level security;
alter table public.project_template enable row level security;
alter table public.project_api_keys enable row level security;
alter table public.usage_analytics enable row level security;
alter table private.provider_secrets enable row level security;

-- Read policies (everyone authenticated can read non-secret metadata)
create policy r_models_read on public.models for select to authenticated using (true);
create policy r_model_providers_read on public.model_providers for select to authenticated using (true);
create policy r_templates_read on public.prompt_templates for select to authenticated using (true);
create policy r_template_versions_read on public.prompt_template_versions for select to authenticated using (true);
create policy r_projects_read on public.projects for select to authenticated using (true);
create policy r_project_template_read on public.project_template for select to authenticated using (true);

-- Write policies (editors/admins)
create policy w_models_write on public.models
for all to authenticated using (public.is_editor(auth.uid())) with check (public.is_editor(auth.uid()));
create policy w_model_providers_write on public.model_providers
for all to authenticated using (public.is_editor(auth.uid())) with check (public.is_editor(auth.uid()));
create policy w_templates_write on public.prompt_templates
for all to authenticated using (public.is_editor(auth.uid())) with check (public.is_editor(auth.uid()));
create policy w_template_versions_write on public.prompt_template_versions
for all to authenticated using (public.is_editor(auth.uid())) with check (public.is_editor(auth.uid()));
create policy w_projects_write on public.projects
for all to authenticated using (public.is_editor(auth.uid())) with check (public.is_editor(auth.uid()));
create policy w_project_template_write on public.project_template
for all to authenticated using (public.is_editor(auth.uid())) with check (public.is_editor(auth.uid()));

-- Project access (users only see what they’re assigned for keys & analytics)
create policy r_user_project_read on public.user_project
for select to authenticated using (user_id = auth.uid());
create policy w_user_project_admin on public.user_project
for all to authenticated using (public.is_admin(auth.uid())) with check (public.is_admin(auth.uid()));

create policy r_project_keys_read on public.project_api_keys
for select to authenticated using (
  exists (
    select 1 from public.user_project up
    where up.project_id = project_api_keys.project_id and up.user_id = auth.uid()
  )
);
create policy w_project_keys_write on public.project_api_keys
for all to authenticated using (public.is_admin(auth.uid())) with check (public.is_admin(auth.uid()));

create policy r_usage_read on public.usage_analytics
for select to authenticated using (
  exists (
    select 1 from public.user_project up
    where up.project_id = usage_analytics.project_id and up.user_id = auth.uid()
  )
);
create policy w_usage_insert on public.usage_analytics
for insert to authenticated with check (
  exists (
    select 1 from public.user_project up
    where up.project_id = usage_analytics.project_id and up.user_id = auth.uid()
  )
);

-- Provider secrets are admin-only
create policy r_provider_secrets_admin on private.provider_secrets
for select to authenticated using (public.is_admin(auth.uid()));
create policy w_provider_secrets_admin on private.provider_secrets
for all to authenticated using (public.is_admin(auth.uid())) with check (public.is_admin(auth.uid()));

-- Profiles: users can see themselves; admins can see all; admins edit
create policy r_profiles_self_or_admin on public.user_profiles
for select to authenticated using (user_id = auth.uid() or public.is_admin(auth.uid()));
create policy w_profiles_admin on public.user_profiles
for all to authenticated using (public.is_admin(auth.uid())) with check (public.is_admin(auth.uid()));

-- ---------------------------------------------------------
-- 10) Convenience seed: promote the first profile to admin (run once)
-- (Optional) After inserting your own profile row, run:
--   update public.user_profiles set role='admin' where email='<you@company.com>';
-- ---------------------------------------------------------

