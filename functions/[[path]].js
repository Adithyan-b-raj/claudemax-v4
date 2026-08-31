const corsHeaders = {
  "Access-Control-Allow-Origin": "*",
  "Access-Control-Allow-Methods": "GET,POST,DELETE,OPTIONS",
  "Access-Control-Allow-Headers": "Authorization,Content-Type,X-Api-Key",
};

function json(body, status = 200) {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "content-type": "application/json", ...corsHeaders },
  });
}

// ===========================================================================
// KV Data Layer
// ===========================================================================

async function getKey(env, apiKey) {
  return await env.API_KEYS.get(apiKey, "json");
}

async function putKey(env, apiKey, record, ttlSec) {
  await env.API_KEYS.put(apiKey, JSON.stringify(record), { expirationTtl: ttlSec });
}

async function deleteKey(env, apiKey) {
  await env.API_KEYS.delete(apiKey);
  await removeFromIndex(env, apiKey);
}

async function getIndex(env) {
  return (await env.API_KEYS.get("__index__", "json")) || [];
}

async function addToIndex(env, apiKey) {
  const index = await getIndex(env);
  if (!index.includes(apiKey)) {
    index.push(apiKey);
    await env.API_KEYS.put("__index__", JSON.stringify(index));
  }
}

async function removeFromIndex(env, apiKey) {
  const index = await getIndex(env);
  await env.API_KEYS.put("__index__", JSON.stringify(index.filter(k => k !== apiKey)));
}

// ===========================================================================
// Proxy relay — pure passthrough to Bedrock OpenAI-compatible endpoint
// ===========================================================================

async function proxyRelay(request, env) {
  const apiKey = request.headers.get("x-api-key")
    || request.headers.get("Authorization")?.replace(/^Bearer\s+/i, "").trim()
    || new URL(request.url).searchParams.get("apiKey");

  if (!apiKey) return json({ error: "Missing X-Api-Key header or apiKey param" }, 401);

  const record = await getKey(env, apiKey);
  if (!record) return json({ error: "Invalid API key" }, 403);
  if (new Date(record.expiresAt) < new Date()) {
    await deleteKey(env, apiKey);
    return json({ error: "API key expired" }, 403);
  }

  const region = env.AWS_REGION || "us-east-1";
  const allowedModel = env.ANTHROPIC_MODEL || "us.anthropic.claude-sonnet-4-6-v1:0";

  // Override model to enforce allowlist — response stream is still pure passthrough
  let body;
  try {
    const parsed = await request.json();
    parsed.model = allowedModel;
    body = JSON.stringify(parsed);
  } catch {
    return json({ error: "Invalid JSON body" }, 400);
  }

  const upstream = await fetch(
    `https://bedrock-runtime.${region}.amazonaws.com/openai/v1/chat/completions`,
    {
      method: "POST",
      headers: {
        "Authorization": `Bearer ${env.AWS_BEARER_TOKEN_BEDROCK}`,
        "Content-Type": "application/json",
      },
      body,
    }
  );

  return new Response(upstream.body, {
    status: upstream.status,
    headers: {
      "Content-Type": upstream.headers.get("Content-Type") || "application/json",
      "Cache-Control": "no-cache",
      ...corsHeaders,
    },
  });
}

// ===========================================================================
// Admin handler
// ===========================================================================

async function handleAdmin(request, env, adminSecret) {
  const url = new URL(request.url);
  const path = url.pathname;

  if (request.method === "GET" && (path === "/admin" || path === "/admin/")) {
    return new Response("", { status: 302, headers: { Location: `${url.origin}/dashboard.html` } });
  }

  const auth = request.headers.get("Authorization") || "";
  if (!auth.startsWith("Bearer ") || auth.slice(7) !== adminSecret) {
    return json({ error: "unauthorized" }, 401);
  }

  if (request.method === "GET" && path === "/admin/keys") {
    const index = await getIndex(env);
    const keys = [];
    for (const k of index) {
      const data = await getKey(env, k);
      if (data) keys.push({ ...data, apiKey: k, id: k });
    }
    return json({ keys });
  }

  if (request.method === "POST" && path === "/admin/create") {
    const body = await request.json().catch(() => ({}));
    const days = Math.min(365, Math.max(1, parseInt(body.days) || 7));
    const name = (body.name || "api-key").slice(0, 50);
    const apiKey = generateKey(32);
    const now = new Date();
    const expiresAt = new Date(now.getTime() + days * 86400000);
    const record = { expiresAt: expiresAt.toISOString(), createdAt: now.toISOString(), name };
    await putKey(env, apiKey, record, Math.ceil((days + 1) * 86400));
    await addToIndex(env, apiKey);
    return json({
      apiKey,
      expiresAt: record.expiresAt,
      name,
      usage: `curl -X POST https://${request.headers.get("host")}/v1/chat/completions -H "x-api-key: ${apiKey}" -H "Content-Type: application/json" -d '{...}'`
    }, 201);
  }

  if (request.method === "POST" && path === "/admin/revoke") {
    const body = await request.json().catch(() => ({}));
    if (!body.apiKey) return json({ error: "apiKey required" }, 400);
    await deleteKey(env, body.apiKey);
    return json({ ok: true });
  }

  return json({ error: "not found" }, 404);
}

// ===========================================================================
// Entry point
// ===========================================================================

export async function onRequest(context) {
  const { request, env } = context;
  const url = new URL(request.url);
  const path = url.pathname;

  if (request.method === "OPTIONS") {
    return new Response(null, { status: 204, headers: corsHeaders });
  }

  if (path === "/v1/chat/completions" && request.method === "POST") {
    return proxyRelay(request, env);
  }

  if (path === "/v1/models" && request.method === "GET") {
    return json({
      object: "list",
      data: [{ id: "us.anthropic.claude-sonnet-4-6-v1:0", object: "model", created: 1700000000, owned_by: "anthropic" }],
    });
  }

  if (path === "/health") {
    return json({ status: "ok" });
  }

  if (path.startsWith("/admin")) {
    const adminSecret = env.ADMIN_SECRET;
    if (!adminSecret) return json({ error: "Set ADMIN_SECRET env var" }, 503);
    return handleAdmin(request, env, adminSecret);
  }

  if (env.ASSETS) return env.ASSETS.fetch(request);

  return json({ error: "not found" }, 404);
}

// ===========================================================================
// Key generation
// ===========================================================================

function generateKey(length) {
  const chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789";
  const max = 256 - (256 % chars.length);
  const result = [];
  while (result.length < length) {
    const arr = new Uint8Array(1);
    crypto.getRandomValues(arr);
    if (arr[0] < max) result.push(chars[arr[0] % chars.length]);
  }
  return result.join("");
}
