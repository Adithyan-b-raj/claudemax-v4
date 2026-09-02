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
// Auth helper
// ===========================================================================

async function validateKey(request, env) {
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
  return null;
}

// ===========================================================================
// Native Bedrock binary event-stream helpers
// ===========================================================================

function parseEventHeaders(buf) {
  const headers = {};
  const dv = new DataView(buf.buffer, buf.byteOffset);
  let i = 0;
  while (i < buf.length) {
    const nameLen = buf[i++];
    const name = new TextDecoder().decode(buf.slice(i, i + nameLen)); i += nameLen;
    i++;
    const valLen = dv.getUint16(i, false); i += 2;
    const value = new TextDecoder().decode(buf.slice(i, i + valLen)); i += valLen;
    headers[name] = value;
  }
  return headers;
}

function readEventFrame(accumulated) {
  if (accumulated.length < 12) return null;
  const dv = new DataView(accumulated.buffer, accumulated.byteOffset);
  const totalLen = dv.getUint32(0, false);
  if (accumulated.length < totalLen) return null;
  const headersLen = dv.getUint32(4, false);
  const headersEnd = 12 + headersLen;
  const payloadEnd = totalLen - 4;
  const headers = parseEventHeaders(accumulated.slice(12, headersEnd));
  const payload = new TextDecoder().decode(accumulated.slice(headersEnd, payloadEnd));
  return { headers, payload, consumed: totalLen };
}

function normalizeBedRockError(body) {
  try {
    const parsed = JSON.parse(body);
    if (parsed.__type) {
      const typeMap = {
        "ValidationException": "invalid_request_error",
        "ThrottlingException": "rate_limit_error",
        "ModelNotReadyException": "api_error",
        "ModelStreamErrorException": "api_error",
        "AccessDeniedException": "authentication_error",
        "ResourceNotFoundException": "not_found_error",
      };
      return JSON.stringify({
        type: "error",
        error: { type: typeMap[parsed.__type] || "api_error", message: parsed.message || parsed.__type },
      });
    }
  } catch {}
  return body;
}

// ===========================================================================
// /v1/messages — native Bedrock invoke, Anthropic SSE output
// ===========================================================================

async function messagesRelay(request, env) {
  const err = await validateKey(request, env);
  if (err) return err;

  const rawBody = await request.text();
  let isStream = false;
  try { isStream = JSON.parse(rawBody).stream === true; } catch {}

  let parsed;
  try { parsed = JSON.parse(rawBody); } catch { return json({ error: "Invalid JSON body" }, 400); }
  delete parsed.model;
  delete parsed.stream;
  delete parsed.context_management;
  parsed.anthropic_version = "bedrock-2023-05-31";
  const bedrockBody = JSON.stringify(parsed);

  const region = env.AWS_REGION || "us-east-1";
  const model = env.ANTHROPIC_MODEL || "us.anthropic.claude-sonnet-4-6";
  const bedrockBase = `https://bedrock-runtime.${region}.amazonaws.com/model/${encodeURIComponent(model)}`;
  const invokeUrl = isStream ? `${bedrockBase}/invoke-with-response-stream` : `${bedrockBase}/invoke`;

  const upstream = await fetch(invokeUrl, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "Authorization": `Bearer ${env.AWS_BEARER_TOKEN_BEDROCK}`,
    },
    body: bedrockBody,
  });

  if (isStream && !upstream.ok) {
    const errBody = await upstream.text();
    return new Response(normalizeBedRockError(errBody), {
      status: upstream.status,
      headers: { "Content-Type": "application/json", ...corsHeaders },
    });
  }

  if (isStream && upstream.body) {
    const { readable, writable } = new TransformStream();
    const writer = writable.getWriter();
    const encoder = new TextEncoder();
    const reader = upstream.body.getReader();
    let accumulated = new Uint8Array(0);

    (async () => {
      try {
        while (true) {
          const { done, value } = await reader.read();
          if (done) break;

          const next = new Uint8Array(accumulated.length + value.length);
          next.set(accumulated);
          next.set(value, accumulated.length);
          accumulated = next;

          while (true) {
            const frame = readEventFrame(accumulated);
            if (!frame) break;
            accumulated = accumulated.slice(frame.consumed);

            const eventType = frame.headers[":event-type"];

            if (eventType === "modelStreamErrorException") {
              try {
                const err = JSON.parse(frame.payload);
                await writer.write(encoder.encode(`data: ${JSON.stringify({ type: "error", error: { type: "api_error", message: err.message || "Bedrock stream error" } })}\n\n`));
              } catch {}
              break;
            }

            if (eventType === "chunk") {
              const wrapper = JSON.parse(frame.payload);
              const anthropicJson = new TextDecoder().decode(Uint8Array.from(atob(wrapper.bytes), c => c.charCodeAt(0)));
              let eventName = "";
              try { eventName = JSON.parse(anthropicJson).type || ""; } catch {}
              const sseData = eventName
                ? `event: ${eventName}\ndata: ${anthropicJson}\n\n`
                : `data: ${anthropicJson}\n\n`;
              await writer.write(encoder.encode(sseData));
            }
          }
        }
      } catch {}
      finally { await writer.close(); }
    })();

    return new Response(readable, {
      status: 200,
      headers: { "Content-Type": "text/event-stream; charset=utf-8", "Cache-Control": "no-cache", "X-Accel-Buffering": "no", ...corsHeaders },
    });
  }

  const respBody = await upstream.text();
  const outBody = upstream.ok ? respBody : normalizeBedRockError(respBody);

  const headers = new Headers();
  const allowed = new Set(["content-type", "date", "cache-control", "retry-after", "x-request-id"]);
  for (const [key, val] of upstream.headers) {
    if (allowed.has(key.toLowerCase())) headers.set(key, val);
  }
  Object.entries(corsHeaders).forEach(([k, v]) => headers.set(k, v));
  return new Response(outBody, { status: upstream.status, headers });
}

// ===========================================================================
// /v1/chat/completions — pure OpenAI passthrough
// ===========================================================================

async function proxyRelay(request, env) {
  const err = await validateKey(request, env);
  if (err) return err;

  const region = env.AWS_REGION || "us-east-1";
  const model = env.ANTHROPIC_MODEL || "us.anthropic.claude-sonnet-4-6";

  let body;
  try {
    const parsed = await request.json();
    parsed.model = parsed.model || model;
    body = JSON.stringify(parsed);
  } catch {
    return json({ error: "Invalid JSON body" }, 400);
  }

  const upstream = await fetch(
    `https://bedrock-runtime.${region}.amazonaws.com/openai/v1/chat/completions`,
    {
      method: "POST",
      headers: { "Authorization": `Bearer ${env.AWS_BEARER_TOKEN_BEDROCK}`, "Content-Type": "application/json" },
      body,
    }
  );

  return new Response(upstream.body, {
    status: upstream.status,
    headers: { "Content-Type": upstream.headers.get("Content-Type") || "application/json", "Cache-Control": "no-cache", ...corsHeaders },
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
      usage: `curl -X POST https://${request.headers.get("host")}/v1/messages -H "x-api-key: ${apiKey}" -H "Content-Type: application/json" -d '{...}'`
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

  if (path === "/v1/messages" && request.method === "POST") {
    return messagesRelay(request, env);
  }

  if (path === "/v1/chat/completions" && request.method === "POST") {
    return proxyRelay(request, env);
  }

  if (path === "/v1/models" && request.method === "GET") {
    return json({
      object: "list",
      data: [{ id: "us.anthropic.claude-sonnet-4-6", object: "model", created: 1700000000, owned_by: "anthropic" }],
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
