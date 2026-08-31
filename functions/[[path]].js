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
// Translation helpers: Anthropic Messages API ↔ OpenAI chat.completions
// ===========================================================================

function anthropicToOpenAI(body, model) {
  const { messages = [], system, max_tokens, stream, temperature, top_p } = body;

  const openaiMessages = [];
  if (system) {
    const text = Array.isArray(system) ? system.map(b => b.text || "").join("") : system;
    openaiMessages.push({ role: "system", content: text });
  }
  for (const msg of messages) {
    const content = Array.isArray(msg.content)
      ? msg.content.map(b => b.text || "").join("")
      : (msg.content || "");
    openaiMessages.push({ role: msg.role, content });
  }

  const out = { model, messages: openaiMessages };
  if (max_tokens) out.max_tokens = max_tokens;
  if (temperature !== undefined) out.temperature = temperature;
  if (top_p !== undefined) out.top_p = top_p;
  if (stream) { out.stream = true; out.stream_options = { include_usage: true }; }
  return out;
}

function openaiToAnthropic(openaiBody, model) {
  const choice = openaiBody.choices?.[0];
  const usage = openaiBody.usage || {};
  return {
    id: openaiBody.id || ("msg_" + crypto.randomUUID().replace(/-/g, "").slice(0, 24)),
    type: "message",
    role: "assistant",
    content: [{ type: "text", text: choice?.message?.content || "" }],
    model,
    stop_reason: choice?.finish_reason === "length" ? "max_tokens" : "end_turn",
    stop_sequence: null,
    usage: { input_tokens: usage.prompt_tokens || 0, output_tokens: usage.completion_tokens || 0 },
  };
}

// Reads OpenAI SSE stream, emits Anthropic SSE stream — text lines only, no binary parsing
function openaiSSEToAnthropicSSE(upstreamBody, model) {
  const { readable, writable } = new TransformStream();
  const writer = writable.getWriter();
  const enc = new TextEncoder();
  const emit = async (obj) => writer.write(enc.encode(`data: ${JSON.stringify(obj)}\n\n`));

  (async () => {
    let buf = "";
    let started = false;
    let stopped = false;
    let outputTokens = 0;

    try {
      const reader = upstreamBody.pipeThrough(new TextDecoderStream()).getReader();
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buf += value;
        const lines = buf.split("\n");
        buf = lines.pop();

        for (const line of lines) {
          if (!line.startsWith("data: ")) continue;
          const raw = line.slice(6).trim();

          if (raw === "[DONE]") {
            if (!stopped) {
              stopped = true;
              await emit({ type: "content_block_stop", index: 0 });
              await emit({ type: "message_delta", delta: { stop_reason: "end_turn", stop_sequence: null }, usage: { output_tokens: outputTokens } });
              await emit({ type: "message_stop" });
            }
            continue;
          }

          let chunk;
          try { chunk = JSON.parse(raw); } catch { continue; }

          // usage-only chunk (from stream_options.include_usage)
          if (chunk.usage) {
            outputTokens = chunk.usage.completion_tokens || 0;
          }

          const choice = chunk.choices?.[0];
          if (!choice) continue;

          if (!started) {
            started = true;
            await emit({ type: "message_start", message: { id: chunk.id, type: "message", role: "assistant", content: [], model, usage: { input_tokens: 0, output_tokens: 0 } } });
            await emit({ type: "content_block_start", index: 0, content_block: { type: "text", text: "" } });
            await emit({ type: "ping" });
          }

          const text = choice.delta?.content;
          if (text) await emit({ type: "content_block_delta", index: 0, delta: { type: "text_delta", text } });

          if (choice.finish_reason && !stopped) {
            stopped = true;
            const stopReason = choice.finish_reason === "length" ? "max_tokens" : "end_turn";
            await emit({ type: "content_block_stop", index: 0 });
            await emit({ type: "message_delta", delta: { stop_reason: stopReason, stop_sequence: null }, usage: { output_tokens: outputTokens } });
            await emit({ type: "message_stop" });
          }
        }
      }
    } catch {}
    finally { await writer.close(); }
  })();

  return readable;
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
    parsed.model = model;
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
// /v1/messages — Anthropic SDK clients, translates to/from OpenAI on the wire
// ===========================================================================

async function messagesRelay(request, env) {
  const err = await validateKey(request, env);
  if (err) return err;

  const region = env.AWS_REGION || "us-east-1";
  const model = env.ANTHROPIC_MODEL || "us.anthropic.claude-sonnet-4-6";

  let anthropicBody;
  try { anthropicBody = await request.json(); }
  catch { return json({ error: "Invalid JSON body" }, 400); }

  const isStream = anthropicBody.stream === true;
  const openaiBody = anthropicToOpenAI(anthropicBody, model);

  const upstream = await fetch(
    `https://bedrock-runtime.${region}.amazonaws.com/openai/v1/chat/completions`,
    {
      method: "POST",
      headers: { "Authorization": `Bearer ${env.AWS_BEARER_TOKEN_BEDROCK}`, "Content-Type": "application/json" },
      body: JSON.stringify(openaiBody),
    }
  );

  if (!upstream.ok) {
    return new Response(upstream.body, {
      status: upstream.status,
      headers: { "Content-Type": "application/json", ...corsHeaders },
    });
  }

  if (isStream) {
    return new Response(openaiSSEToAnthropicSSE(upstream.body, model), {
      status: 200,
      headers: { "Content-Type": "text/event-stream", "Cache-Control": "no-cache", ...corsHeaders },
    });
  }

  return json(openaiToAnthropic(await upstream.json(), model));
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
