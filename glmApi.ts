import type { GeneratedSlide } from './types';
import { createSSEParser } from './sse';
import { createExtractionState, walkEvent, finalizeSlides, OnPartialHtml } from './extractSlides';


// --- tiny debug helpers (no new files, no UI changes)
const debugEnabled = (ch: string) => {
  if (typeof window === 'undefined') return false;
  const qs = new URLSearchParams(location.search).get("debug")?.split(",") ?? [];
  const ls = (localStorage.getItem("slides_debug") || "").split(",");
  const set = new Set([...qs, ...ls].map(s => s.trim()).filter(Boolean));
  return set.has("all") || set.has(ch);
};
const maskSecrets = (s: string) =>
  s.replace(/(Authorization:\s*Bearer\s+)[A-Za-z0-9\.\-\_]+/gi, "$1•••")
   .replace(/("?(token|api[_-]?key|download_token|access[_-]?token)"?\s*:\s*")([^"]+)(")/gi, '$1•••$4');
const hdrsToString = (h: Headers) => {
  const arr: string[] = [];
  h.forEach((v, k) => arr.push(`${k}: ${k.toLowerCase().includes("authorization") ? "•••" : v}`));
  return arr.join("\n");
};
const dumpFormData = async (fd: FormData) => {
  const parts: string[] = [];
  for (const [k, v] of fd.entries()) {
    if (v instanceof File) parts.push(`${k}: [file ${v.name} ${v.type || "application/octet-stream"} ${v.size}B]`);
    else parts.push(`${k}: ${String(v).slice(0, 1000)}`);
  }
  return parts.join("\n");
};
const now = () => new Date().toISOString();
const log = (onLog?: (s: string)=>void, ch = "NET", msg = "") => {
  if (!onLog || !debugEnabled(ch.toLowerCase())) return;
  onLog(`[${now()}] [${ch}] ${msg}`);
};

// Wrap fetch with request/response tracing and retry logic
async function debugFetch(
  input: RequestInfo | URL,
  init: RequestInit,
  onLog?: (s: string)=>void,
  opts: { label?: string; bodyPreview?: boolean; retries?: number } = {}
) {
  const { label = "HTTP", bodyPreview = true, retries = 1 } = opts;
  let attempt = 0;
  let lastError: any;

  while(attempt < retries) {
    attempt++;
    const start = performance.now();
    try {
      if (attempt > 1) {
        const delay = 1000 * (attempt - 1);
        log(onLog, "NET", `Retrying request for "${label}" (${attempt}/${retries}) in ${delay}ms...`);
        await new Promise(r => setTimeout(r, delay));
      }
      
      const method = (init?.method || "GET").toUpperCase();
      const url = typeof input === "string" ? input : (input as URL).toString();
      const headers = new Headers(init?.headers || {});
      if (debugEnabled("net")) {
        let bodyStr = "";
        if (init?.body instanceof FormData) {
          bodyStr = await dumpFormData(init.body);
        } else if (typeof init?.body === "string") {
          bodyStr = maskSecrets(init.body as string);
        }
        log(onLog, "NET", `▶ ${method} ${url}\nHeaders:\n${hdrsToString(headers)}${bodyStr ? `\nBody:\n${bodyStr}` : ""}`);
      }

      const res = await fetch(input, init);
      const ms = Math.round(performance.now() - start);
      const ray = res.headers.get("cf-ray") || res.headers.get("x-amz-cf-id") || "-";
      const respHdrs = hdrsToString(res.headers);
      log(onLog, "NET", `◀ ${method} ${url} → ${res.status} ${res.statusText} (${ms} ms) ray=${ray}`);
      if (debugEnabled("headers")) log(onLog, "NET", `Response Headers:\n${respHdrs}`);

      if (!res.ok && res.status >= 500 && attempt < retries) {
          log(onLog, "NET", `Server error ${res.status} for "${label}". Retrying...`);
          lastError = new Error(`Server error: ${res.status} ${res.statusText}`);
          continue;
      }
      
      const ctype = res.headers.get("content-type") || "";
      if (bodyPreview !== false && (ctype.includes("application/json") || (ctype.includes("text/") && !ctype.includes("event-stream")))) {
        try {
          const preview = await res.clone().text();
          if (preview) {
            const short = maskSecrets(preview).slice(0, 4000);
            log(onLog, "NET", `Response Preview:\n${short}${preview.length > 4000 ? "\n…(truncated)" : ""}`);
          }
        } catch { /* ignore */ }
      }
      return res;
    } catch (err: any) {
      lastError = err;
      const ms = Math.round(performance.now() - start);
      log(onLog, "NET", `✖ ${label} (attempt ${attempt}) failed after ${ms} ms: ${err?.message || err}`);
      
      const isNetworkError = err.name === 'TypeError';
      if (isNetworkError && attempt < retries) {
          continue;
      }
      
      throw err;
    }
  }
  
  throw lastError || new Error(`${label} request failed after ${retries} attempts.`);
}


interface GlmApiParams {
  prompt: string;
  apiKey: string;
  onComplete?: (conversationId: string, slides: GeneratedSlide[]) => void;
  onError?: (error: string) => void;
  onPartial?: (data: { pos: number; html: string; complete: boolean }) => void;
  conversationId?: string;
  signal?: AbortSignal;
  onLog?: (message: string) => void;
  kbId?: string;
}

export async function callGlmAgent({
  prompt,
  apiKey,
  onComplete,
  onError,
  onPartial,
  conversationId,
  signal,
  onLog,
  kbId,
}: GlmApiParams) {
  const body: {
    agent_id: string;
    stream: boolean;
    messages: { role: string; content: { type: string; text: string }[] }[];
    conversation_id?: string;
    tools?: any[];
  } = {
    agent_id: 'slides_glm_agent',
    stream: true,
    messages: [
      {
        role: 'user',
        content: [{ type: 'text', text: prompt }],
      },
    ],
  };

  if (conversationId) {
    body.conversation_id = conversationId;
  }
  
  if (kbId) {
    onLog?.(`[GLM API] Knowledge Base ID available: ${kbId}`);
  }

  try {
    onLog?.(`[GLM API] Sending request... Conversation ID: ${conversationId || 'New'}`);
    const response = await debugFetch('https://open.bigmodel.cn/api/v1/agents', {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${apiKey}`,
        'Content-Type': 'application/json',
        'Accept': 'text/event-stream'
      },
      body: JSON.stringify(body),
      signal,
    }, onLog, { label: "agents", bodyPreview: false, retries: 3 });
    
    onLog?.(`[GLM API] Request sent. Status: ${response.status}`);

    if (!response.ok || !response.body) {
      const errorText = await response.text();
      throw new Error(`API request failed: ${response.status} ${response.statusText} - ${errorText}`);
    }
    
    const ctype = response.headers.get("content-type") || "";
    log(onLog, "SSE", `connected status=${response.status} ray=${response.headers.get("cf-ray") || "-"} content-type=${ctype}`);

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let finalConvId = conversationId || '';
    let rawSseBuffer = '';
    
    const logFn = onLog || (() => {});
    const extractionState = createExtractionState();
    const seenEventIds = new Set<string>();

    const onPartialCallback: OnPartialHtml = (pos, html, complete) => {
        onPartial?.({ pos, html, complete });
    };

    const parse = createSSEParser(evt => {
        if (evt.id && seenEventIds.has(evt.id)) {
            if (debugEnabled("sse")) log(onLog, "SSE", `dedup skip id=${evt.id}`);
            return;
        }
        if (evt.id) {
            seenEventIds.add(evt.id);
        }

        if (debugEnabled("sse-raw")) log(onLog, "SSE-RAW", evt.data);

        try {
            if (evt.data.trim() === '[DONE]') {
                logFn('[SSE] Received [DONE] marker.');
                return;
            }
            const eventData = JSON.parse(evt.data);

            if (debugEnabled("sse")) {
                const choice = eventData?.choices?.[0];
                const msg = choice?.messages?.[0] ?? choice?.message;
                const phase = msg?.phase ?? eventData?.phase;
                const content = msg?.content;
                let toolInfo = "";
                if (Array.isArray(content)) {
                    for(const part of content) {
                        if (part?.type === 'object' && part.object) {
                            const tool = String(part.object.tool_name || part.object.name || "").toLowerCase();
                            const pos = part.object.position?.[0] ?? part.object.page_index;
                            const isAdd = /insert|add/.test(tool) && /page|slide/.test(tool);
                            if (isAdd) {
                                toolInfo += ` tool=${tool}${pos != null ? ` pos=${pos}` : ''}`;
                            }
                        }
                    }
                }
                log(onLog, "SSE", `phase=${phase || "-"}${toolInfo}`);
            }

            if (eventData.conversation_id && !finalConvId) {
                finalConvId = eventData.conversation_id;
            }
            walkEvent(extractionState, eventData, logFn, onPartialCallback);
        } catch (e) {
             logFn(`[SSE] Error parsing event data: ${e}. Data: ${evt.data}`);
        }
    });

    onLog?.('[GLM API] Waiting for stream...');
    while (true) {
      const { done, value } = await reader.read();
      if (done) {
        break;
      }
      const chunk = decoder.decode(value, { stream: true });
      rawSseBuffer += chunk;
      parse(chunk);
    }
    
    parse("", { flush: true }); // Flush any remaining buffer
    onLog?.('[GLM API] Stream finished.');

    if (typeof window !== 'undefined') {
        (window as any).__SSE_LAST__ = { runId: finalConvId, raw: rawSseBuffer };
    }

    if (onComplete) {
      const finalDeck = finalizeSlides(extractionState);
      const finalSlides: GeneratedSlide[] = finalDeck.map(s => ({
          pageNumber: s.position,
          html: s.html,
          draft: s.html,
          complete: true,
      }));
      onComplete(finalConvId, finalSlides);
    }

  } catch (error: any) {
    let message = error instanceof Error ? error.message : 'An unknown error occurred';
    onLog?.(`[GLM API] Error: ${message}`);

    if (error.name === 'AbortError') {
      message = 'Request aborted by client.';
      onLog?.(`[GLM API] ${message}`);
    } else if (error.name === 'TypeError') {
      message = "A network error occurred after multiple retries. This could be due to a CORS policy, a firewall, or loss of internet connectivity. Please check your network and the server's status.";
      onLog?.(`[GLM API] A persistent network error was caught. This is often a CORS or connectivity issue.`);
    }

    console.error('GLM API Error:', error);
    if (onError) {
      onError(message);
    }
  }
}

export function addPage(params: Omit<GlmApiParams, 'conversationId'>) {
    return callGlmAgent(params);
}

export function updatePage(params: GlmApiParams) {
    if (!params.conversationId) {
        const errMsg = "conversationId is required for updatePage";
        console.error(errMsg);
        if (params.onError) params.onError(errMsg);
        return;
    }
    return callGlmAgent(params);
}
