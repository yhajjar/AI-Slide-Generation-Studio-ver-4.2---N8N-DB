import type { CourseData, GeneralCourseSlide, MicrolearningSlide, KbStatus, GeneratedSlide } from './types';
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

const KB_BASE_URL = 'https://open.bigmodel.cn/api/llm-application/open';

async function createKb(apiKey: string, onLog?: (message: string) => void): Promise<string> {
  const url = `${KB_BASE_URL}/knowledge`;
  const name = `ai-slides-kb-${Date.now()}`;
  onLog?.(`[KB Create] Creating new Knowledge Base with name: ${name}`);
  const response = await debugFetch(url, {
    method: 'POST',
    headers: {
      'Authorization': `Bearer ${apiKey}`,
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      name,
      embedding_id: 11, // Embedding-3-new
      description: "KB for AI Slide Generation Studio",
    }),
  }, onLog, { label: "createKb", retries: 3 });
  if (!response.ok) {
    const text = await response.text();
    onLog?.(`[KB Create] Error: ${text}`);
    throw new Error(`Failed to create KB: ${response.status} - ${text}`);
  }
  const data = await response.json();
  const kbId = data?.data?.id;
  if (!kbId) {
    onLog?.(`[KB Create] Error: KB ID not found in response. ${JSON.stringify(data)}`);
    throw new Error('KB ID not found in create response.');
  }
  onLog?.(`[KB Create] Successfully created KB with ID: ${kbId}`);
  return kbId;
}

export async function uploadDocument(
  file: File,
  kbId: string,
  apiKey: string,
  onLog?: (message: string) => void
): Promise<string> {
  const url = `${KB_BASE_URL}/document/upload_document/${kbId}`;
  const formData = new FormData();
  formData.append('files', file, file.name);
  formData.append('knowledge_type', '6'); // Use page-based slicing for PDFs

  onLog?.(`[KB Upload] Uploading file "${file.name}" to KB: ${kbId}`);

  const response = await debugFetch(url, {
    method: 'POST',
    headers: {
      'Authorization': `Bearer ${apiKey}`,
    },
    body: formData,
  }, onLog, { label: "uploadDocument", bodyPreview: false, retries: 3 });

  const data = await response.json();

  if (!response.ok) {
    const errorMessage = data.message || JSON.stringify(data);
    onLog?.(`[KB Upload] Error: ${errorMessage}`);
    throw new Error(`Upload failed: ${response.status} - ${errorMessage}`);
  }

  const documentId = data?.data?.successInfos?.[0]?.documentId;

  if (!documentId) {
    const failedInfo = data?.data?.failedInfos?.[0];
    const reason = failedInfo?.failReason || 'documentId not found in successInfos';
    onLog?.(`[KB Upload] Error: ${reason}. Response: ${JSON.stringify(data)}`);
    throw new Error(`Upload failed: ${reason}`);
  }

  onLog?.(`[KB Upload] File uploaded successfully. Document ID: ${documentId}`);
  return documentId;
}

export async function triggerVectorization(
  documentId: string,
  apiKey: string,
  onLog?: (message: string) => void
): Promise<void> {
  const url = `${KB_BASE_URL}/document/embedding/${documentId}`;
  onLog?.(`[KB Vectorize] Triggering vectorization for Document ID: ${documentId}`);
  
  const response = await debugFetch(url, {
    method: 'POST',
    headers: {
      'Authorization': `Bearer ${apiKey}`,
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({}), // Empty body, no callback URL for now
  }, onLog, { label: "vectorize", retries: 3 });

  if (!response.ok) {
    const errorText = await response.text();
    onLog?.(`[KB Vectorize] Error: ${errorText}`);
    throw new Error(`Vectorization trigger failed: ${response.status} - ${errorText}`);
  }
  onLog?.('[KB Vectorize] Vectorization triggered successfully.');
}

export async function pollDocumentStatus(
  documentId: string,
  apiKey: string,
  onLog?: (message: string) => void,
  maxPollingTime = 180000, // 3 minutes
  initialInterval = 2000,
  maxInterval = 20000
): Promise<void> {
  const url = `${KB_BASE_URL}/document/${documentId}`;
  const startTime = Date.now();
  let currentInterval = initialInterval;

  onLog?.(`[Doc Status] Starting status polling for Document ID: ${documentId}`);

  while (Date.now() - startTime < maxPollingTime) {
    try {
      const response = await debugFetch(url, {
        method: 'GET',
        headers: {
          'Authorization': `Bearer ${apiKey}`,
        },
      }, onLog, { label: "docStatus", retries: 3 });

      if (!response.ok) {
        onLog?.(`[Doc Status] Status check failed: ${response.status}`);
      } else {
        const data = await response.json();
        const docInfo = data?.data;
        const status = docInfo?.embedding_stat; // 0-pending, 1-success, 2-failed
        onLog?.(`[Doc Status] Current embedding_stat: ${status}`);

        if (status === 1) {
          onLog?.('[Doc Status] Document is ready!');
          return; // Success
        } else if (status === 2) {
          const failureInfo = docInfo?.failInfo || 'No details provided.';
          onLog?.(`[Doc Status] Vectorization failed: ${failureInfo}`);
          throw new Error(`Document processing failed on server: ${failureInfo}`);
        }
      }
    } catch (error) {
      if (error instanceof Error && error.message.includes('Document processing failed')) {
          throw error;
      }
      const message = error instanceof Error ? error.message : String(error);
      onLog?.(`[Doc Status] Polling error: ${message}. Retrying...`);
    }
    
    onLog?.(`[Doc Status] Waiting ${Math.round(currentInterval / 1000)}s for next poll.`);
    await new Promise(resolve => setTimeout(resolve, currentInterval));
    currentInterval = Math.min(currentInterval * 1.5, maxInterval);
  }
  throw new Error('Polling timeout - Document did not become ready within the time limit');
}

export async function processDocument(
  file: File,
  apiKey: string,
  onStatusChange: (status: KbStatus) => void,
  onLog?: (message: string) => void
): Promise<{ kbId: string }> {
  try {
    onStatusChange('registering');
    const kbId = await createKb(apiKey, onLog);
    
    onStatusChange('uploading');
    const documentId = await uploadDocument(file, kbId, apiKey, onLog);
    
    onStatusChange('vectorizing');
    await triggerVectorization(documentId, apiKey, onLog);
    
    onStatusChange('polling');
    await pollDocumentStatus(documentId, apiKey, onLog);
    
    onStatusChange('ready');
    return { kbId };
  } catch (error) {
    onStatusChange('error');
    throw error;
  }
}

export async function retrieveKbChunks(
  topic: string,
  kbId: string,
  apiKey: string,
  onLog?: (message: string) => void
): Promise<string> {
  const API_URL = 'https://open.bigmodel.cn/api/paas/v4/chat/completions';
  onLog?.(`[KB Chunks] Retrieving ground truth from KB ID: ${kbId}`);

  const prompt = `Based on the provided document, extract a comprehensive summary of all key information, facts, statistics, and core concepts related to the topic "${topic}". The output should be a single block of plain text, suitable for providing as context to another AI. Do not add any conversational text or introductions.`;

  const body = {
    model: "glm-4",
    messages: [{ role: "user", content: prompt }],
    tools: [{ type: "retrieval", retrieval: { knowledge_id: kbId } }],
  };

  try {
    const response = await debugFetch(API_URL, {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${apiKey}`,
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(body),
    }, onLog, { label: "retrieveKbChunks", retries: 3 });

    onLog?.(`[KB Chunks] Received response with status: ${response.status}`);

    if (!response.ok) {
      const errorText = await response.text();
      onLog?.(`[KB Chunks] Error: ${errorText}`);
      throw new Error(`Failed to retrieve KB chunks: ${response.status} - ${errorText}`);
    }

    const result = await response.json();
    const content = result.choices[0]?.message?.content;

    if (!content) {
      onLog?.('[KB Chunks] API returned an empty response. Returning empty string.');
      return "";
    }
    
    onLog?.(`[KB Chunks] Successfully retrieved ground truth text.`);
    return content.trim();

  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    onLog?.(`[KB Chunks] Request failed: ${message}`);
    throw error;
  }
}

export async function retrieveGeneratedContent(
  payload: CourseData,
  apiKey: string,
  onLog?: (message: string) => void
): Promise<Array<Partial<GeneralCourseSlide | MicrolearningSlide>>> {
    const API_URL = 'https://open.bigmodel.cn/api/paas/v4/chat/completions';
    onLog?.(`[Content Retrieve] Sending request to BigModel Chat API...`);

    const slidePrompts = payload.slides.map(s => `Slide ${s.id} (${s.contentType}): ${s.autoMode ? (s.userContent ? `Instructions: ${s.userContent}` : 'Generate content for this topic.') : 'IGNORE - This slide content is manually provided.'}`).join('\n');

    const prompt = `You are an expert instructional designer. A document has been uploaded to a knowledge base. You MUST use the information from that document to generate content for a course titled "${payload.courseTopic}".

Here is the slide structure:
${slidePrompts}

For each slide that does NOT say "IGNORE", generate concise and informative content based ONLY on the document. Do not use general knowledge.

Respond with a valid JSON object containing a "slides" array. Each object in the array must have an "id" (matching the slide ID) and a "userContent" field with the generated text.

Example response:
{
  "slides": [
    { "id": 1, "userContent": "This is the generated overview..." },
    { "id": 2, "userContent": "These are the learning objectives..." }
  ]
}
`;
    
    const body = {
      model: "glm-4",
      messages: [{ role: "user", content: prompt }],
      tools: payload.kbId ? [{ type: "retrieval", retrieval: { knowledge_id: payload.kbId } }] : [],
    };

    try {
        const response = await debugFetch(API_URL, {
            method: 'POST',
            headers: {
                'Authorization': `Bearer ${apiKey}`,
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(body),
        }, onLog, { label: "retrieveGeneratedContent", retries: 3 });

        onLog?.(`[Content Retrieve] Received response with status: ${response.status}`);

        if (!response.ok) {
            const errorText = await response.text();
            onLog?.(`[Content Retrieve] Error: ${errorText}`);
            throw new Error(`Content retrieval failed: ${response.status} - ${errorText}`);
        }

        const result = await response.json();
        const content = result.choices[0]?.message?.content;
        
        if (!content) {
          throw new Error("API returned an empty response for content retrieval.");
        }

        onLog?.(`[Content Retrieve] Success. Received raw content.`);
        
        try {
            // Find the JSON part of the response, in case the model adds markdown formatting
            const jsonMatch = content.match(/```json\s*([\s\S]*?)\s*```|({[\s\S]*})/);
            if (!jsonMatch) {
                throw new Error("No JSON object found in the response.");
            }
            const jsonString = jsonMatch[1] || jsonMatch[2];
            const parsed = JSON.parse(jsonString);

            if (!parsed || !Array.isArray(parsed.slides)) {
                 throw new Error("API response did not contain a 'slides' array.");
            }
            return parsed.slides;
        } catch (e) {
            const message = e instanceof Error ? e.message : String(e);
            onLog?.(`[Content Retrieve] Failed to parse JSON response: ${message}. Raw content: ${content}`);
            throw new Error(`Failed to parse JSON from API response: ${message}`);
        }

    } catch (error) {
        const message = error instanceof Error ? error.message : String(error);
        onLog?.(`[Content Retrieve] Request failed: ${message}`);
        throw error;
    }
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