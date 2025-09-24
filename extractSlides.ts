// extractSlides.ts

export type OnPartialHtml = (pos: number, html: string, complete: boolean) => void;

// Represents the state for a single slide-generation stream.
export interface StreamExtractionState {
    toolState: Record<number, { buf: string; lastLen: number; finalized: boolean; }>;
    assistantBuf: string;
}

export function createExtractionState(): StreamExtractionState {
    return {
        toolState: {},
        assistantBuf: "",
    };
}

function upsertPos(state: StreamExtractionState, pos: number) {
    return (state.toolState[pos] ||= { buf: "", lastLen: 0, finalized: false });
}

function stripFences(s: string) {
    return s.replace(/^```(?:html)?\s*/i, "").replace(/\s*```$/i, "");
}

function maybeDecodeJsonString(s: string) {
    // If model accidentally stringified HTML fragments (\" \\n)
    if (/\\n|\\t|\\"/.test(s) && !/<[a-z!/]/i.test(s)) {
        try {
            // A string that is JSON-encoded will be wrapped in quotes.
            // e.g. the raw value is "<p>Hi</p>"
            // JSON.parse `"` + raw_value + `"` -> <p>Hi</p>
            return JSON.parse(`"${s}"`);
        } catch { /* fallthrough */ }
        // Fallback for simple cases if JSON parsing fails
        return s.replace(/\\n/g, "\n").replace(/\\"/g, "\"");
    }
    return s;
}

function onToolChunk(state: StreamExtractionState, pos: number, raw: string, log: (s: string) => void, onPartial?: OnPartialHtml) {
    const s = upsertPos(state, pos);

    if (s.finalized) {
        log(`[tool] pos=${pos} is finalized. Ignoring chunk.`);
        return;
    }
    
    const cleanedRaw = maybeDecodeJsonString(stripFences(raw));
    
    if (/^\s*<!doctype html/i.test(cleanedRaw) || /^\s*<html[\s>]/i.test(cleanedRaw)) {
        s.buf = cleanedRaw;
        s.lastLen = cleanedRaw.length;
        log(`[tool] pos=${pos} reset start (len=${s.lastLen})`);
    }
    else if (cleanedRaw.length > s.lastLen) {
        const tail = cleanedRaw.slice(s.lastLen);
        if (tail) {
            s.buf += tail;
            s.lastLen = cleanedRaw.length;
            log(`[tool] pos=${pos} +${tail.length} (total=${s.buf.length})`);
        }
    } else if (cleanedRaw.length < s.lastLen && cleanedRaw.length > 0) {
        s.buf += cleanedRaw;
        s.lastLen = s.buf.length;
        log(`[tool] pos=${pos} non-monotonic, appended raw (total=${s.buf.length})`);
    } else if (cleanedRaw.length > 0 && s.lastLen === 0) {
         s.buf += cleanedRaw;
        s.lastLen = cleanedRaw.length;
    }

    const complete = /<\/html>\s*$/i.test(s.buf);
    if (onPartial) {
        onPartial(pos, s.buf, complete);
    }

    if (complete) {
        s.finalized = true;
        log(`[tool] pos=${pos} finalized with </html> tag.`);
    }
}

const asArray = <T>(x: T | T[] | null | undefined): T[] =>
    x == null ? [] : Array.isArray(x) ? x : [x];

export function walkEvent(state: StreamExtractionState, obj: any, log: (s: string) => void, onPartial?: OnPartialHtml) {
    for (const choice of asArray(obj.choices)) {
        for (const msg of asArray(choice.messages ?? choice.message)) {
            log(`[SSE] phase=${msg?.phase ?? "?"}`);
            for (const part of asArray(msg.content)) {
                if (part?.type === "object" && part.object) {
                    const tool = String(part.object.tool_name || part.object.name || "").toLowerCase();
                    const isAdd = /insert|add/.test(tool) && /page|slide/.test(tool);
                    if (isAdd) {
                        const pos = Number(part.object.position?.[0]) || Number(part.object.page_index) || 1;
                        const raw = part.object.output_delta ?? part.object.delta ?? part.object.output ?? part.object.html ?? "";
                        if (typeof raw === "string" && raw) {
                            onToolChunk(state, pos, raw, log, onPartial);
                        }
                    }
                }
                if (part?.type === "text" && typeof part.text === "string") {
                    state.assistantBuf += part.text;
                }
            }
        }
    }
    const delta = obj?.output_text?.delta ?? obj?.delta ?? "";
    if (typeof delta === "string" && delta) state.assistantBuf += delta;
}

const HTML_DOC_RE = /<!doctype html[\s\S]*?<\/html>/i;
const HTML_TAG_RE = /<html[\s\S]*?<\/html>/i;

function cutFirstHtmlDoc(s: string): string | null {
    const m = HTML_DOC_RE.exec(s) ?? HTML_TAG_RE.exec(s);
    return m ? m[0].trim() : null;
}

export function finalizeSlides(state: StreamExtractionState): { position: number; html: string }[] {
    const entries = Object.entries(state.toolState)
        .map(([pos, st]) => [Number(pos), st.buf] as const)
        .filter(([, buf]) => buf && buf.trim().length > 0)
        .sort((a, b) => a[0] - b[0]);

    const slides: { position: number; html: string }[] = [];
    for (const [pos, rawBuf] of entries) {
        // tolerate accidental markdown fences
        const buf = rawBuf.replace(/^```(?:html)?\s*/i, "").replace(/\s*```$/i, "");
        const html = cutFirstHtmlDoc(buf) ?? `<!doctype html><html><head><meta charset="utf-8"><title>Slide ${pos}</title></head><body>${buf}</body></html>`;
        slides.push({ position: pos, html });
    }

    // Fallback: if no tool slides, try assistant buffer (for raw HTML-in-text runs)
    if (!slides.length) {
        // tolerate accidental markdown fences
        const buf = state.assistantBuf.replace(/^```(?:html)?\s*/i, "").replace(/\s*```$/i, "");
        const html = cutFirstHtmlDoc(buf);
        if (html) slides.push({ position: 1, html });
    }

    return slides;
}