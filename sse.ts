// sse.ts
export type SSEEvent = { event: string; data: string; id?: string; retry?: number };

export function createSSEParser(onEvent: (e: SSEEvent) => void) {
  let buffer = "";
  let ev: SSEEvent = { event: "message", data: "" };

  function dispatch() {
    if (ev.data === "" && !ev.id && !ev.retry) return;
    onEvent({ ...ev });
    ev = { event: "message", data: "" };
  }

  return function feed(chunk: string, opts: { flush?: boolean } = {}) {
    buffer += chunk;
    let i = 0;
    while (true) {
      const nl = buffer.indexOf("\n", i);
      if (nl === -1) break;
      let line = buffer.slice(i, nl);
      if (line.endsWith("\r")) line = line.slice(0, -1);
      i = nl + 1;

      if (line === "") { dispatch(); continue; }
      const sep = line.indexOf(":");
      const field = sep === -1 ? line : line.slice(0, sep);
      let val = sep === -1 ? "" : line.slice(sep + 1);
      if (val.startsWith(" ")) val = val.slice(1);

      if (field === "event") ev.event = val;
      else if (field === "data") ev.data += (ev.data ? "\n" : "") + val;   // <- multi-line data
      else if (field === "id") ev.id = val;
      else if (field === "retry") ev.retry = Number(val) || undefined;
    }
    buffer = buffer.slice(i);

    // final flush if stream ends without blank line
    if (opts.flush && buffer) {
      ev.data += (ev.data ? "\n" : "") + buffer;
      buffer = "";
      dispatch();
    }
  };
}
