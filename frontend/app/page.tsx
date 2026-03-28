"use client";

import { useEffect, useRef, useState } from "react";

type Message = {
  id: string;
  role: "user" | "assistant" | "system";
  content: string;
};

type SourceItem = {
  title: string;
  url: string;
  snippet: string;
};

type MemoryMode = "off" | "session" | "long_term";

const API_BASE_URL = process.env.NEXT_PUBLIC_API_BASE_URL || "http://localhost:8000";

const STREAM_ENDPOINT = "/chat/stream";

export default function Home() {
  const [messages, setMessages] = useState<Message[]>([
    {
      id: "welcome",
      role: "assistant",
      content: "Ask me anything and I will stream the answer as it arrives.",
    },
  ]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [sources, setSources] = useState<SourceItem[]>([]);
  const [memoryMode, setMemoryMode] = useState<MemoryMode>("off");
  const [sessionId] = useState(() =>
    typeof crypto !== "undefined" && "randomUUID" in crypto
      ? crypto.randomUUID()
      : `${Date.now()}-${Math.random().toString(16).slice(2)}`,
  );
  const abortRef = useRef<AbortController | null>(null);
  const bottomRef = useRef<HTMLDivElement | null>(null);
  const streamedTokenRef = useRef(false);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [messages]);

  useEffect(() => {
    return () => {
      abortRef.current?.abort();
    };
  }, []);

  const appendAssistantText = (text: string) => {
    setMessages((prev) => {
      const last = prev[prev.length - 1];
      if (last?.role === "assistant" && last.id.startsWith("stream-")) {
        return [
          ...prev.slice(0, -1),
          { ...last, content: last.content + text },
        ];
      }
      return [
        ...prev,
        { id: `stream-${Date.now()}`, role: "assistant", content: text },
      ];
    });
  };

  const handleSubmit = async (event: React.FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    const trimmed = input.trim();
    if (!trimmed || loading) return;

    setError(null);
    setSources([]);
    setMessages((prev) => [
      ...prev,
      { id: `user-${Date.now()}`, role: "user", content: trimmed },
    ]);
    setInput("");
    setLoading(true);
    streamedTokenRef.current = false;

    abortRef.current?.abort();
    const controller = new AbortController();
    abortRef.current = controller;

    try {
      const response = await fetch(`${API_BASE_URL}${STREAM_ENDPOINT}`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          query: trimmed,
          max_results: 5,
          memory_mode: memoryMode,
          session_id: sessionId,
        }),
        signal: controller.signal,
      });

      if (!response.ok || !response.body) {
        setError(`Streaming failed (${response.status})`);
        setLoading(false);
        return;
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";

      const handleEvent = (eventType: string, data: string) => {
        if (!data) return;
        try {
          const payload = JSON.parse(data);
          if (eventType === "token") {
            streamedTokenRef.current = true;
            appendAssistantText(payload.text || "");
          } else if (eventType === "sources") {
            setSources(payload.sources || []);
          } else if (eventType === "done") {
            if (payload.answer && !streamedTokenRef.current) {
              appendAssistantText(payload.answer);
            }
            setLoading(false);
          } else if (eventType === "error") {
            setError(payload.message || "Streaming error.");
            setLoading(false);
          }
        } catch {
          setError("Failed to parse stream response.");
          setLoading(false);
        }
      };

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        const events = buffer.split("\n\n");
        buffer = events.pop() || "";
        for (const raw of events) {
          const lines = raw.split("\n");
          let eventType = "";
          let data = "";
          for (const line of lines) {
            if (line.startsWith("event:")) {
              eventType = line.replace("event:", "").trim();
            } else if (line.startsWith("data:")) {
              data += line.replace("data:", "").trim();
            }
          }
          if (eventType) {
            handleEvent(eventType, data);
          }
        }
      }
    } catch (err) {
      if ((err as DOMException).name !== "AbortError") {
        setError("Streaming connection failed.");
      }
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-white text-black">
      <main className="mx-auto flex w-full max-w-4xl flex-col px-6 py-10">
        <header className="mb-8">
          <p className="text-sm uppercase tracking-[0.2em] text-blue-600">
            Agent Tools Chat
          </p>
          <h1 className="mt-3 text-3xl font-semibold text-black">
            Research assistant
          </h1>
        </header>

        <section className="flex h-[70vh] flex-col rounded-3xl border border-zinc-200 bg-white p-6 shadow-[0_20px_60px_-40px_rgba(37,99,235,0.25)]">
          <div className="flex-1 space-y-4 overflow-y-auto pr-2">
            {messages.map((message) => (
              <div
                key={message.id}
                className={`flex ${
                  message.role === "user" ? "justify-end" : "justify-start"
                }`}
              >
                <div
                  className={`max-w-[80%] rounded-2xl px-4 py-3 text-sm leading-relaxed ${
                    message.role === "user"
                      ? "bg-blue-600 text-white"
                      : "bg-zinc-100 text-black"
                  }`}
                >
                  {message.content}
                </div>
              </div>
            ))}
            {loading && (
              <div className="text-xs text-blue-600">Generating response…</div>
            )}
            {error && <div className="text-xs text-rose-600">{error}</div>}
            <div ref={bottomRef} />
          </div>

          <form
            onSubmit={handleSubmit}
            className="mt-6 flex flex-col gap-3 border-t border-zinc-200 pt-4"
          >
            <div className="flex flex-wrap items-center gap-3 text-xs text-zinc-600">
              <span className="font-semibold text-zinc-700">Memory mode</span>
              {(["off", "session", "long_term"] as MemoryMode[]).map((mode) => (
                <label
                  key={mode}
                  className={`flex items-center gap-2 rounded-full border px-3 py-1 ${
                    memoryMode === mode
                      ? "border-blue-600 bg-blue-50 text-blue-700"
                      : "border-zinc-300 bg-white text-zinc-600"
                  }`}
                >
                  <input
                    type="radio"
                    name="memory-mode"
                    value={mode}
                    checked={memoryMode === mode}
                    onChange={() => setMemoryMode(mode)}
                    className="h-3 w-3 accent-blue-600"
                  />
                  {mode}
                </label>
              ))}
            </div>
            <textarea
              value={input}
              onChange={(event) => setInput(event.target.value)}
              placeholder="Ask a question about anything…"
              className="min-h-[96px] resize-none rounded-2xl border border-zinc-300 bg-white px-4 py-3 text-sm text-black placeholder:text-zinc-400 focus:border-blue-500 focus:outline-none"
            />
            <div className="flex items-center justify-between text-xs text-zinc-500">
              <span>{input.trim().length} characters</span>
              <button
                type="submit"
                disabled={loading}
                className="rounded-full bg-blue-600 px-5 py-2 text-xs font-semibold uppercase tracking-wide text-white transition hover:bg-blue-500 disabled:cursor-not-allowed disabled:opacity-50"
              >
                {loading ? "Sending…" : "Send"}
              </button>
            </div>
          </form>
        </section>

        {sources.length > 0 && (
          <section className="mt-6 rounded-2xl border border-zinc-200 bg-white p-5">
            <h2 className="text-sm font-semibold text-blue-600">
              Sources
            </h2>
            <ul className="mt-3 space-y-3 text-sm text-zinc-700">
              {sources.map((source, idx) => (
                <li key={`${source.url}-${idx}`} className="space-y-1">
                  <p className="font-semibold text-black">{source.title}</p>
                  <p className="text-xs text-blue-600">{source.url}</p>
                  <p className="text-xs text-zinc-500">{source.snippet}</p>
                </li>
              ))}
            </ul>
          </section>
        )}
      </main>
    </div>
  );
}
