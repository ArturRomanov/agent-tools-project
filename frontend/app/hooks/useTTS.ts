import { useCallback, useRef, useState } from "react";

const TTS_URL =
  process.env.NEXT_PUBLIC_TTS_URL || "http://localhost:8004";

export function useTTS() {
  const [ttsEnabled, setTtsEnabled] = useState(false);
  const [speaking, setSpeaking] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const audioRef = useRef<HTMLAudioElement | null>(null);
  const abortRef = useRef<AbortController | null>(null);
  const audioCtxRef = useRef<AudioContext | null>(null);

  const speak = useCallback(async (text: string) => {
    setError(null);

    abortRef.current?.abort();
    const controller = new AbortController();
    abortRef.current = controller;

    setSpeaking(true);

    try {
      const res = await fetch(`${TTS_URL}/synthesize`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text }),
        signal: controller.signal,
      });

      if (!res.ok) {
        const detail = await res.text();
        throw new Error(detail || `TTS error ${res.status}`);
      }

      const blob = await res.blob();
      const url = URL.createObjectURL(blob);

      const audio = new Audio(url);
      audioRef.current = audio;

      audio.onended = () => {
        setSpeaking(false);
        URL.revokeObjectURL(url);
      };
      audio.onerror = () => {
        setSpeaking(false);
        URL.revokeObjectURL(url);
        setError("Audio playback failed");
      };

      await audio.play();
    } catch (err) {
      if ((err as DOMException).name === "AbortError") {
        setSpeaking(false);
        return;
      }
      const message =
        err instanceof TypeError
          ? "TTS service unavailable"
          : (err as Error).message || "Speech synthesis failed";
      setError(message);
      setSpeaking(false);
    }
  }, []);

  const speakStream = useCallback(async (text: string) => {
    setError(null);

    abortRef.current?.abort();
    const controller = new AbortController();
    abortRef.current = controller;

    try {
      const res = await fetch(`${TTS_URL}/synthesize/stream`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text }),
        signal: controller.signal,
      });

      if (!res.ok) {
        const detail = await res.text();
        throw new Error(detail || `TTS error ${res.status}`);
      }

      if (!res.body) {
        throw new Error("No response body for streaming");
      }

      const audioCtx = new AudioContext();
      audioCtxRef.current = audioCtx;
      let nextStartTime = audioCtx.currentTime;
      let lastSource: AudioBufferSourceNode | null = null;
      let totalChunks = 0;
      let chunksPlayed = 0;

      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });

        const lines = buffer.split("\n");
        buffer = lines.pop() || "";

        for (const line of lines) {
          const trimmed = line.trim();
          if (!trimmed) continue;

          const parsed = JSON.parse(trimmed);
          if (parsed.done) continue;

          totalChunks = parsed.total;
          const binaryStr = atob(parsed.audio);
          const bytes = new Uint8Array(binaryStr.length);
          for (let i = 0; i < binaryStr.length; i++) {
            bytes[i] = binaryStr.charCodeAt(i);
          }

          const audioBuffer = await audioCtx.decodeAudioData(
            bytes.buffer.slice(0)
          );
          const source = audioCtx.createBufferSource();
          source.buffer = audioBuffer;
          source.connect(audioCtx.destination);

          if (nextStartTime < audioCtx.currentTime) {
            nextStartTime = audioCtx.currentTime;
          }

          source.start(nextStartTime);
          nextStartTime += audioBuffer.duration;

          if (!lastSource) {
            setSpeaking(true);
          }

          lastSource = source;
          const chunkIndex = parsed.index;
          source.onended = () => {
            chunksPlayed++;
            if (chunksPlayed >= totalChunks) {
              setSpeaking(false);
              audioCtxRef.current = null;
            }
          };
        }
      }

      if (!lastSource) {
        setSpeaking(false);
      }
    } catch (err) {
      if ((err as DOMException).name === "AbortError") {
        setSpeaking(false);
        audioCtxRef.current?.close();
        audioCtxRef.current = null;
        return;
      }
      const message =
        err instanceof TypeError
          ? "TTS service unavailable"
          : (err as Error).message || "Speech synthesis failed";
      setError(message);
      setSpeaking(false);
      audioCtxRef.current?.close();
      audioCtxRef.current = null;
    }
  }, []);

  const stop = useCallback(() => {
    abortRef.current?.abort();
    if (audioRef.current) {
      audioRef.current.pause();
      audioRef.current = null;
    }
    if (audioCtxRef.current) {
      audioCtxRef.current.close();
      audioCtxRef.current = null;
    }
    setSpeaking(false);
  }, []);

  const toggle = useCallback(() => {
    setTtsEnabled((prev) => {
      if (prev) {
        // Turning off — stop any current playback
        abortRef.current?.abort();
        if (audioRef.current) {
          audioRef.current.pause();
          audioRef.current = null;
        }
        if (audioCtxRef.current) {
          audioCtxRef.current.close();
          audioCtxRef.current = null;
        }
        setSpeaking(false);
      }
      return !prev;
    });
    setError(null);
  }, []);

  return { ttsEnabled, speaking, error, speak, speakStream, stop, toggle };
}
