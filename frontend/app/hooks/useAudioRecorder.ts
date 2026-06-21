import { useCallback, useRef, useState } from "react";

type RecorderStatus = "idle" | "requesting" | "recording" | "processing";

const STT_URL =
  process.env.NEXT_PUBLIC_STT_URL || "http://localhost:8003";

interface UseAudioRecorderOptions {
  onTranscription: (text: string) => void;
}

export function useAudioRecorder({ onTranscription }: UseAudioRecorderOptions) {
  const [status, setStatus] = useState<RecorderStatus>("idle");
  const [error, setError] = useState<string | null>(null);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const chunksRef = useRef<Blob[]>([]);

  const start = useCallback(async () => {
    setError(null);
    setStatus("requesting");

    let stream: MediaStream;
    try {
      stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    } catch {
      setError("Microphone permission denied");
      setStatus("idle");
      return;
    }

    chunksRef.current = [];
    const recorder = new MediaRecorder(stream, { mimeType: "audio/webm" });
    mediaRecorderRef.current = recorder;

    recorder.ondataavailable = (e) => {
      if (e.data.size > 0) chunksRef.current.push(e.data);
    };

    recorder.onstop = async () => {
      stream.getTracks().forEach((t) => t.stop());
      const blob = new Blob(chunksRef.current, { type: "audio/webm" });
      chunksRef.current = [];

      if (blob.size === 0) {
        setError("No audio recorded");
        setStatus("idle");
        return;
      }

      setStatus("processing");

      try {
        const form = new FormData();
        form.append("file", blob, "recording.webm");
        const res = await fetch(`${STT_URL}/transcribe`, {
          method: "POST",
          body: form,
        });
        if (!res.ok) {
          const detail = await res.text();
          throw new Error(detail || `STT error ${res.status}`);
        }
        const data = await res.json();
        if (data.text) {
          onTranscription(data.text);
        }
      } catch (err) {
        const message =
          err instanceof TypeError
            ? "STT service unavailable"
            : (err as Error).message || "Transcription failed";
        setError(message);
      } finally {
        setStatus("idle");
      }
    };

    recorder.start();
    setStatus("recording");
  }, [onTranscription]);

  const stop = useCallback(() => {
    if (mediaRecorderRef.current?.state === "recording") {
      mediaRecorderRef.current.stop();
    }
  }, []);

  return { status, error, start, stop };
}
