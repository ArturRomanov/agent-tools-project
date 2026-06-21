"use client";

type RecorderStatus = "idle" | "requesting" | "recording" | "processing";

interface MicButtonProps {
  status: RecorderStatus;
  onStart: () => void;
  onStop: () => void;
  disabled?: boolean;
}

export function MicButton({ status, onStart, onStop, disabled }: MicButtonProps) {
  const isRecording = status === "recording";
  const isProcessing = status === "processing";
  const isRequesting = status === "requesting";
  const busy = isProcessing || isRequesting;

  return (
    <button
      type="button"
      onClick={isRecording ? onStop : onStart}
      disabled={disabled || busy}
      title={
        isRecording
          ? "Stop recording"
          : isProcessing
            ? "Transcribing..."
            : "Start recording"
      }
      className={`relative flex h-8 w-8 items-center justify-center rounded-full transition ${
        isRecording
          ? "bg-rose-600 text-white"
          : "bg-zinc-100 text-zinc-600 hover:bg-zinc-200"
      } disabled:cursor-not-allowed disabled:opacity-50`}
    >
      {isRecording && (
        <span className="absolute inset-0 animate-ping rounded-full bg-rose-400 opacity-40" />
      )}
      {isProcessing ? (
        <svg
          className="h-4 w-4 animate-spin"
          viewBox="0 0 24 24"
          fill="none"
          stroke="currentColor"
          strokeWidth={2}
        >
          <circle cx="12" cy="12" r="10" className="opacity-25" />
          <path d="M4 12a8 8 0 018-8" className="opacity-75" />
        </svg>
      ) : (
        <svg
          className="relative h-4 w-4"
          viewBox="0 0 24 24"
          fill="none"
          stroke="currentColor"
          strokeWidth={2}
          strokeLinecap="round"
          strokeLinejoin="round"
        >
          <rect x="9" y="1" width="6" height="12" rx="3" />
          <path d="M19 10v1a7 7 0 01-14 0v-1" />
          <line x1="12" y1="19" x2="12" y2="23" />
          <line x1="8" y1="23" x2="16" y2="23" />
        </svg>
      )}
    </button>
  );
}
