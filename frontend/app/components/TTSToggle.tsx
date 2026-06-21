"use client";

interface TTSToggleProps {
  enabled: boolean;
  onToggle: () => void;
  speaking: boolean;
}

export function TTSToggle({ enabled, onToggle, speaking }: TTSToggleProps) {
  return (
    <button
      type="button"
      onClick={onToggle}
      title={enabled ? "Disable text-to-speech" : "Enable text-to-speech"}
      className={`flex items-center gap-1.5 rounded-full border px-3 py-1 text-xs transition ${
        enabled
          ? "border-blue-600 bg-blue-50 text-blue-700"
          : "border-zinc-300 bg-white text-zinc-600 hover:bg-zinc-50"
      }`}
    >
      <svg
        className="h-3.5 w-3.5"
        viewBox="0 0 24 24"
        fill="none"
        stroke="currentColor"
        strokeWidth={2}
        strokeLinecap="round"
        strokeLinejoin="round"
      >
        <polygon points="11 5 6 9 2 9 2 15 6 15 11 19 11 5" />
        {enabled && (
          <>
            <path d="M19.07 4.93a10 10 0 010 14.14" />
            <path d="M15.54 8.46a5 5 0 010 7.07" />
          </>
        )}
      </svg>
      {speaking ? (
        <span className="flex items-center gap-1">
          Speaking
          <span className="inline-block h-1.5 w-1.5 animate-pulse rounded-full bg-blue-600" />
        </span>
      ) : enabled ? (
        "TTS on"
      ) : (
        "TTS"
      )}
    </button>
  );
}
