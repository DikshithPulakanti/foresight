"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { api } from "@/lib/api";

interface VoiceButtonProps {
  userId: string;
  onResult: (transcript: string, intent: string) => void;
}

export default function VoiceButton({ userId, onResult }: VoiceButtonProps) {
  const [isListening, setIsListening] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<{ transcript: string; intent: string } | null>(null);
  const [showResult, setShowResult] = useState(false);

  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const chunksRef = useRef<Blob[]>([]);
  const timeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  const stopRecording = useCallback(() => {
    if (mediaRecorderRef.current && mediaRecorderRef.current.state === "recording") {
      mediaRecorderRef.current.stop();
    }
    if (timeoutRef.current) {
      clearTimeout(timeoutRef.current);
      timeoutRef.current = null;
    }
    setIsListening(false);
  }, []);

  const processAudio = useCallback(
    async (blob: Blob) => {
      setIsProcessing(true);
      setError(null);
      try {
        const buffer = await blob.arrayBuffer();
        const bytes = new Uint8Array(buffer);
        let binary = "";
        for (let i = 0; i < bytes.length; i++) {
          binary += String.fromCharCode(bytes[i]);
        }
        const audio_base64 = btoa(binary);

        const res = await api.runAgent("voice", {
          user_id: userId,
          audio_base64,
          audio_format: "webm",
        });

        const transcript = (res.output.transcript as string) ?? "";
        const intent = (res.output.intent as string) ?? "";
        setResult({ transcript, intent });
        setShowResult(true);
        onResult(transcript, intent);
      } catch (err) {
        setError(err instanceof Error ? err.message : "Voice processing failed");
      } finally {
        setIsProcessing(false);
      }
    },
    [userId, onResult],
  );

  const startRecording = useCallback(async () => {
    setError(null);
    setResult(null);
    setShowResult(false);
    chunksRef.current = [];

    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const recorder = new MediaRecorder(stream, { mimeType: "audio/webm" });
      mediaRecorderRef.current = recorder;

      recorder.ondataavailable = (e) => {
        if (e.data.size > 0) chunksRef.current.push(e.data);
      };

      recorder.onstop = () => {
        stream.getTracks().forEach((t) => t.stop());
        const blob = new Blob(chunksRef.current, { type: "audio/webm" });
        if (blob.size > 0) processAudio(blob);
      };

      recorder.start();
      setIsListening(true);

      timeoutRef.current = setTimeout(() => {
        stopRecording();
      }, 5000);
    } catch {
      setError("Microphone access denied");
    }
  }, [processAudio, stopRecording]);

  const handleClick = useCallback(() => {
    if (isListening) {
      stopRecording();
    } else if (!isProcessing) {
      startRecording();
    }
  }, [isListening, isProcessing, startRecording, stopRecording]);

  useEffect(() => {
    return () => {
      if (timeoutRef.current) clearTimeout(timeoutRef.current);
      if (mediaRecorderRef.current?.state === "recording") {
        mediaRecorderRef.current.stop();
      }
    };
  }, []);

  return (
    <div className="relative inline-flex flex-col items-center">
      {/* Pulse ring animation */}
      {isListening && (
        <span className="absolute inset-0 rounded-full bg-red-500 animate-[pulse-ring_1.5s_ease-out_infinite]" />
      )}

      <button
        onClick={handleClick}
        disabled={isProcessing}
        className={`relative z-10 flex h-14 w-14 items-center justify-center rounded-full transition-all duration-200 shadow-lg disabled:cursor-wait ${
          isListening
            ? "bg-red-500 hover:bg-red-400 scale-110"
            : isProcessing
              ? "bg-gray-700"
              : "bg-emerald-600 hover:bg-emerald-500"
        }`}
        aria-label={isListening ? "Stop recording" : "Start voice input"}
      >
        {isProcessing ? (
          <svg className="w-6 h-6 text-white animate-spin" viewBox="0 0 24 24" fill="none">
            <circle cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="3" className="opacity-25" />
            <path d="M4 12a8 8 0 018-8" stroke="currentColor" strokeWidth="3" strokeLinecap="round" />
          </svg>
        ) : (
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={1.5} className="w-6 h-6 text-white">
            <path strokeLinecap="round" strokeLinejoin="round" d="M12 18.75a6 6 0 0 0 6-6v-1.5m-6 7.5a6 6 0 0 1-6-6v-1.5m6 7.5v3.75m-3.75 0h7.5M12 15.75a3 3 0 0 1-3-3V4.5a3 3 0 1 1 6 0v8.25a3 3 0 0 1-3 3Z" />
          </svg>
        )}
      </button>

      <span className="mt-2 text-xs text-gray-500">
        {isListening ? "Listening…" : isProcessing ? "Processing…" : "Tap to speak"}
      </span>

      {/* Result popup */}
      {showResult && result && (
        <div className="absolute top-full mt-3 w-64 rounded-lg bg-gray-800 border border-gray-700 p-3 shadow-xl z-20">
          <div className="flex items-center justify-between mb-2">
            <span className="text-xs font-medium text-gray-400">Voice Result</span>
            <button
              onClick={() => setShowResult(false)}
              className="text-gray-500 hover:text-white transition-colors"
            >
              <svg viewBox="0 0 20 20" fill="currentColor" className="w-3.5 h-3.5">
                <path d="M6.28 5.22a.75.75 0 0 0-1.06 1.06L8.94 10l-3.72 3.72a.75.75 0 1 0 1.06 1.06L10 11.06l3.72 3.72a.75.75 0 1 0 1.06-1.06L11.06 10l3.72-3.72a.75.75 0 0 0-1.06-1.06L10 8.94 6.28 5.22Z" />
              </svg>
            </button>
          </div>
          <p className="text-sm text-white mb-1">&ldquo;{result.transcript}&rdquo;</p>
          <p className="text-xs text-emerald-400">Intent: {result.intent}</p>
        </div>
      )}

      {/* Error */}
      {error && (
        <div className="absolute top-full mt-3 w-56 rounded-lg bg-red-900/80 border border-red-800 p-2.5 text-xs text-red-300 text-center z-20">
          {error}
        </div>
      )}
    </div>
  );
}
