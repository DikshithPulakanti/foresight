"use client";

import type { Alert } from "@/types";

const SEVERITY_COLORS: Record<string, string> = {
  critical: "border-red-500",
  high: "border-orange-500",
  medium: "border-yellow-500",
  low: "border-blue-500",
  info: "border-gray-500",
};

const SEVERITY_BADGES: Record<string, string> = {
  critical: "bg-red-500/10 text-red-400",
  high: "bg-orange-500/10 text-orange-400",
  medium: "bg-yellow-500/10 text-yellow-400",
  low: "bg-blue-500/10 text-blue-400",
  info: "bg-gray-500/10 text-gray-400",
};

function timeAgo(dateString: string): string {
  const seconds = Math.floor(
    (Date.now() - new Date(dateString).getTime()) / 1000,
  );
  if (seconds < 60) return "just now";
  const minutes = Math.floor(seconds / 60);
  if (minutes < 60) return `${minutes}m ago`;
  const hours = Math.floor(minutes / 60);
  if (hours < 24) return `${hours}h ago`;
  const days = Math.floor(hours / 24);
  return `${days}d ago`;
}

interface AlertCardProps {
  alert: Alert;
  onDismiss?: (id: string) => void;
}

export default function AlertCard({ alert, onDismiss }: AlertCardProps) {
  const borderColor = SEVERITY_COLORS[alert.severity] ?? SEVERITY_COLORS.info;
  const badgeColor = SEVERITY_BADGES[alert.severity] ?? SEVERITY_BADGES.info;

  return (
    <div
      className={`relative rounded-lg border-l-4 ${borderColor} bg-gray-900 px-4 py-3`}
    >
      <div className="flex items-start justify-between gap-3">
        <div className="min-w-0 flex-1">
          <div className="flex items-center gap-2 mb-1">
            <span
              className={`inline-flex items-center rounded-full px-2 py-0.5 text-xs font-medium ${badgeColor}`}
            >
              {alert.severity}
            </span>
            <span className="text-xs text-gray-500">
              {timeAgo(alert.created_at)}
            </span>
          </div>
          <h3 className="text-sm font-medium text-white truncate">
            {alert.title}
          </h3>
          <p className="mt-0.5 text-sm text-gray-400 line-clamp-2">
            {alert.message}
          </p>
        </div>

        {onDismiss && (
          <button
            onClick={() => onDismiss(alert.id)}
            className="shrink-0 rounded p-1 text-gray-500 hover:bg-gray-800 hover:text-gray-300 transition-colors"
            aria-label="Dismiss alert"
          >
            <svg viewBox="0 0 20 20" fill="currentColor" className="w-4 h-4">
              <path d="M6.28 5.22a.75.75 0 0 0-1.06 1.06L8.94 10l-3.72 3.72a.75.75 0 1 0 1.06 1.06L10 11.06l3.72 3.72a.75.75 0 1 0 1.06-1.06L11.06 10l3.72-3.72a.75.75 0 0 0-1.06-1.06L10 8.94 6.28 5.22Z" />
            </svg>
          </button>
        )}
      </div>

      {alert.amount !== undefined && (
        <div className="mt-2 text-xs font-medium text-gray-500">
          Amount: ${alert.amount.toLocaleString()}
        </div>
      )}
    </div>
  );
}
