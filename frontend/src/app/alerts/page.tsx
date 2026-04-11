"use client";

import { useCallback, useEffect, useMemo, useState } from "react";
import AlertCard from "@/components/ui/AlertCard";
import { api } from "@/lib/api";
import type { Alert } from "@/types";

const USER_ID = "demo-user-123";

const MOCK_ALERTS: Alert[] = [
  { id: "a1", type: "cashflow_risk", title: "Balance dropping fast", message: "Your balance may fall below $500 in 12 days based on current spending.", severity: "high", created_at: new Date().toISOString() },
  { id: "a2", type: "subscription", title: "Netflix price increase", message: "Netflix is raising your price from $15.99 to $22.99 on May 1st.", severity: "medium", created_at: new Date().toISOString() },
  { id: "a3", type: "unusual_transaction", title: "Large purchase flagged", message: "$312 at Best Buy — 3x your normal electronics spending.", severity: "medium", created_at: new Date().toISOString() },
  { id: "a4", type: "goal_achieved", title: "Laptop goal almost complete!", message: "You're 92% of the way to your New Laptop goal. $150 to go!", severity: "low", created_at: new Date().toISOString() },
];

const SEVERITY_ORDER: Alert["severity"][] = ["critical", "high", "medium", "low", "info"];

const SEVERITY_HEADER_STYLES: Record<string, string> = {
  critical: "text-red-400",
  high: "text-orange-400",
  medium: "text-yellow-400",
  low: "text-blue-400",
  info: "text-gray-400",
};

type FilterValue = "all" | "critical" | "high" | "medium" | "low";

export default function AlertsPage() {
  const [alerts, setAlerts] = useState<Alert[]>([]);
  const [loading, setLoading] = useState(true);
  const [filter, setFilter] = useState<FilterValue>("all");
  const [dismissed, setDismissed] = useState<Set<string>>(new Set());
  const [runningAgent, setRunningAgent] = useState(false);

  useEffect(() => {
    async function load() {
      try {
        const data = await api.getAlerts(USER_ID);
        setAlerts(data.length > 0 ? (data as unknown as Alert[]) : MOCK_ALERTS);
      } catch {
        setAlerts(MOCK_ALERTS);
      } finally {
        setLoading(false);
      }
    }
    load();
  }, []);

  const visibleAlerts = useMemo(() => {
    let result = alerts.filter((a) => !dismissed.has(a.id));
    if (filter !== "all") {
      result = result.filter((a) => a.severity === filter);
    }
    return result;
  }, [alerts, filter, dismissed]);

  const grouped = useMemo(() => {
    const map = new Map<string, Alert[]>();
    for (const sev of SEVERITY_ORDER) {
      map.set(sev, []);
    }
    for (const alert of visibleAlerts) {
      const list = map.get(alert.severity) ?? [];
      list.push(alert);
      map.set(alert.severity, list);
    }
    return map;
  }, [visibleAlerts]);

  const unreadCount = visibleAlerts.length;

  const handleDismiss = useCallback((id: string) => {
    setDismissed((prev) => new Set(prev).add(id));
  }, []);

  const markAllRead = useCallback(() => {
    setDismissed(new Set(alerts.map((a) => a.id)));
  }, [alerts]);

  const runSentinel = useCallback(async () => {
    setRunningAgent(true);
    try {
      const res = await api.runAgent("alert-sentinel", { user_id: USER_ID });
      const newAlerts = (res.output.alerts ?? []) as Alert[];
      if (newAlerts.length > 0) {
        setAlerts((prev) => {
          const existing = new Set(prev.map((a) => a.id));
          const merged = [...prev];
          for (const a of newAlerts) {
            if (!existing.has(a.id)) merged.unshift(a);
          }
          return merged;
        });
      }
    } catch (err) {
      console.error("Alert sentinel failed:", err);
    } finally {
      setRunningAgent(false);
    }
  }, []);

  const FILTER_OPTIONS: { value: FilterValue; label: string }[] = [
    { value: "all", label: "All" },
    { value: "critical", label: "Critical" },
    { value: "high", label: "High" },
    { value: "medium", label: "Medium" },
    { value: "low", label: "Low" },
  ];

  return (
    <div className="space-y-6 pb-24 md:pb-8">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
        <div className="flex items-center gap-3">
          <h1 className="text-2xl font-bold tracking-tight">Alerts</h1>
          {unreadCount > 0 && (
            <span className="inline-flex items-center justify-center rounded-full bg-red-600 px-2.5 py-0.5 text-xs font-bold text-white">
              {unreadCount}
            </span>
          )}
        </div>
        <div className="flex items-center gap-3">
          <button
            onClick={markAllRead}
            className="rounded-lg border border-gray-700 px-3 py-2 text-sm font-medium text-gray-400 hover:bg-gray-800 hover:text-white transition-colors"
          >
            Mark All Read
          </button>
          <button
            onClick={runSentinel}
            disabled={runningAgent}
            className="rounded-lg bg-emerald-600 px-4 py-2 text-sm font-medium text-white hover:bg-emerald-500 transition-colors disabled:opacity-50 disabled:cursor-wait"
          >
            {runningAgent ? "Scanning…" : "Run Alert Sentinel"}
          </button>
        </div>
      </div>

      {/* Filter pills */}
      <div className="flex flex-wrap gap-2">
        {FILTER_OPTIONS.map((opt) => (
          <button
            key={opt.value}
            onClick={() => setFilter(opt.value)}
            className={`rounded-full px-3 py-1 text-xs font-medium transition-colors ${
              filter === opt.value
                ? "bg-emerald-600 text-white"
                : "bg-gray-800 text-gray-400 hover:bg-gray-700 hover:text-white"
            }`}
          >
            {opt.label}
          </button>
        ))}
      </div>

      {/* Alert groups */}
      {loading ? (
        <div className="space-y-4">
          {[1, 2, 3, 4].map((i) => (
            <div key={i} className="skeleton h-20 w-full rounded-xl" />
          ))}
        </div>
      ) : (
        <div className="space-y-8">
          {SEVERITY_ORDER.map((severity) => {
            const items = grouped.get(severity) ?? [];
            if (filter !== "all" && filter !== severity) return null;

            return (
              <section key={severity}>
                <h2 className={`text-sm font-semibold uppercase tracking-wide mb-3 ${SEVERITY_HEADER_STYLES[severity]}`}>
                  {severity} {items.length > 0 && <span className="text-gray-600">({items.length})</span>}
                </h2>

                {items.length > 0 ? (
                  <div className="space-y-3">
                    {items.map((alert) => (
                      <AlertCard key={alert.id} alert={alert} onDismiss={handleDismiss} />
                    ))}
                  </div>
                ) : (
                  <div className="rounded-xl bg-gray-900 p-6 text-center text-sm text-gray-600">
                    No {severity} alerts — you&apos;re clear
                  </div>
                )}
              </section>
            );
          })}
        </div>
      )}
    </div>
  );
}
