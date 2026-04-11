"use client";

import { useCallback, useEffect, useState } from "react";
import AlertCard from "@/components/ui/AlertCard";
import CashflowForecast from "@/components/charts/CashflowForecast";
import { api } from "@/lib/api";
import type { Alert } from "@/types";

const USER_ID = "demo-user-123";

/* ------------------------------------------------------------------ */
/* Metric card                                                         */
/* ------------------------------------------------------------------ */

interface MetricCardProps {
  label: string;
  value: string;
  indicator?: "green" | "yellow" | "red" | "blue";
  loading?: boolean;
}

function MetricCard({ label, value, indicator = "green", loading }: MetricCardProps) {
  const indicatorColor: Record<string, string> = {
    green: "bg-emerald-500",
    yellow: "bg-yellow-500",
    red: "bg-red-500",
    blue: "bg-blue-500",
  };

  if (loading) {
    return (
      <div className="rounded-xl bg-gray-900 p-5">
        <div className="skeleton h-3 w-24 mb-3" />
        <div className="skeleton h-8 w-32" />
      </div>
    );
  }

  return (
    <div className="rounded-xl bg-gray-900 p-5">
      <div className="flex items-center gap-2 mb-1">
        <span className={`inline-block h-2 w-2 rounded-full ${indicatorColor[indicator]}`} />
        <span className="text-xs font-medium text-gray-400 uppercase tracking-wide">
          {label}
        </span>
      </div>
      <p className="text-2xl font-bold tracking-tight">{value}</p>
    </div>
  );
}

/* ------------------------------------------------------------------ */
/* Quick-action button                                                 */
/* ------------------------------------------------------------------ */

interface QuickActionProps {
  label: string;
  onClick: () => void;
  loading?: boolean;
  icon: React.ReactNode;
}

function QuickAction({ label, onClick, loading, icon }: QuickActionProps) {
  return (
    <button
      onClick={onClick}
      disabled={loading}
      className="flex flex-col items-center gap-2 rounded-xl bg-gray-900 px-4 py-4 text-sm font-medium text-gray-300 hover:bg-gray-800 hover:text-white transition-colors disabled:opacity-50 disabled:cursor-wait"
    >
      <span className="text-emerald-500">{icon}</span>
      {loading ? "Running…" : label}
    </button>
  );
}

/* ------------------------------------------------------------------ */
/* Dashboard page                                                      */
/* ------------------------------------------------------------------ */

interface ForecastProjection {
  date: string;
  projected_balance: number;
}

export default function DashboardPage() {
  const [balance, setBalance] = useState<number | null>(null);
  const [weekSpending, setWeekSpending] = useState<number | null>(null);
  const [subscriptionCount, setSubscriptionCount] = useState<number | null>(null);
  const [healthScore, setHealthScore] = useState<number | null>(null);
  const [alerts, setAlerts] = useState<Alert[]>([]);
  const [forecast, setForecast] = useState<ForecastProjection[]>([]);
  const [loading, setLoading] = useState(true);
  const [runningAgent, setRunningAgent] = useState<string | null>(null);

  /* ── Initial data fetch ─────────────────────────────────────────── */

  useEffect(() => {
    async function load() {
      try {
        const [alertsRes, txnRes] = await Promise.allSettled([
          api.getAlerts(USER_ID),
          api.getTransactions(USER_ID),
        ]);

        if (alertsRes.status === "fulfilled") {
          setAlerts(
            (alertsRes.value as Alert[]).slice(0, 5),
          );
        }

        if (txnRes.status === "fulfilled") {
          const txns = txnRes.value as { amount: number }[];
          const total = txns.reduce((sum, t) => sum + Math.abs(t.amount), 0);
          setWeekSpending(total);
        }
      } catch {
        // API may be unreachable during development
      } finally {
        setLoading(false);
      }
    }

    load();
  }, []);

  /* ── Agent runners ──────────────────────────────────────────────── */

  const runAgent = useCallback(async (agentName: string) => {
    setRunningAgent(agentName);
    try {
      const result = await api.runAgent(agentName, { user_id: USER_ID });
      const output = result.output ?? {};

      switch (agentName) {
        case "cashflow-prophet": {
          const projections = (output.daily_projections ?? []) as ForecastProjection[];
          setForecast(projections);
          break;
        }
        case "subscription-auditor": {
          const subs = (output.subscriptions ?? []) as unknown[];
          setSubscriptionCount(subs.length);
          break;
        }
        case "advisor": {
          setHealthScore((output.health_score as number) ?? null);
          setBalance((output.current_balance as number) ?? null);
          break;
        }
      }
    } catch (err) {
      console.error(`Agent ${agentName} failed:`, err);
    } finally {
      setRunningAgent(null);
    }
  }, []);

  const handleDismiss = useCallback((id: string) => {
    setAlerts((prev) => prev.filter((a) => a.id !== id));
  }, []);

  /* ── Derived values ─────────────────────────────────────────────── */

  const balanceDisplay = balance !== null ? `$${balance.toLocaleString()}` : "$—";
  const spendingDisplay = weekSpending !== null ? `$${weekSpending.toLocaleString()}` : "$—";
  const subsDisplay = subscriptionCount !== null ? String(subscriptionCount) : "—";
  const healthDisplay = healthScore !== null ? `${healthScore}/100` : "—/100";

  const healthIndicator: MetricCardProps["indicator"] =
    healthScore === null ? "blue"
    : healthScore >= 75 ? "green"
    : healthScore >= 50 ? "yellow"
    : "red";

  return (
    <div className="space-y-8 pb-24 md:pb-8">
      {/* ── Header ──────────────────────────────────────────────── */}
      <div>
        <h1 className="text-2xl font-bold tracking-tight">Dashboard</h1>
        <p className="mt-1 text-sm text-gray-400">
          Your financial overview at a glance
        </p>
      </div>

      {/* ── Metric cards ────────────────────────────────────────── */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
        <MetricCard
          label="Current Balance"
          value={balanceDisplay}
          indicator="green"
          loading={loading}
        />
        <MetricCard
          label="This Week"
          value={spendingDisplay}
          indicator={
            weekSpending !== null && weekSpending > 1000 ? "red"
            : weekSpending !== null && weekSpending > 500 ? "yellow"
            : "green"
          }
          loading={loading}
        />
        <MetricCard
          label="Subscriptions"
          value={subsDisplay}
          indicator="blue"
          loading={loading}
        />
        <MetricCard
          label="Health Score"
          value={healthDisplay}
          indicator={healthIndicator}
          loading={loading}
        />
      </div>

      {/* ── Cashflow forecast ───────────────────────────────────── */}
      <section className="rounded-xl bg-gray-900 p-5">
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-lg font-semibold">60-Day Cashflow Forecast</h2>
          <button
            onClick={() => runAgent("cashflow-prophet")}
            disabled={runningAgent !== null}
            className="rounded-lg bg-emerald-600 px-3 py-1.5 text-xs font-medium text-white hover:bg-emerald-500 transition-colors disabled:opacity-50 disabled:cursor-wait"
          >
            {runningAgent === "cashflow-prophet" ? "Running…" : "Run Forecast"}
          </button>
        </div>

        {forecast.length > 0 ? (
          <CashflowForecast projections={forecast} />
        ) : (
          <div className="flex flex-col items-center justify-center h-64 text-gray-500 text-sm">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={1.5} className="w-10 h-10 mb-3 text-gray-600">
              <path strokeLinecap="round" strokeLinejoin="round" d="M3 13.125C3 12.504 3.504 12 4.125 12h2.25c.621 0 1.125.504 1.125 1.125v6.75C7.5 20.496 6.996 21 6.375 21h-2.25A1.125 1.125 0 0 1 3 19.875v-6.75ZM9.75 8.625c0-.621.504-1.125 1.125-1.125h2.25c.621 0 1.125.504 1.125 1.125v11.25c0 .621-.504 1.125-1.125 1.125h-2.25a1.125 1.125 0 0 1-1.125-1.125V8.625ZM16.5 4.125c0-.621.504-1.125 1.125-1.125h2.25C20.496 3 21 3.504 21 4.125v15.75c0 .621-.504 1.125-1.125 1.125h-2.25a1.125 1.125 0 0 1-1.125-1.125V4.125Z" />
            </svg>
            Run the Cashflow Prophet to see your 60-day forecast
          </div>
        )}
      </section>

      {/* ── Recent alerts ───────────────────────────────────────── */}
      <section>
        <h2 className="text-lg font-semibold mb-4">Recent Alerts</h2>
        {loading ? (
          <div className="space-y-3">
            {[1, 2, 3].map((i) => (
              <div key={i} className="skeleton h-20 w-full" />
            ))}
          </div>
        ) : alerts.length > 0 ? (
          <div className="space-y-3">
            {alerts.map((alert) => (
              <AlertCard
                key={alert.id}
                alert={alert}
                onDismiss={handleDismiss}
              />
            ))}
          </div>
        ) : (
          <div className="rounded-xl bg-gray-900 p-8 text-center text-sm text-gray-500">
            All clear — no alerts right now
          </div>
        )}
      </section>

      {/* ── Quick actions ───────────────────────────────────────── */}
      <section>
        <h2 className="text-lg font-semibold mb-4">Quick Actions</h2>
        <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
          <QuickAction
            label="Scan Receipt"
            loading={runningAgent === "receipt-scanner"}
            onClick={() => runAgent("receipt-scanner")}
            icon={
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={1.5} className="w-6 h-6">
                <path strokeLinecap="round" strokeLinejoin="round" d="M6.827 6.175A2.31 2.31 0 0 1 5.186 7.23c-.38.054-.757.112-1.134.175C2.999 7.58 2.25 8.507 2.25 9.574V18a2.25 2.25 0 0 0 2.25 2.25h15A2.25 2.25 0 0 0 21.75 18V9.574c0-1.067-.75-1.994-1.802-2.169a47.865 47.865 0 0 0-1.134-.175 2.31 2.31 0 0 1-1.64-1.055l-.822-1.316a2.192 2.192 0 0 0-1.736-1.039 48.774 48.774 0 0 0-5.232 0 2.192 2.192 0 0 0-1.736 1.039l-.821 1.316Z" />
                <path strokeLinecap="round" strokeLinejoin="round" d="M16.5 12.75a4.5 4.5 0 1 1-9 0 4.5 4.5 0 0 1 9 0ZM18.75 10.5h.008v.008h-.008V10.5Z" />
              </svg>
            }
          />
          <QuickAction
            label="Ask Voice"
            loading={runningAgent === "voice"}
            onClick={() => runAgent("voice")}
            icon={
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={1.5} className="w-6 h-6">
                <path strokeLinecap="round" strokeLinejoin="round" d="M12 18.75a6 6 0 0 0 6-6v-1.5m-6 7.5a6 6 0 0 1-6-6v-1.5m6 7.5v3.75m-3.75 0h7.5M12 15.75a3 3 0 0 1-3-3V4.5a3 3 0 1 1 6 0v8.25a3 3 0 0 1-3 3Z" />
              </svg>
            }
          />
          <QuickAction
            label="Weekly Briefing"
            loading={runningAgent === "advisor"}
            onClick={() => runAgent("advisor")}
            icon={
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={1.5} className="w-6 h-6">
                <path strokeLinecap="round" strokeLinejoin="round" d="M19.114 5.636a9 9 0 0 1 0 12.728M16.463 8.288a5.25 5.25 0 0 1 0 7.424M6.75 8.25l4.72-4.72a.75.75 0 0 1 1.28.53v15.88a.75.75 0 0 1-1.28.53l-4.72-4.72H4.51c-.88 0-1.704-.507-1.938-1.354A9.009 9.009 0 0 1 2.25 12c0-.83.112-1.633.322-2.396C2.806 8.756 3.63 8.25 4.51 8.25H6.75Z" />
              </svg>
            }
          />
          <QuickAction
            label="Check Subs"
            loading={runningAgent === "subscription-auditor"}
            onClick={() => runAgent("subscription-auditor")}
            icon={
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={1.5} className="w-6 h-6">
                <path strokeLinecap="round" strokeLinejoin="round" d="M16.023 9.348h4.992v-.001M2.985 19.644v-4.992m0 0h4.992m-4.993 0 3.181 3.183a8.25 8.25 0 0 0 13.803-3.7M4.031 9.865a8.25 8.25 0 0 1 13.803-3.7l3.181 3.182" />
              </svg>
            }
          />
        </div>
      </section>
    </div>
  );
}
