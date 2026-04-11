"use client";

import { useCallback, useEffect, useState } from "react";
import { api } from "@/lib/api";

const USER_ID = "demo-user-123";

interface AgentConfig {
  name: string;
  label: string;
  enabled: boolean;
}

const AGENT_GROUPS: { title: string; agents: AgentConfig[] }[] = [
  {
    title: "Monitoring Agents",
    agents: [
      { name: "transaction-monitor", label: "Transaction Monitor", enabled: true },
      { name: "email-monitor", label: "Email Monitor", enabled: true },
      { name: "alert-sentinel", label: "Alert Sentinel", enabled: true },
    ],
  },
  {
    title: "Analysis Agents",
    agents: [
      { name: "subscription-auditor", label: "Subscription Auditor", enabled: true },
      { name: "bill-negotiator", label: "Bill Negotiator", enabled: false },
      { name: "cashflow-prophet", label: "Cashflow Prophet", enabled: true },
      { name: "calendar-analyst", label: "Calendar Analyst", enabled: false },
      { name: "goal-tracker", label: "Goal Tracker", enabled: true },
    ],
  },
  {
    title: "Interface Agents",
    agents: [
      { name: "voice", label: "Voice Assistant", enabled: true },
      { name: "receipt-scanner", label: "Receipt Scanner", enabled: true },
      { name: "document-analyzer", label: "Document Analyzer", enabled: false },
      { name: "advisor", label: "Financial Advisor", enabled: true },
    ],
  },
];

function Toggle({ checked, onChange, disabled }: { checked: boolean; onChange: (v: boolean) => void; disabled?: boolean }) {
  return (
    <button
      type="button"
      role="switch"
      aria-checked={checked}
      disabled={disabled}
      onClick={() => onChange(!checked)}
      className={`relative inline-flex h-6 w-11 shrink-0 cursor-pointer rounded-full border-2 border-transparent transition-colors duration-200 focus:outline-none focus:ring-2 focus:ring-emerald-600 focus:ring-offset-2 focus:ring-offset-gray-950 disabled:cursor-not-allowed disabled:opacity-50 ${
        checked ? "bg-emerald-600" : "bg-gray-700"
      }`}
    >
      <span
        className={`pointer-events-none inline-block h-5 w-5 rounded-full bg-white shadow-lg transition-transform duration-200 ${
          checked ? "translate-x-5" : "translate-x-0"
        }`}
      />
    </button>
  );
}

export default function SettingsPage() {
  const [agents, setAgents] = useState(AGENT_GROUPS);
  const [notifHigh, setNotifHigh] = useState(true);
  const [notifMedium, setNotifMedium] = useState(true);
  const [notifLow, setNotifLow] = useState(false);
  const [quietStart, setQuietStart] = useState("22:00");
  const [quietEnd, setQuietEnd] = useState("07:00");
  const [apiStatus, setApiStatus] = useState<"checking" | "online" | "offline">("checking");
  const [runningAll, setRunningAll] = useState(false);
  const [runProgress, setRunProgress] = useState(0);

  useEffect(() => {
    api.health()
      .then(() => setApiStatus("online"))
      .catch(() => setApiStatus("offline"));
  }, []);

  const toggleAgent = useCallback((groupIdx: number, agentIdx: number) => {
    setAgents((prev) => {
      const next = prev.map((g, gi) => ({
        ...g,
        agents: g.agents.map((a, ai) =>
          gi === groupIdx && ai === agentIdx ? { ...a, enabled: !a.enabled } : a,
        ),
      }));
      return next;
    });
  }, []);

  const runFullAnalysis = useCallback(async () => {
    setRunningAll(true);
    setRunProgress(0);
    const allAgents = agents.flatMap((g) => g.agents.filter((a) => a.enabled));
    for (let i = 0; i < allAgents.length; i++) {
      try {
        await api.runAgent(allAgents[i].name, { user_id: USER_ID });
      } catch {
        // continue on failure
      }
      setRunProgress(Math.round(((i + 1) / allAgents.length) * 100));
    }
    setRunningAll(false);
  }, [agents]);

  const connections = [
    { name: "Plaid (Banking)", connected: true },
    { name: "Gmail", connected: false },
    { name: "Google Calendar", connected: false },
  ];

  return (
    <div className="space-y-8 pb-24 md:pb-8">
      <div>
        <h1 className="text-2xl font-bold tracking-tight">Settings</h1>
        <p className="mt-1 text-sm text-gray-400">Manage your account and agent configuration</p>
      </div>

      {/* Profile */}
      <section className="rounded-xl bg-gray-900 p-5 space-y-4">
        <h2 className="text-base font-semibold">Profile</h2>
        <div className="flex items-center gap-3">
          <div className="flex h-10 w-10 items-center justify-center rounded-full bg-emerald-600 text-sm font-bold">
            DU
          </div>
          <div>
            <p className="text-sm font-medium text-white">Demo User</p>
            <p className="text-xs text-gray-500">{USER_ID}</p>
          </div>
        </div>
        <div className="grid sm:grid-cols-3 gap-3 pt-2">
          {connections.map((c) => (
            <div key={c.name} className="flex items-center justify-between rounded-lg bg-gray-800/50 px-4 py-3">
              <span className="text-sm text-gray-300">{c.name}</span>
              <span className={`inline-flex items-center gap-1.5 rounded-full px-2.5 py-0.5 text-xs font-medium ${
                c.connected ? "bg-emerald-900/50 text-emerald-400" : "bg-gray-700 text-gray-400"
              }`}>
                <span className={`h-1.5 w-1.5 rounded-full ${c.connected ? "bg-emerald-400" : "bg-gray-500"}`} />
                {c.connected ? "Connected" : "Not connected"}
              </span>
            </div>
          ))}
        </div>
      </section>

      {/* Agents */}
      <section className="rounded-xl bg-gray-900 p-5 space-y-5">
        <h2 className="text-base font-semibold">Agents</h2>
        {agents.map((group, gi) => (
          <div key={group.title}>
            <h3 className="text-xs font-medium text-gray-500 uppercase tracking-wide mb-3">{group.title}</h3>
            <div className="space-y-2">
              {group.agents.map((agent, ai) => (
                <div key={agent.name} className="flex items-center justify-between rounded-lg bg-gray-800/50 px-4 py-3">
                  <span className="text-sm text-gray-300">{agent.label}</span>
                  <Toggle checked={agent.enabled} onChange={() => toggleAgent(gi, ai)} />
                </div>
              ))}
            </div>
          </div>
        ))}
      </section>

      {/* Notifications */}
      <section className="rounded-xl bg-gray-900 p-5 space-y-4">
        <h2 className="text-base font-semibold">Notification Preferences</h2>
        <div className="space-y-2">
          <div className="flex items-center justify-between rounded-lg bg-gray-800/50 px-4 py-3">
            <span className="text-sm text-gray-300">Critical Alerts</span>
            <Toggle checked={true} onChange={() => {}} disabled />
          </div>
          <div className="flex items-center justify-between rounded-lg bg-gray-800/50 px-4 py-3">
            <span className="text-sm text-gray-300">High Alerts</span>
            <Toggle checked={notifHigh} onChange={setNotifHigh} />
          </div>
          <div className="flex items-center justify-between rounded-lg bg-gray-800/50 px-4 py-3">
            <span className="text-sm text-gray-300">Medium Alerts</span>
            <Toggle checked={notifMedium} onChange={setNotifMedium} />
          </div>
          <div className="flex items-center justify-between rounded-lg bg-gray-800/50 px-4 py-3">
            <span className="text-sm text-gray-300">Low Alerts</span>
            <Toggle checked={notifLow} onChange={setNotifLow} />
          </div>
        </div>
        <div className="grid grid-cols-2 gap-4 pt-2">
          <div>
            <label className="block text-xs font-medium text-gray-400 mb-1">Quiet Hours Start</label>
            <input
              type="time"
              value={quietStart}
              onChange={(e) => setQuietStart(e.target.value)}
              className="w-full rounded-lg border border-gray-700 bg-gray-800 px-3 py-2 text-sm text-white focus:border-emerald-600 focus:outline-none focus:ring-1 focus:ring-emerald-600"
            />
          </div>
          <div>
            <label className="block text-xs font-medium text-gray-400 mb-1">Quiet Hours End</label>
            <input
              type="time"
              value={quietEnd}
              onChange={(e) => setQuietEnd(e.target.value)}
              className="w-full rounded-lg border border-gray-700 bg-gray-800 px-3 py-2 text-sm text-white focus:border-emerald-600 focus:outline-none focus:ring-1 focus:ring-emerald-600"
            />
          </div>
        </div>
      </section>

      {/* Data */}
      <section className="rounded-xl bg-gray-900 p-5 space-y-4">
        <h2 className="text-base font-semibold">Data &amp; Diagnostics</h2>
        <div className="flex items-center gap-3">
          <span className={`h-2.5 w-2.5 rounded-full ${
            apiStatus === "online" ? "bg-emerald-400" : apiStatus === "offline" ? "bg-red-500" : "bg-yellow-500 animate-pulse"
          }`} />
          <span className="text-sm text-gray-300">
            API Status: {apiStatus === "online" ? "Online" : apiStatus === "offline" ? "Offline" : "Checking…"}
          </span>
        </div>
        <div className="flex flex-wrap gap-3">
          <button
            onClick={runFullAnalysis}
            disabled={runningAll}
            className="rounded-lg bg-emerald-600 px-4 py-2 text-sm font-medium text-white hover:bg-emerald-500 transition-colors disabled:opacity-50 disabled:cursor-wait"
          >
            {runningAll ? `Running… ${runProgress}%` : "Run Full Analysis"}
          </button>
          <button
            onClick={() => console.log("Cache cleared")}
            className="rounded-lg border border-gray-700 px-4 py-2 text-sm font-medium text-gray-400 hover:bg-gray-800 hover:text-white transition-colors"
          >
            Clear Cache
          </button>
        </div>
        {runningAll && (
          <div className="h-2 rounded-full bg-gray-800 overflow-hidden">
            <div
              className="h-full rounded-full bg-emerald-500 transition-all duration-300"
              style={{ width: `${runProgress}%` }}
            />
          </div>
        )}
      </section>
    </div>
  );
}
