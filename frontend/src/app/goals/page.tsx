"use client";

import { useCallback, useEffect, useMemo, useState } from "react";
import { api } from "@/lib/api";
import type { Goal } from "@/types";

const USER_ID = "demo-user-123";

interface MockGoal {
  id: string;
  name: string;
  target_amount: number;
  current_amount: number;
  deadline: string;
}

const MOCK_GOALS: MockGoal[] = [
  { id: "g1", name: "Emergency Fund", target_amount: 10000, current_amount: 6500, deadline: "2025-12-31" },
  { id: "g2", name: "New Laptop", target_amount: 2000, current_amount: 1850, deadline: "2025-05-01" },
  { id: "g3", name: "Europe Trip", target_amount: 5000, current_amount: 1200, deadline: "2026-06-01" },
];

function mockToGoal(m: MockGoal): Goal {
  const pct = Math.min(100, Math.round((m.current_amount / m.target_amount) * 100));
  const remaining = Math.max(0, m.target_amount - m.current_amount);
  const daysRemaining = Math.max(0, Math.ceil((new Date(m.deadline).getTime() - Date.now()) / 86_400_000));
  const monthsLeft = Math.max(1, daysRemaining / 30);
  const status: Goal["status"] =
    pct >= 100 ? "completed" : daysRemaining <= 0 ? "overdue" : pct >= (100 - daysRemaining / 3) ? "on_track" : "behind";

  return {
    goal_id: m.id,
    goal_name: m.name,
    target_amount: m.target_amount,
    current_amount: m.current_amount,
    percent_complete: pct,
    amount_remaining: remaining,
    deadline: m.deadline,
    days_remaining: daysRemaining,
    required_monthly_savings: Math.round(remaining / monthsLeft),
    estimated_monthly_savings: Math.round(remaining / monthsLeft * 0.9),
    projected_completion_date: m.deadline,
    status,
  };
}

function goalEmoji(name: string): string {
  const lower = name.toLowerCase();
  if (lower.includes("house") || lower.includes("home")) return "🏠";
  if (lower.includes("car")) return "🚗";
  if (lower.includes("travel") || lower.includes("trip") || lower.includes("vacation")) return "✈️";
  if (lower.includes("laptop") || lower.includes("computer")) return "💻";
  if (lower.includes("emergency") || lower.includes("fund")) return "🛡️";
  return "🎯";
}

const STATUS_STYLES: Record<string, string> = {
  on_track: "bg-emerald-900/50 text-emerald-400",
  ahead: "bg-emerald-900/50 text-emerald-300",
  behind: "bg-amber-900/50 text-amber-400",
  completed: "bg-blue-900/50 text-blue-400",
  overdue: "bg-red-900/50 text-red-400",
};

const PROGRESS_COLORS: Record<string, string> = {
  on_track: "bg-emerald-500",
  ahead: "bg-emerald-400",
  behind: "bg-amber-500",
  completed: "bg-blue-500",
  overdue: "bg-red-500",
};

export default function GoalsPage() {
  const [goals, setGoals] = useState<Goal[]>([]);
  const [loading, setLoading] = useState(true);
  const [showAddGoal, setShowAddGoal] = useState(false);
  const [runningAgent, setRunningAgent] = useState(false);
  const [agentResult, setAgentResult] = useState<Record<string, unknown> | null>(null);
  const [showResult, setShowResult] = useState(false);

  const [formName, setFormName] = useState("");
  const [formTarget, setFormTarget] = useState("");
  const [formDate, setFormDate] = useState("");
  const [formStarting, setFormStarting] = useState("");

  useEffect(() => {
    async function load() {
      try {
        const res = await api.runAgent("goal-tracker", { user_id: USER_ID });
        const output = res.output ?? {};
        const goalData = (output.goals ?? []) as Goal[];
        setGoals(goalData.length > 0 ? goalData : MOCK_GOALS.map(mockToGoal));
      } catch {
        setGoals(MOCK_GOALS.map(mockToGoal));
      } finally {
        setLoading(false);
      }
    }
    load();
  }, []);

  const { totalSaved, totalTarget, overallPct } = useMemo(() => {
    const saved = goals.reduce((s, g) => s + g.current_amount, 0);
    const target = goals.reduce((s, g) => s + g.target_amount, 0);
    return { totalSaved: saved, totalTarget: target, overallPct: target > 0 ? Math.round((saved / target) * 100) : 0 };
  }, [goals]);

  const runGoalTracker = useCallback(async () => {
    setRunningAgent(true);
    try {
      const res = await api.runAgent("goal-tracker", { user_id: USER_ID });
      setAgentResult(res.output);
      setShowResult(true);
      const goalData = (res.output.goals ?? []) as Goal[];
      if (goalData.length > 0) setGoals(goalData);
    } catch (err) {
      console.error("Goal tracker failed:", err);
    } finally {
      setRunningAgent(false);
    }
  }, []);

  function handleAddGoal() {
    console.log("Add goal:", { name: formName, target: formTarget, date: formDate, starting: formStarting });
    setShowAddGoal(false);
    setFormName("");
    setFormTarget("");
    setFormDate("");
    setFormStarting("");
  }

  return (
    <div className="space-y-6 pb-24 md:pb-8">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold tracking-tight">Savings Goals</h1>
          <p className="mt-1 text-sm text-gray-400">Track your progress toward what matters</p>
        </div>
        <div className="flex items-center gap-3">
          <button
            onClick={runGoalTracker}
            disabled={runningAgent}
            className="rounded-lg bg-gray-800 px-4 py-2 text-sm font-medium text-gray-300 hover:bg-gray-700 hover:text-white transition-colors disabled:opacity-50"
          >
            {runningAgent ? "Running…" : "Run Goal Tracker"}
          </button>
          <button
            onClick={() => setShowAddGoal(true)}
            className="rounded-lg bg-emerald-600 px-4 py-2 text-sm font-medium text-white hover:bg-emerald-500 transition-colors"
          >
            + Add Goal
          </button>
        </div>
      </div>

      {/* Summary */}
      {!loading && goals.length > 0 && (
        <div className="grid grid-cols-3 gap-4">
          <div className="rounded-xl bg-gray-900 p-4 text-center">
            <p className="text-xs text-gray-500 uppercase tracking-wide mb-1">Total Saved</p>
            <p className="text-xl font-bold text-emerald-400">${totalSaved.toLocaleString()}</p>
          </div>
          <div className="rounded-xl bg-gray-900 p-4 text-center">
            <p className="text-xs text-gray-500 uppercase tracking-wide mb-1">Total Target</p>
            <p className="text-xl font-bold text-white">${totalTarget.toLocaleString()}</p>
          </div>
          <div className="rounded-xl bg-gray-900 p-4 text-center">
            <p className="text-xs text-gray-500 uppercase tracking-wide mb-1">Overall</p>
            <p className="text-xl font-bold text-white">{overallPct}%</p>
          </div>
        </div>
      )}

      {/* Goal cards */}
      {loading ? (
        <div className="grid md:grid-cols-2 gap-4">
          {[1, 2, 3].map((i) => (
            <div key={i} className="skeleton h-48 w-full rounded-xl" />
          ))}
        </div>
      ) : (
        <div className="grid md:grid-cols-2 gap-4">
          {goals.map((goal) => (
            <div key={goal.goal_id} className="rounded-xl bg-gray-900 p-5 space-y-4">
              <div className="flex items-start justify-between">
                <div className="flex items-center gap-3">
                  <span className="text-2xl">{goalEmoji(goal.goal_name)}</span>
                  <div>
                    <h3 className="text-sm font-semibold text-white">{goal.goal_name}</h3>
                    <p className="text-xs text-gray-500">
                      {goal.days_remaining > 0 ? `${goal.days_remaining} days left` : "Deadline passed"}
                    </p>
                  </div>
                </div>
                <span className={`rounded-full px-2.5 py-0.5 text-xs font-medium ${STATUS_STYLES[goal.status] ?? STATUS_STYLES.on_track}`}>
                  {goal.status.replace("_", " ")}
                </span>
              </div>

              {/* Progress bar */}
              <div>
                <div className="flex items-center justify-between text-xs text-gray-400 mb-1.5">
                  <span>${goal.current_amount.toLocaleString()}</span>
                  <span>${goal.target_amount.toLocaleString()}</span>
                </div>
                <div className="h-2.5 rounded-full bg-gray-800 overflow-hidden">
                  <div
                    className={`h-full rounded-full transition-all duration-500 ${PROGRESS_COLORS[goal.status] ?? PROGRESS_COLORS.on_track}`}
                    style={{ width: `${Math.min(100, goal.percent_complete)}%` }}
                  />
                </div>
                <p className="mt-1.5 text-xs text-gray-500 text-right">
                  {goal.percent_complete}% complete &middot; ${goal.amount_remaining.toLocaleString()} to go
                </p>
              </div>

              {/* Details */}
              <div className="grid grid-cols-2 gap-2 text-xs">
                <div className="rounded-lg bg-gray-800/50 p-2">
                  <p className="text-gray-500">Monthly needed</p>
                  <p className="font-medium text-white">${goal.required_monthly_savings.toLocaleString()}</p>
                </div>
                <div className="rounded-lg bg-gray-800/50 p-2">
                  <p className="text-gray-500">Deadline</p>
                  <p className="font-medium text-white">
                    {goal.deadline ? new Date(goal.deadline).toLocaleDateString("en-US", { month: "short", day: "numeric", year: "numeric" }) : "None"}
                  </p>
                </div>
              </div>
            </div>
          ))}
        </div>
      )}

      {/* Agent result */}
      {showResult && agentResult && (
        <div className="rounded-xl bg-gray-900 p-5">
          <div className="flex items-center justify-between mb-3">
            <h3 className="text-sm font-semibold">Goal Tracker Results</h3>
            <button onClick={() => setShowResult(false)} className="text-xs text-gray-500 hover:text-white transition-colors">
              Hide
            </button>
          </div>
          <pre className="text-xs text-gray-300 whitespace-pre-wrap break-words max-h-64 overflow-y-auto">
            {JSON.stringify(agentResult, null, 2)}
          </pre>
        </div>
      )}

      {/* Add Goal modal */}
      {showAddGoal && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm">
          <div className="w-full max-w-md mx-4 rounded-xl bg-gray-900 border border-gray-800 p-6 space-y-5">
            <div className="flex items-center justify-between">
              <h2 className="text-lg font-semibold">Add New Goal</h2>
              <button onClick={() => setShowAddGoal(false)} className="rounded p-1 text-gray-500 hover:bg-gray-800 hover:text-white transition-colors">
                <svg viewBox="0 0 20 20" fill="currentColor" className="w-5 h-5">
                  <path d="M6.28 5.22a.75.75 0 0 0-1.06 1.06L8.94 10l-3.72 3.72a.75.75 0 1 0 1.06 1.06L10 11.06l3.72 3.72a.75.75 0 1 0 1.06-1.06L11.06 10l3.72-3.72a.75.75 0 0 0-1.06-1.06L10 8.94 6.28 5.22Z" />
                </svg>
              </button>
            </div>

            <div className="space-y-4">
              <div>
                <label className="block text-xs font-medium text-gray-400 mb-1">Goal Name</label>
                <input
                  type="text"
                  value={formName}
                  onChange={(e) => setFormName(e.target.value)}
                  placeholder="e.g. Emergency Fund"
                  className="w-full rounded-lg border border-gray-700 bg-gray-800 px-3 py-2 text-sm text-white placeholder-gray-500 focus:border-emerald-600 focus:outline-none focus:ring-1 focus:ring-emerald-600"
                />
              </div>
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <label className="block text-xs font-medium text-gray-400 mb-1">Target Amount ($)</label>
                  <input
                    type="number"
                    value={formTarget}
                    onChange={(e) => setFormTarget(e.target.value)}
                    placeholder="10000"
                    className="w-full rounded-lg border border-gray-700 bg-gray-800 px-3 py-2 text-sm text-white placeholder-gray-500 focus:border-emerald-600 focus:outline-none focus:ring-1 focus:ring-emerald-600"
                  />
                </div>
                <div>
                  <label className="block text-xs font-medium text-gray-400 mb-1">Starting Amount ($)</label>
                  <input
                    type="number"
                    value={formStarting}
                    onChange={(e) => setFormStarting(e.target.value)}
                    placeholder="0"
                    className="w-full rounded-lg border border-gray-700 bg-gray-800 px-3 py-2 text-sm text-white placeholder-gray-500 focus:border-emerald-600 focus:outline-none focus:ring-1 focus:ring-emerald-600"
                  />
                </div>
              </div>
              <div>
                <label className="block text-xs font-medium text-gray-400 mb-1">Target Date</label>
                <input
                  type="date"
                  value={formDate}
                  onChange={(e) => setFormDate(e.target.value)}
                  className="w-full rounded-lg border border-gray-700 bg-gray-800 px-3 py-2 text-sm text-white focus:border-emerald-600 focus:outline-none focus:ring-1 focus:ring-emerald-600"
                />
              </div>
            </div>

            <div className="flex justify-end gap-3 pt-2">
              <button
                onClick={() => setShowAddGoal(false)}
                className="rounded-lg px-4 py-2 text-sm font-medium text-gray-400 hover:text-white transition-colors"
              >
                Cancel
              </button>
              <button
                onClick={handleAddGoal}
                className="rounded-lg bg-emerald-600 px-4 py-2 text-sm font-medium text-white hover:bg-emerald-500 transition-colors"
              >
                Create Goal
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
