"use client";

import { useCallback, useEffect, useMemo, useState } from "react";
import { api } from "@/lib/api";
import type { Transaction } from "@/types";

const USER_ID = "demo-user-123";

const MOCK_TRANSACTIONS: Transaction[] = [
  { id: "t1", merchant_name: "Whole Foods", amount: 87.43, category: "grocery", date: "2025-04-10", pending: false },
  { id: "t2", merchant_name: "Starbucks", amount: 6.75, category: "restaurant", date: "2025-04-10", pending: false },
  { id: "t3", merchant_name: "Netflix", amount: 15.99, category: "entertainment", date: "2025-04-09", pending: false },
  { id: "t4", merchant_name: "Uber", amount: 23.5, category: "transport", date: "2025-04-09", pending: true },
  { id: "t5", merchant_name: "Amazon", amount: 134.99, category: "shopping", date: "2025-04-08", pending: false },
];

const CATEGORIES = ["All", "Grocery", "Restaurant", "Transport", "Shopping", "Entertainment", "Utilities"] as const;

const CATEGORY_COLORS: Record<string, string> = {
  grocery: "bg-green-900 text-green-300",
  restaurant: "bg-orange-900 text-orange-300",
  transport: "bg-blue-900 text-blue-300",
  shopping: "bg-purple-900 text-purple-300",
  entertainment: "bg-pink-900 text-pink-300",
  utilities: "bg-gray-700 text-gray-300",
  general: "bg-gray-800 text-gray-400",
};

type SortKey = "date-desc" | "date-asc" | "amount-desc" | "amount-asc";

const SORT_OPTIONS: { value: SortKey; label: string }[] = [
  { value: "date-desc", label: "Date (newest)" },
  { value: "date-asc", label: "Date (oldest)" },
  { value: "amount-desc", label: "Amount (highest)" },
  { value: "amount-asc", label: "Amount (lowest)" },
];

export default function TransactionsPage() {
  const [transactions, setTransactions] = useState<Transaction[]>([]);
  const [loading, setLoading] = useState(true);
  const [searchQuery, setSearchQuery] = useState("");
  const [selectedCategory, setSelectedCategory] = useState("All");
  const [sortBy, setSortBy] = useState<SortKey>("date-desc");
  const [runningAgent, setRunningAgent] = useState(false);
  const [agentOutput, setAgentOutput] = useState<Record<string, unknown> | null>(null);
  const [panelOpen, setPanelOpen] = useState(false);

  useEffect(() => {
    async function load() {
      try {
        const data = await api.getTransactions(USER_ID);
        setTransactions(data.length > 0 ? data : MOCK_TRANSACTIONS);
      } catch {
        setTransactions(MOCK_TRANSACTIONS);
      } finally {
        setLoading(false);
      }
    }
    load();
  }, []);

  const filtered = useMemo(() => {
    let result = [...transactions];

    if (searchQuery) {
      const q = searchQuery.toLowerCase();
      result = result.filter((t) => t.merchant_name.toLowerCase().includes(q));
    }

    if (selectedCategory !== "All") {
      result = result.filter((t) => t.category.toLowerCase() === selectedCategory.toLowerCase());
    }

    result.sort((a, b) => {
      switch (sortBy) {
        case "date-desc":
          return new Date(b.date).getTime() - new Date(a.date).getTime();
        case "date-asc":
          return new Date(a.date).getTime() - new Date(b.date).getTime();
        case "amount-desc":
          return Math.abs(b.amount) - Math.abs(a.amount);
        case "amount-asc":
          return Math.abs(a.amount) - Math.abs(b.amount);
      }
    });

    return result;
  }, [transactions, searchQuery, selectedCategory, sortBy]);

  const totalSpent = useMemo(
    () => filtered.reduce((sum, t) => sum + Math.abs(t.amount), 0),
    [filtered],
  );
  const avgAmount = filtered.length > 0 ? totalSpent / filtered.length : 0;

  const runMonitor = useCallback(async () => {
    setRunningAgent(true);
    try {
      const res = await api.runAgent("transaction-monitor", { user_id: USER_ID });
      setAgentOutput(res.output);
      setPanelOpen(true);
    } catch (err) {
      console.error("Transaction monitor failed:", err);
    } finally {
      setRunningAgent(false);
    }
  }, []);

  return (
    <div className="space-y-6 pb-24 md:pb-8">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
        <div>
          <h1 className="text-2xl font-bold tracking-tight">Transactions</h1>
          <p className="mt-1 text-sm text-gray-400">Review and monitor your spending</p>
        </div>
        <div className="flex items-center gap-3">
          <div className="relative">
            <svg viewBox="0 0 20 20" fill="currentColor" className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-gray-500">
              <path fillRule="evenodd" d="M9 3.5a5.5 5.5 0 1 0 0 11 5.5 5.5 0 0 0 0-11ZM2 9a7 7 0 1 1 12.452 4.391l3.328 3.329a.75.75 0 1 1-1.06 1.06l-3.329-3.328A7 7 0 0 1 2 9Z" clipRule="evenodd" />
            </svg>
            <input
              type="text"
              placeholder="Search merchants…"
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="w-full sm:w-64 rounded-lg border border-gray-800 bg-gray-900 py-2 pl-9 pr-3 text-sm text-white placeholder-gray-500 focus:border-emerald-600 focus:outline-none focus:ring-1 focus:ring-emerald-600"
            />
          </div>
          <button
            onClick={runMonitor}
            disabled={runningAgent}
            className="shrink-0 rounded-lg bg-emerald-600 px-4 py-2 text-sm font-medium text-white hover:bg-emerald-500 transition-colors disabled:opacity-50 disabled:cursor-wait"
          >
            {runningAgent ? "Running…" : "Run Monitor"}
          </button>
        </div>
      </div>

      {/* Filters */}
      <div className="flex flex-col sm:flex-row sm:items-center gap-3">
        <div className="flex flex-wrap gap-2">
          {CATEGORIES.map((cat) => (
            <button
              key={cat}
              onClick={() => setSelectedCategory(cat)}
              className={`rounded-full px-3 py-1 text-xs font-medium transition-colors ${
                selectedCategory === cat
                  ? "bg-emerald-600 text-white"
                  : "bg-gray-800 text-gray-400 hover:bg-gray-700 hover:text-white"
              }`}
            >
              {cat}
            </button>
          ))}
        </div>
        <select
          value={sortBy}
          onChange={(e) => setSortBy(e.target.value as SortKey)}
          className="ml-auto rounded-lg border border-gray-800 bg-gray-900 px-3 py-1.5 text-sm text-gray-300 focus:border-emerald-600 focus:outline-none"
        >
          {SORT_OPTIONS.map((opt) => (
            <option key={opt.value} value={opt.value}>
              {opt.label}
            </option>
          ))}
        </select>
      </div>

      {/* Transaction list */}
      <div className="rounded-xl bg-gray-900 overflow-hidden">
        {/* Desktop header */}
        <div className="hidden sm:grid sm:grid-cols-[1fr_120px_100px_100px_80px] gap-4 px-5 py-3 border-b border-gray-800 text-xs font-medium text-gray-500 uppercase tracking-wide">
          <span>Merchant</span>
          <span>Category</span>
          <span className="text-right">Amount</span>
          <span className="text-right">Date</span>
          <span className="text-center">Status</span>
        </div>

        {loading ? (
          <div className="divide-y divide-gray-800">
            {[1, 2, 3, 4, 5].map((i) => (
              <div key={i} className="px-5 py-4">
                <div className="skeleton h-5 w-full" />
              </div>
            ))}
          </div>
        ) : filtered.length === 0 ? (
          <div className="px-5 py-16 text-center text-sm text-gray-500">
            No transactions match your filters
          </div>
        ) : (
          <div className="divide-y divide-gray-800">
            {filtered.map((txn) => (
              <div
                key={txn.id}
                className="grid grid-cols-2 sm:grid-cols-[1fr_120px_100px_100px_80px] gap-2 sm:gap-4 items-center px-5 py-3 hover:bg-gray-800/50 transition-colors"
              >
                <span className="text-sm font-medium text-white truncate">{txn.merchant_name}</span>
                <span>
                  <span className={`inline-block rounded-full px-2.5 py-0.5 text-xs font-medium ${CATEGORY_COLORS[txn.category] ?? CATEGORY_COLORS.general}`}>
                    {txn.category}
                  </span>
                </span>
                <span className={`text-sm font-semibold text-right ${txn.amount < 0 ? "text-emerald-400" : "text-red-400"}`}>
                  {txn.amount < 0 ? "+" : "-"}${Math.abs(txn.amount).toFixed(2)}
                </span>
                <span className="text-sm text-gray-400 text-right">
                  {new Date(txn.date).toLocaleDateString("en-US", { month: "short", day: "numeric" })}
                </span>
                <span className="text-center">
                  {txn.pending ? (
                    <span className="inline-block rounded-full bg-yellow-900/50 px-2 py-0.5 text-xs font-medium text-yellow-400">
                      Pending
                    </span>
                  ) : (
                    <span className="inline-block rounded-full bg-gray-800 px-2 py-0.5 text-xs text-gray-500">
                      Posted
                    </span>
                  )}
                </span>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Summary bar */}
      {!loading && filtered.length > 0 && (
        <div className="flex flex-wrap gap-6 rounded-xl bg-gray-900 px-5 py-4 text-sm">
          <div>
            <span className="text-gray-500">Transactions </span>
            <span className="font-semibold text-white">{filtered.length}</span>
          </div>
          <div>
            <span className="text-gray-500">Total spent </span>
            <span className="font-semibold text-white">${totalSpent.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}</span>
          </div>
          <div>
            <span className="text-gray-500">Average </span>
            <span className="font-semibold text-white">${avgAmount.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}</span>
          </div>
        </div>
      )}

      {/* Agent output slide-in panel */}
      {panelOpen && agentOutput && (
        <div className="fixed inset-y-0 right-0 z-50 w-full max-w-md bg-gray-900 border-l border-gray-800 shadow-2xl flex flex-col">
          <div className="flex items-center justify-between px-5 py-4 border-b border-gray-800">
            <h3 className="text-sm font-semibold">Transaction Monitor Results</h3>
            <button
              onClick={() => setPanelOpen(false)}
              className="rounded p-1 text-gray-500 hover:bg-gray-800 hover:text-gray-300 transition-colors"
            >
              <svg viewBox="0 0 20 20" fill="currentColor" className="w-5 h-5">
                <path d="M6.28 5.22a.75.75 0 0 0-1.06 1.06L8.94 10l-3.72 3.72a.75.75 0 1 0 1.06 1.06L10 11.06l3.72 3.72a.75.75 0 1 0 1.06-1.06L11.06 10l3.72-3.72a.75.75 0 0 0-1.06-1.06L10 8.94 6.28 5.22Z" />
              </svg>
            </button>
          </div>
          <div className="flex-1 overflow-y-auto p-5">
            <pre className="text-xs text-gray-300 whitespace-pre-wrap break-words">
              {JSON.stringify(agentOutput, null, 2)}
            </pre>
          </div>
        </div>
      )}
    </div>
  );
}
