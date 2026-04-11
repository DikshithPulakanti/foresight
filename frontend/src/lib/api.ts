const BASE_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

/* ------------------------------------------------------------------ */
/* Response types                                                      */
/* ------------------------------------------------------------------ */

export interface HealthResponse {
  status: string;
  version: string;
  env: string;
}

export interface AgentRunRequest {
  user_id: string;
  [key: string]: unknown;
}

export interface AgentRunResponse {
  status: string;
  session_id: string;
  output: Record<string, unknown>;
  steps: string[];
  error?: string;
}

export interface Alert {
  id: string;
  type: string;
  title: string;
  message: string;
  severity: string;
  amount?: number;
  created_at: string;
}

export interface Transaction {
  id: string;
  amount: number;
  merchant_name: string;
  category: string;
  date: string;
  pending: boolean;
}

/* ------------------------------------------------------------------ */
/* API client                                                          */
/* ------------------------------------------------------------------ */

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE_URL}${path}`, init);
  if (!res.ok) {
    const body = await res.text().catch(() => "");
    throw new Error(`API ${res.status}: ${body}`);
  }
  return res.json() as Promise<T>;
}

export const api = {
  health: () => request<HealthResponse>("/health"),

  runAgent: (agentName: string, body: AgentRunRequest) =>
    request<AgentRunResponse>(`/agents/${agentName}/run`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    }),

  getTransactions: (userId: string) =>
    request<Transaction[]>(`/transactions?user_id=${userId}`),

  getGraphStats: () =>
    request<Record<string, unknown>>("/graph/stats"),

  getAlerts: (userId: string) =>
    request<Alert[]>(`/alerts?user_id=${userId}`),
};
