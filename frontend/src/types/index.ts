/* ------------------------------------------------------------------ */
/* Shared TypeScript types used across the Foresight frontend          */
/* ------------------------------------------------------------------ */

export interface User {
  id: string;
  name: string;
  email: string;
  created_at: string;
}

export interface Transaction {
  id: string;
  amount: number;
  merchant_name: string;
  category: string;
  date: string;
  pending: boolean;
  source?: string;
}

export interface Alert {
  id: string;
  type: string;
  title: string;
  message: string;
  severity: "critical" | "high" | "medium" | "low" | "info";
  amount?: number;
  created_at: string;
  status?: string;
}

export interface Goal {
  goal_id: string;
  goal_name: string;
  target_amount: number;
  current_amount: number;
  percent_complete: number;
  amount_remaining: number;
  deadline: string | null;
  days_remaining: number;
  required_monthly_savings: number;
  estimated_monthly_savings: number;
  projected_completion_date: string;
  status: "on_track" | "behind" | "ahead" | "completed" | "overdue";
}

export interface Subscription {
  service_name: string;
  amount: number;
  frequency: string;
  source: "bank" | "email" | "both";
  last_charged: string;
  next_expected: string;
  confidence_score: number;
}

export interface CashflowProjection {
  date: string;
  projected_balance: number;
  events?: string[];
}

export interface CashflowForecast {
  daily_projections: CashflowProjection[];
  balance_30d: number;
  balance_60d: number;
  lowest_point: number;
  lowest_point_date: string;
  days_until_negative: number | null;
  days_below_500: number | null;
  average_daily_net: number;
}

export interface AgentRun {
  status: string;
  session_id: string;
  output: Record<string, unknown>;
  steps: string[];
  error?: string;
}

export interface HealthScore {
  score: number;
  cashflow_risk_level: string;
  goals_on_track: number;
  goals_behind: number;
  total_goals: number;
  unusual_transactions: number;
  potential_savings: number;
  week_spending_total: number;
  current_balance: number;
}
