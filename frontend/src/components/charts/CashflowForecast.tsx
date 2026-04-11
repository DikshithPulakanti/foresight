"use client";

import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Filler,
  Tooltip,
  type ChartData,
  type ChartOptions,
} from "chart.js";
import { Line } from "react-chartjs-2";

ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Filler,
  Tooltip,
);

interface Projection {
  date: string;
  projected_balance: number;
}

interface CashflowForecastProps {
  projections: Projection[];
}

function balanceColor(balance: number): string {
  if (balance < 100) return "rgb(239, 68, 68)";   // red-500
  if (balance < 500) return "rgb(234, 179, 8)";    // yellow-500
  return "rgb(16, 185, 129)";                      // emerald-500
}

export default function CashflowForecast({ projections }: CashflowForecastProps) {
  if (projections.length === 0) {
    return (
      <div className="flex items-center justify-center h-64 text-gray-500 text-sm">
        No forecast data available
      </div>
    );
  }

  const labels = projections.map((p) =>
    new Date(p.date).toLocaleDateString("en-US", { month: "short", day: "numeric" }),
  );

  const balances = projections.map((p) => p.projected_balance);

  const lastBalance = balances[balances.length - 1] ?? 0;
  const lineColor = balanceColor(Math.min(...balances));

  const data: ChartData<"line"> = {
    labels,
    datasets: [
      {
        data: balances,
        borderColor: lineColor,
        backgroundColor: `${lineColor.replace("rgb", "rgba").replace(")", ", 0.1)")}`,
        borderWidth: 2,
        pointRadius: 0,
        pointHitRadius: 8,
        fill: true,
        tension: 0.3,
      },
    ],
  };

  const options: ChartOptions<"line"> = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      tooltip: {
        callbacks: {
          label: (ctx) => `$${(ctx.parsed.y ?? 0).toLocaleString()}`,
        },
      },
    },
    scales: {
      x: {
        ticks: { color: "#6b7280", maxTicksLimit: 8 },
        grid: { color: "rgba(75, 85, 99, 0.2)" },
      },
      y: {
        ticks: {
          color: "#6b7280",
          callback: (v) => `$${Number(v).toLocaleString()}`,
        },
        grid: { color: "rgba(75, 85, 99, 0.2)" },
      },
    },
  };

  const lowest = Math.min(...balances);

  return (
    <div>
      <div className="h-64">
        <Line data={data} options={options} />
      </div>
      <div className="mt-3 flex items-center gap-4 text-xs text-gray-400">
        <span>
          60-day projection:{" "}
          <span className="font-medium text-white">
            ${lastBalance.toLocaleString()}
          </span>
        </span>
        <span>
          Lowest point:{" "}
          <span
            className={`font-medium ${lowest < 100 ? "text-red-400" : lowest < 500 ? "text-yellow-400" : "text-emerald-400"}`}
          >
            ${lowest.toLocaleString()}
          </span>
        </span>
      </div>
    </div>
  );
}
