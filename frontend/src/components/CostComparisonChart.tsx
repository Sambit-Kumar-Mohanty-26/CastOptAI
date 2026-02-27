"use client";

import React from "react";
import {
    BarChart, Bar, XAxis, YAxis, CartesianGrid,
    Tooltip, ResponsiveContainer, Cell, LabelList,
} from "recharts";

interface CostComparisonChartProps {
    traditional: number;
    optimized: number;
}

export default function CostComparisonChart({ traditional, optimized }: CostComparisonChartProps) {
    const data = [
        { name: "Traditional", cost: Math.round(traditional), fill: "#94A3B8" },
        { name: "AI Optimized Mix", cost: Math.round(optimized), fill: "#0F172A" },
    ];

    const savings = traditional - optimized;
    const savingsPct = traditional > 0 ? ((savings / traditional) * 100).toFixed(0) : "0";

    return (
        <div className="w-full">
            <div className="h-[240px]">
                <ResponsiveContainer width="100%" height={240}>
                    <BarChart data={data} margin={{ top: 30, right: 20, left: 0, bottom: 5 }} barSize={64}>
                        <CartesianGrid strokeDasharray="3 3" stroke="#E2E8F0" vertical={false} />
                        <XAxis
                            dataKey="name"
                            tick={{ fontSize: 11, fontWeight: 800, fill: "#64748B", fontFamily: "var(--font-sans)" }}
                            stroke="#CBD5E1"
                            axisLine={{ strokeWidth: 2 }}
                            tickFormatter={(value) => value.toUpperCase()}
                        />
                        <YAxis
                            tick={{ fontSize: 11, fill: "#94A3B8", fontFamily: "var(--font-mono)", fontWeight: 700 }}
                            stroke="#CBD5E1"
                            axisLine={false}
                            tickLine={false}
                        />
                        <Tooltip
                            contentStyle={{
                                borderRadius: "4px",
                                border: "1px solid #0F172A",
                                boxShadow: "3px 3px 0px rgba(15,23,42,1)",
                                fontSize: 12,
                                background: "#FFFFFF",
                                color: "#0F172A",
                                fontWeight: 800,
                            }}
                            cursor={{ fill: "#F1F5F9" }}
                            formatter={(value) => [`₹${value}`]}
                        />
                        <Bar dataKey="cost" radius={[0, 0, 0, 0]}>
                            <Cell fill="#E2E8F0" stroke="#94A3B8" strokeWidth={1} />
                            <Cell fill="#FFCB05" stroke="#0F172A" strokeWidth={2} />
                            <LabelList
                                dataKey="cost"
                                position="top"
                                style={{ fontSize: 12, fontWeight: 800, fill: "#0F172A", fontFamily: "var(--font-mono)" }}
                                formatter={(v) => `₹${v}`}
                            />
                        </Bar>
                    </BarChart>
                </ResponsiveContainer>
            </div>
            {savings > 0 && (
                <div className="text-center mt-3">
                    <span className="inline-flex items-center gap-2 bg-[#0F172A] text-white text-[11px] font-bold px-3 py-1.5 rounded-full uppercase tracking-wider shadow-[2px_2px_0px_#CBD5E1]">
                        <span className="w-2 h-2 rounded-full bg-[#10B981] animate-pulse"></span>
                        Saving ₹{Math.round(savings)}/m³ ({savingsPct}%)
                    </span>
                </div>
            )}
        </div>
    );
}
