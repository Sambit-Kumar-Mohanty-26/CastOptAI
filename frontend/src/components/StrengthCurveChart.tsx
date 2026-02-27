"use client";

import React from "react";
import {
    LineChart, Line, XAxis, YAxis, CartesianGrid,
    Tooltip, ReferenceLine, ResponsiveContainer, Legend,
} from "recharts";

interface DataPoint { hour: number; strength: number; }

interface StrengthCurveChartProps {
    data: DataPoint[];
    baselineData?: DataPoint[];
    targetStrength: number;
    targetTime: number;
}

export default function StrengthCurveChart({
    data, baselineData, targetStrength, targetTime,
}: StrengthCurveChartProps) {
    const merged = data.map((d, i) => ({
        hour: d.hour,
        optimized: d.strength,
        baseline: baselineData?.[i]?.strength ?? null,
    }));

    return (
        <div className="w-full h-[280px]">
            <ResponsiveContainer width="100%" height={280}>
                <LineChart data={merged} margin={{ top: 10, right: 20, left: 0, bottom: 5 }}>
                    <CartesianGrid strokeDasharray="2 2" stroke="#E2E8F0" vertical={false} />
                    <XAxis
                        dataKey="hour"
                        tick={{ fontSize: 11, fill: "#64748B", fontFamily: "var(--font-mono)", fontWeight: 700 }}
                        label={{ value: "Curing Time (Hours)", position: "insideBottomRight", offset: -5, fontSize: 10, fill: "#64748B", fontWeight: 700, textAnchor: "end" }}
                        stroke="#CBD5E1"
                        tickMargin={8}
                    />
                    <YAxis
                        tick={{ fontSize: 11, fill: "#64748B", fontFamily: "var(--font-mono)", fontWeight: 700 }}
                        label={{ value: "Strength (MPa)", angle: -90, position: "insideLeft", fontSize: 10, fill: "#64748B", fontWeight: 700 }}
                        stroke="#CBD5E1"
                        tickMargin={8}
                    />
                    <Tooltip
                        contentStyle={{
                            borderRadius: "8px",
                            border: "1px solid #0F172A",
                            boxShadow: "4px 4px 0px rgba(15,23,42,1)",
                            fontSize: 12,
                            background: "#FFFFFF",
                            color: "#0F172A",
                            fontFamily: "var(--font-mono)",
                            fontWeight: 700,
                            padding: "8px 12px"
                        }}
                        itemStyle={{ color: "#0F172A", fontWeight: 800 }}
                        formatter={(value) => [`${value} MPa`]}
                        cursor={{ stroke: '#0F172A', strokeWidth: 1, strokeDasharray: '4 4' }}
                    />
                    <Legend wrapperStyle={{ fontSize: 10, color: "#64748B", fontWeight: 800, marginTop: "10px" }} iconType="circle" />
                    <ReferenceLine
                        y={targetStrength}
                        stroke="#EF4444"
                        strokeDasharray="4 4"
                        strokeWidth={1.5}
                        label={{ value: `TARGET: ${targetStrength} MPa`, position: "right", fill: "#EF4444", fontSize: 10, fontWeight: 800, fontFamily: "var(--font-sans)" }}
                    />
                    <ReferenceLine
                        x={targetTime}
                        stroke="#0F172A"
                        strokeDasharray="4 4"
                        strokeWidth={1}
                        label={{ value: `T+${targetTime}h`, position: "top", fill: "#0F172A", fontSize: 10, fontWeight: 800, fontFamily: "var(--font-mono)" }}
                    />
                    {baselineData && (
                        <Line name="Traditional Mix" type="stepAfter" dataKey="baseline"
                            stroke="#94A3B8" strokeWidth={2} strokeDasharray="4 4" dot={false} />
                    )}
                    {/* The Bold Optimized Path */}
                    <Line name="AI Optimized Mix" type="monotone" dataKey="optimized"
                        stroke="#0F172A" strokeWidth={3} dot={false}
                        activeDot={{ r: 5, strokeWidth: 2, fill: "#FFCB05", stroke: "#0F172A" }} />
                </LineChart>
            </ResponsiveContainer>
        </div>
    );
}
