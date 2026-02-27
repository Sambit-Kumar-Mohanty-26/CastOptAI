"use client";

import React from "react";
import {
    Leaf,
    Zap,
    PiggyBank,
    ShieldCheck,
    Filter,
    Download,
} from "lucide-react";
import {
    AreaChart, Area, BarChart, Bar,
    XAxis, YAxis, CartesianGrid, Tooltip,
    ResponsiveContainer, Legend
} from "recharts";
import Sidebar from "@/components/Sidebar";
import Header from "@/components/Header";

const MONTHLY_DATA = [
    { month: "Jul", co2: 12400, energy: 8200, cost: 185000, jobs: 42 },
    { month: "Aug", co2: 11800, energy: 7900, cost: 178000, jobs: 45 },
    { month: "Sep", co2: 10500, energy: 7100, cost: 162000, jobs: 48 },
    { month: "Oct", co2: 9800, energy: 6500, cost: 155000, jobs: 50 },
    { month: "Nov", co2: 9200, energy: 6100, cost: 148000, jobs: 52 },
    { month: "Dec", co2: 8600, energy: 5700, cost: 140000, jobs: 55 },
];

const ESG_SCORES = [
    { label: "Carbon Intensity", value: "0.72", unit: "kg CO₂/m³", target: "< 0.80", status: "good" },
    { label: "Energy Efficiency", value: "85", unit: "%", target: "> 80%", status: "good" },
    { label: "Water Recycling Rate", value: "62", unit: "%", target: "> 70%", status: "warning" },
    { label: "Waste Diversion", value: "91", unit: "%", target: "> 85%", status: "good" },
];

export default function ESGReports() {
    return (
        <div className="flex h-screen w-full bg-transparent overflow-hidden">
            <Sidebar />
            <div className="flex-1 flex flex-col h-full bg-transparent">
                <Header showExport={true} onExport={() => window.print()} />
                <main className="flex-1 overflow-auto p-4 md:p-8">
                    <div className="max-w-[1400px] mx-auto w-full">
                        <div className="flex items-center justify-between mb-8 animate-assembly">
                            <div>
                                <h1 className="text-3xl font-extrabold text-[#0F172A] tracking-tight">Environmental & Economic Impact</h1>
                                <p className="text-[13px] text-[#64748B] font-medium mt-1">Live corporate ESG tracking across all optimization jobs.</p>
                            </div>
                            <div className="flex items-center gap-3">
                                <button className="flex items-center gap-2 px-4 py-2 bg-white border border-[#CBD5E1] text-[#0F172A] rounded-md text-[12px] font-bold shadow-[2px_2px_0px_#E2E8F0] hover:shadow-[3px_3px_0px_#CBD5E1] hover:-translate-y-px transition-all">
                                    <Filter className="w-4 h-4" /> YTD 2024
                                </button>
                                <button className="btn-primary">
                                    <Download className="w-4 h-4" /> Download PDF
                                </button>
                            </div>
                        </div>


                        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
                            {[
                                { label: "Total CO₂ Saved", value: "84.2", unit: "Tons", trend: "-12% vs LY", icon: Leaf, color: "text-[#10B981]", bg: "bg-[#F0FDF4]", border: "border-[#10B981]" },
                                { label: "Energy Reduced", value: "320", unit: "MWh", trend: "-8% vs LY", icon: Zap, color: "text-[#D97706]", bg: "bg-[#FEFCE8]", border: "border-[#FDE047]" },
                                { label: "Material Cost Saved", value: "₹2.4", unit: "Cr", trend: "+15% vs LY", icon: PiggyBank, color: "text-[#0F172A]", bg: "bg-[#F1F5F9]", border: "border-[#94A3B8]" },
                                { label: "Target Compliance", value: "98.5", unit: "%", trend: "On Track", icon: ShieldCheck, color: "text-[#2563EB]", bg: "bg-[#EFF6FF]", border: "border-[#93C5FD]" },
                            ].map((stat, i) => (
                                <div key={i} className={`card p-5 animate-assembly delay-${(i + 1) * 100} shadow-[3px_3px_0px_#E2E8F0]`}>
                                    <div className="flex justify-between items-start mb-4">
                                        <div className={`p-2 rounded-lg ${stat.bg} ${stat.color}`}>
                                            <stat.icon className="w-5 h-5" />
                                        </div>
                                        <span className={`text-[10px] font-bold px-2 py-0.5 rounded-full border ${stat.border} ${stat.color} bg-white bg-opacity-50 uppercase tracking-widest`}>
                                            {stat.trend}
                                        </span>
                                    </div>
                                    <p className="text-[12px] font-bold text-[#64748B] uppercase tracking-widest">{stat.label}</p>
                                    <p className="text-3xl font-extrabold text-[#0F172A] font-mono-data tracking-tight mt-1">
                                        {stat.value}
                                        <span className="text-[14px] font-medium text-[#64748B] ml-1">{stat.unit}</span>
                                    </p>
                                </div>
                            ))}
                        </div>

                        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 mb-8">
                            <div className="card animate-assembly delay-500 overflow-hidden">
                                <div className="px-5 py-4 border-b border-[#E2E8F0] bg-[#F8FAFC]">
                                    <h3 className="text-[12px] font-extrabold text-[#0F172A] uppercase tracking-widest">CO₂ Emissions Trend (Kg)</h3>
                                </div>
                                <div className="p-6 h-[280px]">
                                    <ResponsiveContainer width="100%" height={280}>
                                        <AreaChart data={MONTHLY_DATA}>
                                            <defs>
                                                <linearGradient id="colorCo2" x1="0" y1="0" x2="0" y2="1">
                                                    <stop offset="5%" stopColor="#10B981" stopOpacity={0.2} />
                                                    <stop offset="95%" stopColor="#10B981" stopOpacity={0} />
                                                </linearGradient>
                                            </defs>
                                            <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#E2E8F0" />
                                            <XAxis dataKey="month" tick={{ fontSize: 11, fontWeight: 700, fill: "#64748B", fontFamily: "var(--font-mono)" }} stroke="#CBD5E1" />
                                            <YAxis tick={{ fontSize: 11, fontWeight: 700, fill: "#64748B", fontFamily: "var(--font-mono)" }} stroke="#CBD5E1" />
                                            <Tooltip contentStyle={{ borderRadius: "8px", border: "1px solid #0F172A", boxShadow: "4px 4px 0px rgba(15,23,42,1)", fontSize: 12, fontWeight: 800, fontFamily: "var(--font-mono)" }} />
                                            <Area type="monotone" dataKey="co2" stroke="#10B981" strokeWidth={3} fillOpacity={1} fill="url(#colorCo2)" />
                                        </AreaChart>
                                    </ResponsiveContainer>
                                </div>
                            </div>
                            <div className="card animate-assembly delay-600 overflow-hidden">
                                <div className="px-5 py-4 border-b border-[#E2E8F0] bg-[#F8FAFC]">
                                    <h3 className="text-[12px] font-extrabold text-[#0F172A] uppercase tracking-widest">Energy Usage vs Cost Savings</h3>
                                </div>
                                <div className="p-6 h-[280px]">
                                    <ResponsiveContainer width="100%" height={280}>
                                        <BarChart data={MONTHLY_DATA}>
                                            <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#E2E8F0" />
                                            <XAxis dataKey="month" tick={{ fontSize: 11, fontWeight: 700, fill: "#64748B", fontFamily: "var(--font-mono)" }} stroke="#CBD5E1" />
                                            <YAxis yAxisId="left" tick={{ fontSize: 11, fontWeight: 700, fill: "#64748B", fontFamily: "var(--font-mono)" }} stroke="#CBD5E1" />
                                            <YAxis yAxisId="right" orientation="right" tick={{ fontSize: 11, fontWeight: 700, fill: "#64748B", fontFamily: "var(--font-mono)" }} stroke="#CBD5E1" />
                                            <Tooltip contentStyle={{ borderRadius: "8px", border: "1px solid #0F172A", boxShadow: "4px 4px 0px rgba(15,23,42,1)", fontSize: 12, fontWeight: 800, fontFamily: "var(--font-mono)" }} cursor={{ fill: "#F1F5F9" }} />
                                            <Legend wrapperStyle={{ fontSize: 11, fontWeight: 800, fontFamily: "var(--font-sans)", color: "#64748B" }} />
                                            <Bar yAxisId="left" dataKey="energy" fill="#0F172A" name="Energy (kWh)" radius={[4, 4, 0, 0]} />
                                            <Bar yAxisId="right" dataKey="cost" fill="#FFCB05" name="Cost (₹)" radius={[4, 4, 0, 0]} />
                                        </BarChart>
                                    </ResponsiveContainer>
                                </div>
                            </div>
                        </div>

                    </div>
                </main>
            </div>
        </div>
    );
}
