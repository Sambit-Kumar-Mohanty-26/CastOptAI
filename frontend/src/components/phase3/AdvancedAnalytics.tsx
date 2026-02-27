"use client";

import React, { useState } from "react";
import { 
  BarChart, Bar, XAxis, YAxis, CartesianGrid, 
  Tooltip, ResponsiveContainer, PieChart, Pie, Cell,
  LineChart, Line
} from "recharts";

interface Insight {
  title: string;
  description: string;
  severity: 'high' | 'medium' | 'low';
  recommendations: string[];
  data_points: Record<string, any>;
}

interface ClusterData {
  size: number;
  avg_values: Record<string, number>;
  characteristics: string;
}

interface AdvancedAnalyticsProps {
  insights?: Insight[];
  clusters?: Record<string, ClusterData>;
  performanceReport?: Record<string, any>;
}

export default function AdvancedAnalyticsPanel({ 
  insights, 
  clusters, 
  performanceReport 
}: AdvancedAnalyticsProps) {
  const [analysisType, setAnalysisType] = useState<'insights' | 'clusters' | 'performance'>('insights');
  
  // Demo data if no real data provided
  const demoInsights: Insight[] = [
    {
      title: "Cost Optimization Trend",
      description: "Recent costs are 8.2% lower than historical average",
      severity: 'medium',
      recommendations: [
        "Maintain current optimization strategies",
        "Document successful approaches for replication"
      ],
      data_points: {
        recent_average: 4200,
        historical_average: 4550,
        trend_percentage: -8.2,
        sample_size: 45
      }
    },
    {
      title: "Cement Usage Pattern",
      description: "Current cement value (310.0) vs. average (305.2)",
      severity: 'low',
      recommendations: [
        "Cement usage is within normal range"
      ],
      data_points: {
        current_value: 310.0,
        average_value: 305.2,
        std_deviation: 12.3,
        deviation_sigma: 0.4
      }
    }
  ];

  const demoClusters = {
    'cluster_0': {
      size: 25,
      avg_values: {
        target_strength: 22.5,
        cost: 4800,
        predicted_strength: 24.2
      },
      characteristics: "Characterized by: target_strength (22.5), predicted_strength (24.2), cost (4800.0)"
    },
    'cluster_1': {
      size: 18,
      avg_values: {
        target_strength: 18.2,
        cost: 3900,
        predicted_strength: 19.8
      },
      characteristics: "Characterized by: cost (3900.0), target_strength (18.2), predicted_strength (19.8)"
    }
  };

  const demoPerformanceReport = {
    summary: {
      total_optimizations: 67,
      date_range: {
        start: "2024-01-01",
        end: "2024-02-27"
      }
    },
    cost_analysis: {
      average_cost: 4520,
      cost_std_dev: 320,
      min_cost: 3800,
      max_cost: 5200
    },
    efficiency_metrics: {
      avg_savings_percent: 22.5,
      avg_cost_per_strength: 198
    },
    recommendations: [
      "Continue current optimization approach",
      "Monitor for seasonal pricing effects"
    ]
  };

  const currentInsights = insights || demoInsights;
  const currentClusters = clusters || demoClusters;
  const currentReport = performanceReport || demoPerformanceReport;

  // Prepare chart data
  const clusterChartData = Object.entries(currentClusters).map(([key, cluster]) => ({
    name: key,
    size: cluster.size,
    avg_cost: cluster.avg_values?.cost || 0,
    avg_strength: cluster.avg_values?.predicted_strength || 0
  }));

  const recommendationColors: Record<string, string> = {
    'high': '#EF4444',    // red
    'medium': '#F59E0B',  // amber
    'low': '#10B981'      // green
  };

  return (
    <div className="border rounded-lg p-4 bg-white shadow-sm">
      <div className="mb-4">
        <h3 className="text-lg font-bold text-gray-800 mb-2">Advanced Analytics</h3>
        <div className="flex flex-wrap gap-2 mb-3">
          <span className="inline-block bg-purple-100 text-purple-800 text-xs px-2 py-1 rounded font-bold">
            Phase 3
          </span>
          <select 
            value={analysisType}
            onChange={(e) => setAnalysisType(e.target.value as any)}
            className="text-xs border rounded px-2 py-1"
          >
            <option value="insights">Pattern Insights</option>
            <option value="clusters">Project Clusters</option>
            <option value="performance">Performance Report</option>
          </select>
        </div>
      </div>

      {analysisType === 'insights' && (
        <div className="space-y-4">
          <h4 className="font-semibold text-gray-700">Detected Patterns</h4>
          {currentInsights.map((insight, index) => (
            <div key={index} className={`border-l-4 pl-4 py-2 ${
              insight.severity === 'high' ? 'border-red-500 bg-red-50' :
              insight.severity === 'medium' ? 'border-yellow-500 bg-yellow-50' :
              'border-green-500 bg-green-50'
            }`}>
              <div className="flex justify-between items-start">
                <h5 className="font-bold text-gray-800">{insight.title}</h5>
                <span className={`text-xs px-2 py-1 rounded-full ${
                  insight.severity === 'high' ? 'bg-red-200 text-red-800' :
                  insight.severity === 'medium' ? 'bg-yellow-200 text-yellow-800' :
                  'bg-green-200 text-green-800'
                }`}>
                  {insight.severity.toUpperCase()}
                </span>
              </div>
              <p className="text-sm text-gray-600 mt-1">{insight.description}</p>
              <div className="mt-2">
                <h6 className="text-xs font-semibold text-gray-700">Recommendations:</h6>
                <ul className="text-xs text-gray-600 list-disc list-inside">
                  {insight.recommendations.map((rec, i) => (
                    <li key={i}>{rec}</li>
                  ))}
                </ul>
              </div>
            </div>
          ))}
        </div>
      )}

      {analysisType === 'clusters' && (
        <div>
          <h4 className="font-semibold text-gray-700 mb-3">Project Clusters Analysis</h4>
          
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-4">
            {Object.entries(currentClusters).map(([key, cluster]) => (
              <div key={key} className="border rounded p-3">
                <h5 className="font-bold text-gray-800">{key.replace('_', ' ')}</h5>
                <p className="text-sm text-gray-600">{cluster.characteristics}</p>
                <div className="mt-2 text-xs">
                  <p>Size: {cluster.size} projects</p>
                  <p>Avg Cost: ₹{cluster.avg_values?.cost?.toFixed(0) || 0}</p>
                  <p>Avg Strength: {cluster.avg_values?.predicted_strength?.toFixed(1) || 0} MPa</p>
                </div>
              </div>
            ))}
          </div>

          <div className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={clusterChartData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#E2E8F0" />
                <XAxis dataKey="name" tick={{ fontSize: 10, fill: "#64748B" }} />
                <YAxis tick={{ fontSize: 10, fill: "#64748B" }} />
                <Tooltip 
                  formatter={(value) => [Number(value).toFixed(2), 'Value']}
                  labelFormatter={(value) => `Cluster: ${value}`}
                />
                <Bar dataKey="size" name="Project Count" fill="#8884d8" />
                <Bar dataKey="avg_cost" name="Avg Cost" fill="#00C49F" />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      )}

      {analysisType === 'performance' && (
        <div>
          <h4 className="font-semibold text-gray-700 mb-3">Performance Summary</h4>
          
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-4">
            <div className="border rounded p-3 bg-blue-50">
              <div className="text-xs font-semibold text-blue-800">Total Optimizations</div>
              <div className="text-lg font-bold text-blue-600">{currentReport?.summary?.total_optimizations || 0}</div>
            </div>
            <div className="border rounded p-3 bg-green-50">
              <div className="text-xs font-semibold text-green-800">Avg Savings %</div>
              <div className="text-lg font-bold text-green-600">{currentReport?.efficiency_metrics?.avg_savings_percent?.toFixed(1) || 0}%</div>
            </div>
            <div className="border rounded p-3 bg-purple-50">
              <div className="text-xs font-semibold text-purple-800">Avg Cost/Strength</div>
              <div className="text-lg font-bold text-purple-600">₹{currentReport?.efficiency_metrics?.avg_cost_per_strength?.toFixed(0) || 0}</div>
            </div>
          </div>

          <div className="h-64 mb-4">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={[
                { name: 'Jan', savings: 18, cost: 4600 },
                { name: 'Feb', savings: 22, cost: 4400 },
                { name: 'Mar', savings: 25, cost: 4200 },
                { name: 'Apr', savings: 20, cost: 4500 },
                { name: 'May', savings: 23, cost: 4300 },
              ]}>
                <CartesianGrid strokeDasharray="3 3" stroke="#E2E8F0" />
                <XAxis dataKey="name" tick={{ fontSize: 10, fill: "#64748B" }} />
                <YAxis yAxisId="left" tick={{ fontSize: 10, fill: "#64748B" }} />
                <YAxis yAxisId="right" orientation="right" tick={{ fontSize: 10, fill: "#64748B" }} />
                <Tooltip />
                <Line yAxisId="left" type="monotone" dataKey="savings" stroke="#10B981" name="Savings %" />
                <Line yAxisId="right" type="monotone" dataKey="cost" stroke="#3B82F6" name="Cost" />
              </LineChart>
            </ResponsiveContainer>
          </div>

          <div>
            <h5 className="font-semibold text-gray-700 mb-2">Recommendations</h5>
            <ul className="text-sm text-gray-600 list-disc list-inside">
              {(currentReport.recommendations || []).map((rec: string, i: number) => (
                <li key={i}>{rec}</li>
              ))}
            </ul>
          </div>
        </div>
      )}
    </div>
  );
}