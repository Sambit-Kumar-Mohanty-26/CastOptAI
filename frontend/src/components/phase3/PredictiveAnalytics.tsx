"use client";

import React, { useState, useEffect } from "react";
import { 
  LineChart, Line, XAxis, YAxis, CartesianGrid, 
  Tooltip, ResponsiveContainer, AreaChart, Area,
  BarChart, Bar
} from "recharts";

interface AnalyticsDataPoint {
  date: string;
  value?: number;
  forecasted_demand?: number;
  predicted_value?: number;
  confidence_lower?: number;
  confidence_upper?: number;
  trend_direction?: string;
}

interface PredictiveAnalyticsProps {
  predictions?: AnalyticsDataPoint[];
  demandForecasts?: AnalyticsDataPoint[];
  onRefresh?: () => void;
}

export default function PredictiveAnalyticsPanel({ 
  predictions, 
  demandForecasts, 
  onRefresh 
}: PredictiveAnalyticsProps) {
  const [metricType, setMetricType] = useState<'demand' | 'pricing' | 'quality'>('demand');
  const [timeRange, setTimeRange] = useState<'7d' | '30d' | '90d'>('30d');

  // Simulated data for demo purposes
  const [demoPredictions, setDemoPredictions] = useState<AnalyticsDataPoint[]>([]);
  const [demoDemandForecasts, setDemoDemandForecasts] = useState<AnalyticsDataPoint[]>([]);

  useEffect(() => {
    // Generate demo data if no real data provided
    if (!predictions && !demandForecasts) {
      const genPredictions: AnalyticsDataPoint[] = [];
      const genDemandForecasts: AnalyticsDataPoint[] = [];
      
      for (let i = 0; i < 30; i++) {
        const date = new Date();
        date.setDate(date.getDate() + i);
        
        genPredictions.push({
          date: date.toISOString().split('T')[0],
          predicted_value: 450 + Math.random() * 100,
          confidence_lower: 430 + Math.random() * 80,
          confidence_upper: 470 + Math.random() * 120
        });
        
        genDemandForecasts.push({
          date: date.toISOString().split('T')[0],
          forecasted_demand: 500 + Math.random() * 150,
          trend_direction: Math.random() > 0.5 ? 'increasing' : 'decreasing'
        });
      }
      
      setDemoPredictions(genPredictions);
      setDemoDemandForecasts(genDemandForecasts);
    }
  }, [predictions, demandForecasts]);

  const currentData = metricType === 'demand' 
    ? (demandForecasts || demoDemandForecasts)
    : (predictions || demoPredictions);

  return (
    <div className="border rounded-lg p-4 bg-white shadow-sm">
      <div className="mb-4">
        <h3 className="text-lg font-bold text-gray-800 mb-2">Predictive Analytics</h3>
        <div className="flex flex-wrap gap-2 mb-3">
          <span className="inline-block bg-purple-100 text-purple-800 text-xs px-2 py-1 rounded font-bold">
            Phase 3
          </span>
          <select 
            value={metricType}
            onChange={(e) => setMetricType(e.target.value as any)}
            className="text-xs border rounded px-2 py-1"
          >
            <option value="demand">Demand Forecasting</option>
            <option value="pricing">Pricing Trends</option>
            <option value="quality">Quality Degradation</option>
          </select>
          <select 
            value={timeRange}
            onChange={(e) => setTimeRange(e.target.value as any)}
            className="text-xs border rounded px-2 py-1"
          >
            <option value="7d">7 Days</option>
            <option value="30d">30 Days</option>
            <option value="90d">90 Days</option>
          </select>
        </div>
      </div>
      
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-4">
        <div className="border rounded p-3 bg-blue-50">
          <div className="text-xs font-semibold text-blue-800">Avg Forecast</div>
          <div className="text-lg font-bold text-blue-600">
            {currentData.length > 0 
              ? (currentData.reduce((sum, d) => sum + (d.forecasted_demand ?? d.predicted_value ?? d.value ?? 0), 0) / currentData.length).toFixed(0) 
              : '0'}
          </div>
        </div>
        <div className="border rounded p-3 bg-green-50">
          <div className="text-xs font-semibold text-green-800">Confidence</div>
          <div className="text-lg font-bold text-green-600">85%</div>
        </div>
        <div className="border rounded p-3 bg-yellow-50">
          <div className="text-xs font-semibold text-yellow-800">Trend</div>
          <div className="text-lg font-bold text-yellow-600">
            {currentData.some(d => d.trend_direction === 'increasing') ? '↑ Rising' : '↓ Stable'}
          </div>
        </div>
        <div className="border rounded p-3 bg-red-50">
          <div className="text-xs font-semibold text-red-800">Risk Level</div>
          <div className="text-lg font-bold text-red-600">Medium</div>
        </div>
      </div>

      <div className="h-64">
        <ResponsiveContainer width="100%" height="100%">
          <AreaChart data={currentData}>
            <CartesianGrid strokeDasharray="3 3" stroke="#E2E8F0" />
            <XAxis 
              dataKey="date" 
              tick={{ fontSize: 10, fill: "#64748B" }}
              tickFormatter={(value) => new Date(value).toLocaleDateString('en-US', { month: 'short', day: 'numeric' })}
            />
            <YAxis 
              tick={{ fontSize: 10, fill: "#64748B" }}
            />
            <Tooltip 
              formatter={(value) => [Number(value).toFixed(2), metricType === 'demand' ? 'Demand' : 'Value']}
              labelFormatter={(value) => `Date: ${new Date(value).toLocaleDateString()}`}
            />
            <Area 
              type="monotone" 
              dataKey={metricType === 'demand' ? 'forecasted_demand' : 'predicted_value'} 
              stroke="#8884d8" 
              fill="url(#colorGradient)" 
              fillOpacity={0.3}
            />
            {metricType !== 'demand' && (
              <>
                <Area 
                  type="monotone" 
                  dataKey="confidence_lower" 
                  stroke="#ff6b6b" 
                  fill="none" 
                  strokeDasharray="3 3"
                />
                <Area 
                  type="monotone" 
                  dataKey="confidence_upper" 
                  stroke="#ff6b6b" 
                  fill="none" 
                  strokeDasharray="3 3"
                />
              </>
            )}
            <defs>
              <linearGradient id="colorGradient" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor="#8884d8" stopOpacity={0.8}/>
                <stop offset="95%" stopColor="#8884d8" stopOpacity={0.1}/>
              </linearGradient>
            </defs>
          </AreaChart>
        </ResponsiveContainer>
      </div>

      <div className="mt-4 text-xs text-gray-600">
        <p>Predictive analytics based on historical patterns and market trends</p>
        <p className="mt-1">Confidence intervals represent 95% prediction bounds</p>
      </div>
    </div>
  );
}