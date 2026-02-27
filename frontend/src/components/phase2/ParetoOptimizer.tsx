"use client";

import React, { useState } from "react";
import { 
  LineChart, Line, XAxis, YAxis, CartesianGrid, 
  Tooltip, ResponsiveContainer, Scatter, ScatterChart 
} from "recharts";

interface ObjectiveWeights {
  cost: number;
  time: number;
  strength: number;
  environmental: number;
}

interface ParetoSolution {
  id: number;
  recipe: {
    cement: number;
    chemicals: number;
    steam_hours: number;
    water: number;
  };
  cost: number;
  strength_score: number;
  environmental_score: number;
  time_score: number;
  constraints_satisfied: boolean;
}

interface ParetoOptimizerProps {
  onOptimize: (weights: ObjectiveWeights) => void;
  paretoData?: ParetoSolution[];
}

export default function ParetoOptimizerPanel({ onOptimize, paretoData }: ParetoOptimizerProps) {
  const [weights, setWeights] = useState<ObjectiveWeights>({
    cost: 0.4,
    time: 0.3,
    strength: 0.2,
    environmental: 0.1
  });

  const handleWeightChange = (key: keyof ObjectiveWeights, value: number[]) => {
    const newWeights = { ...weights, [key]: value[0] };
    // Normalize weights to sum to 1
    const total = Object.values(newWeights).reduce((sum, val) => sum + val, 0);
    const normalizedWeights = {
      cost: newWeights.cost / total,
      time: newWeights.time / total,
      strength: newWeights.strength / total,
      environmental: newWeights.environmental / total
    };
    setWeights(normalizedWeights);
  };

  const handleOptimize = () => {
    onOptimize(weights);
  };

  return (
    <div className="border rounded-lg p-4 bg-white shadow-sm">
      <div className="mb-4">
        <h3 className="text-lg font-bold text-gray-800 mb-2">Multi-Objective Optimization</h3>
        <span className="inline-block bg-yellow-100 text-gray-800 text-xs px-2 py-1 rounded font-bold">
          Phase 2
        </span>
      </div>
      
      {/* Weight Sliders */}
      <div className="space-y-4 mb-4">
        <div>
          <div className="flex justify-between text-sm mb-1">
            <span className="font-medium text-gray-700">Cost Priority</span>
            <span className="font-mono font-bold text-gray-800">{weights.cost.toFixed(2)}</span>
          </div>
          <input
            type="range"
            min="0"
            max="1"
            step="0.01"
            value={weights.cost}
            onChange={(e) => handleWeightChange('cost', [parseFloat(e.target.value)])}
            className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer"
          />
        </div>

        <div>
          <div className="flex justify-between text-sm mb-1">
            <span className="font-medium text-gray-700">Time Priority</span>
            <span className="font-mono font-bold text-gray-800">{weights.time.toFixed(2)}</span>
          </div>
          <input
            type="range"
            min="0"
            max="1"
            step="0.01"
            value={weights.time}
            onChange={(e) => handleWeightChange('time', [parseFloat(e.target.value)])}
            className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer"
          />
        </div>

        <div>
          <div className="flex justify-between text-sm mb-1">
            <span className="font-medium text-gray-700">Strength Priority</span>
            <span className="font-mono font-bold text-gray-800">{weights.strength.toFixed(2)}</span>
          </div>
          <input
            type="range"
            min="0"
            max="1"
            step="0.01"
            value={weights.strength}
            onChange={(e) => handleWeightChange('strength', [parseFloat(e.target.value)])}
            className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer"
          />
        </div>

        <div>
          <div className="flex justify-between text-sm mb-1">
            <span className="font-medium text-gray-700">Environmental Priority</span>
            <span className="font-mono font-bold text-gray-800">{weights.environmental.toFixed(2)}</span>
          </div>
          <input
            type="range"
            min="0"
            max="1"
            step="0.01"
            value={weights.environmental}
            onChange={(e) => handleWeightChange('environmental', [parseFloat(e.target.value)])}
            className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer"
          />
        </div>
      </div>

      {/* Optimize Button */}
      <button 
        onClick={handleOptimize}
        className="w-full bg-gray-800 hover:bg-gray-900 text-white font-bold py-3 px-4 rounded transition-all duration-300"
      >
        Find Pareto Optimal Solutions
      </button>

      {/* Pareto Front Visualization */}
      {paretoData && paretoData.length > 0 && (
        <div className="mt-6">
          <h4 className="text-sm font-bold text-gray-800 mb-3">Pareto Frontier</h4>
          <div className="h-64">
            <ResponsiveContainer width="100%" height={260}>
              <ScatterChart margin={{ top: 20, right: 20, bottom: 20, left: 20 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#E2E8F0" />
                <XAxis 
                  dataKey="cost" 
                  type="number"
                  tick={{ fontSize: 10, fill: "#64748B" }}
                  label={{ value: 'Cost (₹)', position: 'insideBottomRight', offset: -5 }}
                />
                <YAxis 
                  dataKey="strength_score" 
                  type="number"
                  tick={{ fontSize: 10, fill: "#64748B" }}
                  label={{ value: 'Strength Score', angle: -90, position: 'insideLeft' }}
                />
                <Tooltip />
                <Scatter 
                  data={paretoData} 
                  fill="#FFCB05" 
                  stroke="#0F172A"
                  strokeWidth={1}
                />
              </ScatterChart>
            </ResponsiveContainer>
          </div>
        </div>
      )}
    </div>
  );
}