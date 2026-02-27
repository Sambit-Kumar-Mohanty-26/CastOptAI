"use client";

import React from "react";

interface RiskAssessmentProps {
  riskData?: any;
}

export default function RiskAssessmentPanel({ riskData }: RiskAssessmentProps) {
  if (!riskData) {
    return (
      <div className="border rounded-lg p-4 bg-white shadow-sm">
        <div className="mb-4">
          <h3 className="text-lg font-bold text-gray-800 mb-2">Risk Assessment</h3>
          <span className="inline-block bg-yellow-100 text-gray-800 text-xs px-2 py-1 rounded font-bold">
            Phase 2
          </span>
        </div>
        <p className="text-gray-600 text-sm">Run optimization to see risk assessment</p>
      </div>
    );
  }

  const { risk_assessment, monte_carlo_results, emergency_triggers } = riskData;

  return (
    <div className="border rounded-lg p-4 bg-white shadow-sm">
      <div className="mb-4">
        <h3 className="text-lg font-bold text-gray-800 mb-2">Risk Assessment</h3>
        <span className="inline-block bg-yellow-100 text-gray-800 text-xs px-2 py-1 rounded font-bold">
          Phase 2
        </span>
      </div>
      
      <div className="grid grid-cols-2 gap-4 mb-4">
        <div className="border rounded p-3 bg-red-50">
          <div className="text-xs font-semibold text-red-800">Overall Risk</div>
          <div className="text-lg font-bold text-red-600">{risk_assessment?.overall_risk?.toFixed(1)}%</div>
        </div>
        <div className="border rounded p-3 bg-green-50">
          <div className="text-xs font-semibold text-green-800">Success Probability</div>
          <div className="text-lg font-bold text-green-600">{(monte_carlo_results?.probability_of_success * 100)?.toFixed(1)}%</div>
        </div>
      </div>

      <div className="space-y-3">
        <div className="flex justify-between text-sm">
          <span className="text-gray-600">Strength Risk:</span>
          <span className="font-bold">{risk_assessment?.strength_risk?.toFixed(1)}%</span>
        </div>
        <div className="flex justify-between text-sm">
          <span className="text-gray-600">Cost Risk:</span>
          <span className="font-bold">{risk_assessment?.cost_risk?.toFixed(1)}%</span>
        </div>
        <div className="flex justify-between text-sm">
          <span className="text-gray-600">Time Risk:</span>
          <span className="font-bold">{risk_assessment?.time_risk?.toFixed(1)}%</span>
        </div>
        <div className="flex justify-between text-sm">
          <span className="text-gray-600">Quality Risk:</span>
          <span className="font-bold">{risk_assessment?.quality_risk?.toFixed(1)}%</span>
        </div>
      </div>

      {emergency_triggers && emergency_triggers.recommended_actions.length > 0 && (
        <div className="mt-4 p-3 bg-yellow-50 border border-yellow-200 rounded">
          <h4 className="text-sm font-bold text-yellow-800 mb-2">⚠️ Emergency Actions Required:</h4>
          <ul className="text-xs text-yellow-700 space-y-1">
            {emergency_triggers.recommended_actions.map((action: string, idx: number) => (
              <li key={idx}>• {action}</li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}