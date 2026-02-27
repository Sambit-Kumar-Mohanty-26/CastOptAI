"""Risk Management System for CastOptAI Phase 2"""
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
from scipy import stats
from sklearn.ensemble import RandomForestRegressor
import json
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

@dataclass
class RiskAssessment:
    """Comprehensive risk assessment for concrete mix"""
    strength_risk: float  # 0-1 probability of not meeting target strength
    cost_risk: float      # 0-1 probability of cost overruns
    time_risk: float      # 0-1 probability of delays
    quality_risk: float   # 0-1 probability of quality issues
    overall_risk: float   # Combined risk score
    confidence_level: float  # Confidence in risk assessment (0-1)

@dataclass
class MonteCarloResult:
    """Results from Monte Carlo simulation"""
    strength_distribution: List[float]
    probability_of_success: float
    confidence_interval: Tuple[float, float]
    risk_percentile: float

class RiskManagementSystem:
    """Comprehensive risk management for concrete optimization"""
    
    def __init__(self):
        self.historical_data = []
        self.supplier_reliability = {}
        self.defect_models = {}
        
    def monte_carlo_strength_simulation(self,
                                      recipe: Dict[str, float],
                                      target_strength: float,
                                      n_simulations: int = 1000) -> MonteCarloResult:
        """Perform Monte Carlo simulation for strength prediction"""
        
        # Simulate input parameter variations
        cement_samples = np.random.normal(
            recipe['cement'], 
            recipe['cement'] * 0.02,  # 2% coefficient of variation
            n_simulations
        )
        
        chemical_samples = np.random.normal(
            recipe['chemicals'],
            recipe['chemicals'] * 0.05,  # 5% coefficient of variation
            n_simulations
        )
        
        steam_samples = np.random.normal(
            recipe['steam_hours'],
            recipe['steam_hours'] * 0.1,  # 10% coefficient of variation
            n_simulations
        )
        
        # Temperature and humidity variations
        temp_samples = np.random.normal(25, 3, n_simulations)  # ±3°C
        humidity_samples = np.random.normal(60, 10, n_simulations)  # ±10% RH
        
        # Calculate strength for each simulation
        strength_predictions = []
        for i in range(n_simulations):
            # Simplified strength model (in practice, use trained ML model)
            strength = (
                0.05 * cement_samples[i] +
                2.5 * chemical_samples[i] +
                0.8 * steam_samples[i] +
                0.1 * temp_samples[i] -
                0.05 * humidity_samples[i] +
                np.random.normal(0, 2)  # Model uncertainty
            )
            strength_predictions.append(max(0, strength))
        
        # Calculate statistics
        strength_array = np.array(strength_predictions)
        probability_of_success = np.mean(strength_array >= target_strength)
        confidence_interval = np.percentile(strength_array, [2.5, 97.5])
        risk_percentile = np.percentile(strength_array, 10)  # 10th percentile as risk measure
        
        return MonteCarloResult(
            strength_distribution=[float(x) for x in strength_predictions],  # Convert numpy array to list
            probability_of_success=float(probability_of_success),
            confidence_interval=(float(confidence_interval[0]), float(confidence_interval[1])),
            risk_percentile=float(risk_percentile)
        )
    
    def defect_probability_modeling(self,
                                  recipe: Dict[str, float],
                                  environmental_conditions: Dict[str, float]) -> Dict[str, float]:
        """Model probability of different types of defects"""
        
        defects = {
            'strength_deficiency': 0.0,
            'workability_issues': 0.0,
            'surface_defects': 0.0,
            'curing_problems': 0.0,
            'material_contamination': 0.0
        }
        
        # Strength deficiency risk
        cement_content = recipe['cement']
        if cement_content < 300:  # Low cement content risk
            defects['strength_deficiency'] = 0.3
        elif cement_content > 500:  # High cement content risk
            defects['strength_deficiency'] = 0.15
        else:
            defects['strength_deficiency'] = 0.05
        
        # Workability issues risk
        water_cement_ratio = recipe['water'] / cement_content
        if water_cement_ratio > 0.55 or water_cement_ratio < 0.35:
            defects['workability_issues'] = 0.4
        else:
            defects['workability_issues'] = 0.1
        
        # Surface defects risk (humidity related)
        humidity = environmental_conditions.get('humidity', 60)
        if humidity > 80 or humidity < 30:
            defects['surface_defects'] = 0.25
        else:
            defects['surface_defects'] = 0.08
        
        # Curing problems risk (temperature related)
        temperature = environmental_conditions.get('temperature', 25)
        if temperature < 5 or temperature > 40:
            defects['curing_problems'] = 0.35
        else:
            defects['curing_problems'] = 0.12
        
        # Material contamination risk
        chemical_content = recipe['chemicals']
        if chemical_content > 6:  # High chemical content risk
            defects['material_contamination'] = 0.2
        else:
            defects['material_contamination'] = 0.05
        
        return defects
    
    def supplier_reliability_scoring(self, 
                                   supplier_data: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate reliability scores for suppliers"""
        
        reliability_scores = {}
        
        for supplier in supplier_data:
            supplier_id = supplier['id']
            
            # Calculate reliability based on multiple factors
            on_time_delivery = supplier.get('on_time_delivery_rate', 0.95)
            quality_score = supplier.get('quality_rating', 4.5) / 5.0
            price_stability = 1.0 - supplier.get('price_volatility', 0.1)
            years_experience = min(supplier.get('years_operating', 10), 20) / 20.0
            
            # Weighted reliability score
            reliability = (
                0.4 * on_time_delivery +
                0.3 * quality_score +
                0.2 * price_stability +
                0.1 * years_experience
            )
            
            reliability_scores[supplier_id] = round(reliability, 3)
        
        return reliability_scores
    
    def alternative_material_recommendations(self,
                                           current_recipe: Dict[str, float],
                                           constraints: Dict[str, Tuple[float, float]],
                                           risk_threshold: float = 0.3) -> List[Dict[str, Any]]:
        """Recommend alternative materials when risk is high"""
        
        alternatives = []
        
        # Check if current recipe has high risk
        current_risk = self.assess_recipe_risk(current_recipe, {})
        
        if current_risk.overall_risk > risk_threshold:
            # Suggest cement alternatives
            if current_recipe['cement'] > constraints['cement'][1] * 0.8:
                alternatives.append({
                    'type': 'cement_reduction',
                    'description': 'Reduce cement content with supplementary cementitious materials',
                    'modification': {'cement': current_recipe['cement'] * 0.9},
                    'risk_reduction': 0.15,
                    'cost_impact': 'decrease'
                })
            
            # Suggest chemical alternatives
            if current_recipe['chemicals'] < constraints['chemicals'][0] * 1.2:
                alternatives.append({
                    'type': 'chemical_increase',
                    'description': 'Increase superplasticizer for better workability',
                    'modification': {'chemicals': current_recipe['chemicals'] * 1.3},
                    'risk_reduction': 0.12,
                    'cost_impact': 'increase'
                })
            
            # Suggest steam curing alternatives
            if current_recipe['steam_hours'] > constraints['steam_hours'][1] * 0.7:
                alternatives.append({
                    'type': 'steam_reduction',
                    'description': 'Optimize steam curing duration',
                    'modification': {'steam_hours': current_recipe['steam_hours'] * 0.85},
                    'risk_reduction': 0.08,
                    'cost_impact': 'decrease'
                })
        
        return alternatives
    
    def emergency_protocol_triggers(self,
                                  current_conditions: Dict[str, Any],
                                  recipe: Dict[str, float]) -> Dict[str, Any]:
        """Check for emergency conditions requiring immediate action"""
        
        triggers = {
            'material_shortage': False,
            'quality_critical': False,
            'schedule_risk': False,
            'cost_overrun': False,
            'recommended_actions': []
        }
        
        # Material shortage check
        stock_levels = current_conditions.get('stock_levels', {})
        for material, level in stock_levels.items():
            if level < 0.15:  # Less than 15% stock
                triggers['material_shortage'] = True
                triggers['recommended_actions'].append(
                    f"Urgent: Low {material} stock - reorder immediately"
                )
        
        # Quality critical check
        predicted_strength = current_conditions.get('predicted_strength', 0)
        target_strength = current_conditions.get('target_strength', 25)
        if predicted_strength < target_strength * 0.85:  # 15% below target
            triggers['quality_critical'] = True
            triggers['recommended_actions'].append(
                "Critical: Strength prediction significantly below target - revise mix design"
            )
        
        # Schedule risk check
        remaining_time = current_conditions.get('remaining_time', 48)
        if remaining_time < 12:  # Less than 12 hours
            triggers['schedule_risk'] = True
            triggers['recommended_actions'].append(
                "Warning: Insufficient time for proper curing - expedite process"
            )
        
        # Cost overrun check
        current_cost = current_conditions.get('current_cost', 5000)
        budget_limit = current_conditions.get('budget_limit', 4500)
        if current_cost > budget_limit * 1.1:  # 10% over budget
            triggers['cost_overrun'] = True
            triggers['recommended_actions'].append(
                "Alert: Project exceeding budget - review cost optimization"
            )
        
        return triggers
    
    def assess_recipe_risk(self, 
                          recipe: Dict[str, float],
                          environmental_conditions: Dict[str, float]) -> RiskAssessment:
        """Comprehensive risk assessment for a recipe"""
        
        # Perform Monte Carlo simulation
        mc_result = self.monte_carlo_strength_simulation(recipe, 25)  # Assuming 25 MPa target
        
        # Get defect probabilities
        defect_probs = self.defect_probability_modeling(recipe, environmental_conditions)
        
        # Calculate individual risks
        strength_risk = 1.0 - mc_result.probability_of_success
        cost_risk = self._calculate_cost_risk(recipe)
        time_risk = self._calculate_time_risk(recipe, environmental_conditions)
        quality_risk = max(defect_probs.values())
        
        # Calculate overall risk (weighted average)
        overall_risk = (
            0.35 * strength_risk +
            0.25 * cost_risk +
            0.20 * time_risk +
            0.20 * quality_risk
        )
        
        # Confidence level based on simulation quality
        confidence_level = min(0.95, 0.7 + 0.25 * (1 - strength_risk))
        
        return RiskAssessment(
            strength_risk=round(strength_risk, 3),
            cost_risk=round(cost_risk, 3),
            time_risk=round(time_risk, 3),
            quality_risk=round(quality_risk, 3),
            overall_risk=round(overall_risk, 3),
            confidence_level=round(confidence_level, 3)
        )
    
    def _calculate_cost_risk(self, recipe: Dict[str, float]) -> float:
        """Calculate cost overrun risk"""
        # Simplified model - in practice, use historical price data
        cement_cost_volatility = 0.15
        chemical_cost_volatility = 0.25
        steam_cost_volatility = 0.10
        
        weighted_volatility = (
            0.5 * cement_cost_volatility +
            0.3 * chemical_cost_volatility +
            0.2 * steam_cost_volatility
        )
        
        return min(1.0, weighted_volatility)
    
    def _calculate_time_risk(self, 
                           recipe: Dict[str, float],
                           environmental_conditions: Dict[str, float]) -> float:
        """Calculate schedule delay risk"""
        # Temperature effect on curing time
        temperature = environmental_conditions.get('temperature', 25)
        if temperature < 10 or temperature > 35:
            return 0.4
        elif temperature < 15 or temperature > 30:
            return 0.25
        else:
            return 0.1

# Example usage
def run_risk_assessment(recipe: Dict[str, float],
                       environmental_conditions: Dict[str, float],
                       target_strength: float) -> Dict[str, Any]:
    """Run complete risk assessment"""
    
    risk_system = RiskManagementSystem()
    
    # Monte Carlo simulation
    mc_result = risk_system.monte_carlo_strength_simulation(recipe, target_strength)
    
    # Defect probability modeling
    defect_probs = risk_system.defect_probability_modeling(recipe, environmental_conditions)
    
    # Risk assessment
    risk_assessment = risk_system.assess_recipe_risk(recipe, environmental_conditions)
    
    # Alternative recommendations
    alternatives = risk_system.alternative_material_recommendations(
        recipe, 
        {
            'cement': (200, 550),
            'chemicals': (0, 15),
            'steam_hours': (0, 12)
        }
    )
    
    # Emergency triggers
    emergency_triggers = risk_system.emergency_protocol_triggers(
        {
            'predicted_strength': mc_result.risk_percentile,
            'target_strength': target_strength,
            'remaining_time': 24,
            'current_cost': 4500,
            'budget_limit': 5000,
            'stock_levels': {'cement': 0.2, 'chemicals': 0.8}
        },
        recipe
    )
    
    return {
        'monte_carlo_results': {
            'probability_of_success': float(mc_result.probability_of_success),
            'confidence_interval': [float(mc_result.confidence_interval[0]), float(mc_result.confidence_interval[1])],
            'risk_percentile': float(mc_result.risk_percentile)
        },
        'defect_probabilities': {k: float(v) for k, v in defect_probs.items()},
        'risk_assessment': {k: float(v) if isinstance(v, (int, float, np.number)) else v for k, v in risk_assessment.__dict__.items()},
        'alternative_recommendations': alternatives,
        'emergency_triggers': emergency_triggers
    }