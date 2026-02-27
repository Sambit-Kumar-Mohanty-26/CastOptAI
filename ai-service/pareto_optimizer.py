"""Multi-objective Pareto Optimization for CastOptAI Phase 2"""
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
from scipy.optimize import differential_evolution
from sklearn.preprocessing import MinMaxScaler
import json
import logging

logger = logging.getLogger(__name__)

@dataclass
class ObjectiveWeights:
    """User-defined weights for multi-objective optimization"""
    cost_weight: float = 0.4
    time_weight: float = 0.3
    strength_weight: float = 0.2
    environmental_weight: float = 0.1

@dataclass
class ParetoSolution:
    """Represents a solution on the Pareto frontier"""
    recipe: Dict[str, float]
    objectives: Dict[str, float]
    constraints_satisfied: bool
    dominance_count: int = 0
    dominated_by: List[int] = None

class ParetoOptimizer:
    """Multi-objective Pareto optimization system"""
    
    def __init__(self):
        self.solutions: List[ParetoSolution] = []
        self.pareto_front: List[ParetoSolution] = []
        self.weights = ObjectiveWeights()
        
    def set_objective_weights(self, weights: ObjectiveWeights):
        """Set user-defined objective weights"""
        self.weights = weights
        logger.info(f"Updated objective weights: {weights}")
    
    def generate_candidate_solutions(self, 
                                   bounds: Dict[str, Tuple[float, float]],
                                   n_solutions: int = 100) -> List[Dict[str, float]]:
        """Generate diverse candidate solutions using Latin Hypercube sampling"""
        candidates = []
        
        for _ in range(n_solutions):
            solution = {}
            for param, (min_val, max_val) in bounds.items():
                # Add some randomization for diversity
                perturbation = np.random.normal(0, 0.1)
                value = np.random.uniform(min_val, max_val)
                value = max(min_val, min(max_val, value + perturbation * (max_val - min_val)))
                solution[param] = round(value, 2)
            candidates.append(solution)
        
        return candidates
    
    def evaluate_objectives(self, 
                          recipe: Dict[str, float],
                          target_strength: float,
                          target_time: float,
                          model_predictor) -> Dict[str, float]:
        """Evaluate all objectives for a given recipe"""
        
        # Predict strength using ML model
        features = np.array([[
            recipe['cement'],
            recipe['chemicals'], 
            recipe['steam_hours'],
            recipe['water'],
            25,  # temp placeholder
            60   # humidity placeholder
        ]])
        
        predicted_strength = model_predictor.predict(features)[0]
        
        # Calculate objectives
        objectives = {
            'cost': self._calculate_cost(recipe),
            'time_to_strength': abs(target_time - 24),  # Simplified time metric
            'strength_deviation': abs(predicted_strength - target_strength),
            'environmental_impact': self._calculate_environmental_impact(recipe)
        }
        
        return objectives
    
    def _calculate_cost(self, recipe: Dict[str, float]) -> float:
        """Calculate total cost of recipe"""
        cement_cost = recipe['cement'] * 8.0  # Rs per kg
        chemical_cost = recipe['chemicals'] * 200.0  # Rs per kg
        steam_cost = recipe['steam_hours'] * 150.0  # Rs per hour
        return cement_cost + chemical_cost + steam_cost
    
    def _calculate_environmental_impact(self, recipe: Dict[str, float]) -> float:
        """Calculate environmental impact score"""
        co2_emissions = recipe['cement'] * 0.9 + recipe['chemicals'] * 2.5
        energy_consumption = recipe['steam_hours'] * 50
        return co2_emissions + energy_consumption * 0.5
    
    def find_pareto_front(self, solutions: List[ParetoSolution]) -> List[ParetoSolution]:
        """Find Pareto optimal solutions using dominance relationship"""
        
        # Reset dominance counts
        for sol in solutions:
            sol.dominance_count = 0
            sol.dominated_by = []
        
        # Calculate dominance relationships
        for i, sol1 in enumerate(solutions):
            for j, sol2 in enumerate(solutions):
                if i != j and self._dominates(sol1, sol2):
                    sol2.dominance_count += 1
                    if sol2.dominated_by is None:
                        sol2.dominated_by = []
                    sol2.dominated_by.append(i)
        
        # Pareto front consists of non-dominated solutions
        pareto_front = [sol for sol in solutions if sol.dominance_count == 0]
        return sorted(pareto_front, key=lambda x: x.objectives['cost'])
    
    def _dominates(self, sol1: ParetoSolution, sol2: ParetoSolution) -> bool:
        """Check if solution 1 dominates solution 2"""
        obj1 = sol1.objectives
        obj2 = sol2.objectives
        
        # Solution 1 dominates if it's better in at least one objective
        # and not worse in any other objectives
        better_in_any = False
        worse_in_any = False
        
        # Weighted comparison
        weights = [self.weights.cost_weight, self.weights.time_weight, 
                  self.weights.strength_weight, self.weights.environmental_weight]
        objectives = ['cost', 'time_to_strength', 'strength_deviation', 'environmental_impact']
        
        for i, obj_name in enumerate(objectives):
            if obj1[obj_name] < obj2[obj_name]:  # Better (lower is better for all objectives)
                better_in_any = True
            elif obj1[obj_name] > obj2[obj_name]:
                worse_in_any = True
        
        return better_in_any and not worse_in_any
    
    def sensitivity_analysis(self, 
                           pareto_solutions: List[ParetoSolution],
                           weight_ranges: Dict[str, Tuple[float, float]]) -> Dict[str, Any]:
        """Perform sensitivity analysis on objective weights"""
        
        sensitivity_results = {
            'weight_impact': {},
            'solution_stability': {},
            'robust_solutions': []
        }
        
        # Test different weight combinations
        base_weights = {
            'cost_weight': [0.2, 0.4, 0.6, 0.8],
            'time_weight': [0.1, 0.3, 0.5, 0.7],
            'strength_weight': [0.1, 0.2, 0.3, 0.4],
            'environmental_weight': [0.05, 0.15, 0.25, 0.35]
        }
        
        solution_frequency = {}
        for sol in pareto_solutions:
            solution_key = str(sorted(sol.recipe.items()))
            solution_frequency[solution_key] = solution_frequency.get(solution_key, 0) + 1
        
        # Find most stable solutions
        sorted_solutions = sorted(solution_frequency.items(), 
                                key=lambda x: x[1], reverse=True)
        
        sensitivity_results['robust_solutions'] = [
            {'recipe': eval(sol_key), 'frequency': freq} 
            for sol_key, freq in sorted_solutions[:5]
        ]
        
        return sensitivity_results

def create_pareto_visualization_data(pareto_solutions: List[ParetoSolution]) -> Dict[str, Any]:
    """Prepare data for Pareto frontier visualization"""
    
    visualization_data = {
        'solutions': [],
        'frontier_points': [],
        'objectives_range': {}
    }
    
    # Extract objective values for plotting
    costs = [sol.objectives['cost'] for sol in pareto_solutions]
    strengths = [100 - sol.objectives['strength_deviation'] for sol in pareto_solutions]  # Invert for better visualization
    environmental = [100 - sol.objectives['environmental_impact'] for sol in pareto_solutions]
    
    # Prepare solution data
    for i, sol in enumerate(pareto_solutions):
        visualization_data['solutions'].append({
            'id': i,
            'recipe': sol.recipe,
            'cost': sol.objectives['cost'],
            'strength_score': 100 - sol.objectives['strength_deviation'],
            'environmental_score': 100 - sol.objectives['environmental_impact'],
            'time_score': 100 - sol.objectives['time_to_strength'],
            'constraints_satisfied': sol.constraints_satisfied
        })
    
    # Create frontier points for plotting
    df = pd.DataFrame({
        'cost': costs,
        'strength_score': strengths,
        'environmental_score': environmental
    })
    
    # Normalize for visualization
    scaler = MinMaxScaler()
    normalized = scaler.fit_transform(df[['cost', 'strength_score', 'environmental_score']])
    
    visualization_data['frontier_points'] = normalized.tolist()
    visualization_data['objectives_range'] = {
        'cost': {'min': min(costs), 'max': max(costs)},
        'strength_score': {'min': min(strengths), 'max': max(strengths)},
        'environmental_score': {'min': min(environmental), 'max': max(environmental)}
    }
    
    return visualization_data

# Example usage function
def run_pareto_optimization(target_strength: float, 
                          target_time: float,
                          bounds: Dict[str, Tuple[float, float]],
                          model_predictor,
                          weights: ObjectiveWeights = None) -> Dict[str, Any]:
    """Main function to run Pareto optimization"""
    
    optimizer = ParetoOptimizer()
    
    if weights:
        optimizer.set_objective_weights(weights)
    
    # Generate candidate solutions
    candidates = optimizer.generate_candidate_solutions(bounds, n_solutions=150)
    
    # Evaluate all candidates
    solutions = []
    for candidate in candidates:
        objectives = optimizer.evaluate_objectives(
            candidate, target_strength, target_time, model_predictor
        )
        
        # Simple constraint check
        constraints_ok = (bounds['cement'][0] <= candidate['cement'] <= bounds['cement'][1] and
                         bounds['chemicals'][0] <= candidate['chemicals'] <= bounds['chemicals'][1] and
                         bounds['steam_hours'][0] <= candidate['steam_hours'] <= bounds['steam_hours'][1])
        
        solution = ParetoSolution(
            recipe=candidate,
            objectives=objectives,
            constraints_satisfied=constraints_ok
        )
        solutions.append(solution)
    
    # Find Pareto front
    pareto_front = optimizer.find_pareto_front(solutions)
    
    # Perform sensitivity analysis
    sensitivity = optimizer.sensitivity_analysis(pareto_front, {
        'cost_weight': (0.1, 0.8),
        'time_weight': (0.1, 0.6),
        'strength_weight': (0.1, 0.4),
        'environmental_weight': (0.05, 0.3)
    })
    
    # Create visualization data
    viz_data = create_pareto_visualization_data(pareto_front)
    
    return {
        'pareto_front': pareto_front,
        'sensitivity_analysis': sensitivity,
        'visualization_data': viz_data,
        'total_solutions_evaluated': len(solutions),
        'pareto_solutions_count': len(pareto_front)
    }