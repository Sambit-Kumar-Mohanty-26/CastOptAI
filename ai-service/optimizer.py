import numpy as np
from scipy.optimize import minimize
import joblib

# Load the trained model
model = joblib.load('model.pkl')

# Cost constants (Example values for L&T use case)
COSTS = {
    "cement": 0.15,      # per kg
    "chemicals": 2.5,   # per kg (superplasticizer)
    "steam": 50.0       # per hour
}

def objective_function(x, target_time, temp, humidity):
    # x = [cement_amount, chemical_amount, steam_hours]
    # Minimize total cost
    return (x[0] * COSTS["cement"]) + (x[1] * COSTS["chemicals"]) + (x[2] * COSTS["steam"])

def strength_constraint(x, target_strength, target_time, temp, humidity):
    # Predict strength for this specific combination
    # Fixed values for coarse/fine aggregate and slag/fly_ash for simplicity
    prediction_input = [[
        x[0], 0, 0, 180, x[1], 1000, 700, 
        target_time, temp, humidity, 1 if x[2] > 0 else 0
    ]]
    predicted_strength = model.predict(prediction_input)[0]
    return predicted_strength - target_strength

def get_optimal_recipe(target_strength, target_time, temp, humidity):
    # Initial guess: [300kg cement, 2kg chemicals, 0 hrs steam]
    x0 = [300, 2, 0]
    
    # Constraints and Bounds
    cons = {'type': 'ineq', 'fun': strength_constraint, 'args': (target_strength, target_time, temp, humidity)}
    bounds = [(200, 550), (0, 15), (0, 12)] # Min/Max limits

    res = minimize(objective_function, x0, args=(target_time, temp, humidity), 
                   method='SLSQP', bounds=bounds, constraints=cons)
    
    if res.success:
        return {
            "cement": round(res.x[0], 2),
            "chemicals": round(res.x[1], 2),
            "steam_hours": round(res.x[2], 1)
        }
    return None