import numpy as np
import pandas as pd
from scipy.optimize import minimize
import joblib
from constraints import constraint_engine


model = joblib.load('model.pkl')


COSTS = {
    "cement_per_kg": 300 / 50,       # ₹300 per 50kg bag = ₹6/kg
    "chemicals_per_kg": 150,          # ₹150 per kg of admixture
    "steam_per_hour": 500,            # ₹500 per hour of steam curing
}


CO2_PER_KG_CEMENT = 0.9              # kg CO₂ per kg cement
CO2_PER_HOUR_STEAM = 40              # kg CO₂ per hour of steam (50 kWh → ~40 kg CO₂)
ENERGY_PER_HOUR_STEAM = 50           # kWh per hour of steam


DEFAULT_SLAG = 0
DEFAULT_FLY_ASH = 0
DEFAULT_WATER = 180
DEFAULT_COARSE_AGG = 1000
DEFAULT_FINE_AGG = 700


FEATURE_COLUMNS = [
    'cement', 'slag', 'fly_ash', 'water',
    'superplasticizer', 'coarse_agg', 'fine_agg',
    'age_hours', 'temperature', 'humidity', 'curing_method'
]


def build_feature_vector(cement, chemicals, steam_hours, target_time, temp, humidity):
    """Build the 11-feature input DataFrame expected by the trained model."""
    curing_method = 1 if steam_hours > 0.1 else 0
    return pd.DataFrame([[
        cement, DEFAULT_SLAG, DEFAULT_FLY_ASH, DEFAULT_WATER,
        chemicals, DEFAULT_COARSE_AGG, DEFAULT_FINE_AGG,
        target_time, temp, humidity, curing_method
    ]], columns=FEATURE_COLUMNS)


def predict_strength(cement, chemicals, steam_hours, time_hours, temp, humidity):
    """Predict compressive strength for a given recipe and conditions."""
    features = build_feature_vector(cement, chemicals, steam_hours, time_hours, temp, humidity)
    return float(model.predict(features)[0])


def calculate_cost(cement, chemicals, steam_hours):
    """Calculate total cost in ₹ for a recipe."""
    return (
        cement * COSTS["cement_per_kg"] +
        chemicals * COSTS["chemicals_per_kg"] +
        steam_hours * COSTS["steam_per_hour"]
    )


def calculate_co2(cement, steam_hours):
    """Calculate total CO₂ emissions in kg for a recipe."""
    return cement * CO2_PER_KG_CEMENT + steam_hours * CO2_PER_HOUR_STEAM


def calculate_energy(steam_hours):
    """Calculate total energy usage in kWh."""
    return steam_hours * ENERGY_PER_HOUR_STEAM


def calculate_risk(predicted_strength, target_strength):
    """
    Calculate risk level and confidence based on buffer between predicted and target.
    Uses a logarithmic scale for more realistic confidence scores.
    Buffer < 5% → High Risk, 5-15% → Medium Risk, >15% → Low Risk
    """
    if target_strength <= 0:
        return "Low", 92.0
    
    buffer_pct = ((predicted_strength - target_strength) / target_strength) * 100
    

    import math
    if buffer_pct <= 0:
        confidence = max(40, 50 + buffer_pct)
    elif buffer_pct < 5:
        confidence = 55 + buffer_pct * 2  # 55-65 range
    elif buffer_pct < 15:
        confidence = 65 + buffer_pct * 1.2  # 71-83 range
    elif buffer_pct < 30:
        confidence = 78 + math.log(buffer_pct) * 3  # ~86-88 range
    elif buffer_pct < 60:
        confidence = 85 + math.log(buffer_pct) * 1.5  # ~90-91 range
    else:
        confidence = 88 + math.log(buffer_pct) * 1.2  # ~93-94 range
    

    if predicted_strength > 50:
        confidence -= (predicted_strength - 50) * 0.1
    
    confidence = round(min(96, max(45, confidence)), 1)
    
    if buffer_pct < 5:
        return "High", confidence
    elif buffer_pct < 15:
        return "Medium", confidence
    else:
        return "Low", confidence


def get_traditional_baseline(target_strength, target_time, temp, humidity):
    """Simulate the 'traditional' approach: max steam + max chemicals."""
    baseline_cement = 400
    baseline_chemicals = 10
    baseline_steam = 8
    pred = predict_strength(baseline_cement, baseline_chemicals, baseline_steam, target_time, temp, humidity)
    cost = calculate_cost(baseline_cement, baseline_chemicals, baseline_steam)
    co2 = calculate_co2(baseline_cement, baseline_steam)
    energy = calculate_energy(baseline_steam)
    return {
        "cement": baseline_cement,
        "chemicals": baseline_chemicals,
        "steam_hours": baseline_steam,
        "predicted_strength": round(pred, 2),
        "cost": round(cost, 2),
        "co2": round(co2, 2),
        "energy": round(energy, 2),
    }



def objective_cheapest(x, target_time, temp, humidity):
    """Minimize total cost."""
    return calculate_cost(x[0], x[1], x[2])


def objective_fastest(x, target_time, temp, humidity):
    """Minimize time to reach strength (by maximizing strength gain rate).
       We proxy this by minimizing negative predicted strength (maximize strength)."""
    pred = predict_strength(x[0], x[1], x[2], target_time, temp, humidity)
    return -pred  # maximize strength → fastest to reach target


def objective_eco(x, target_time, temp, humidity):
    """Minimize CO₂ emissions."""
    return calculate_co2(x[0], x[2])


def strength_constraint(x, target_strength, target_time, temp, humidity):
    """Constraint: predicted strength must be >= target."""
    pred = predict_strength(x[0], x[1], x[2], target_time, temp, humidity)
    return pred - target_strength


def run_optimization(objective_fn, target_strength, target_time, temp, humidity, site_id=None):
    """Run SciPy optimization with dynamic bounds based on site constraints."""
    # Set current site if provided
    if site_id:
        constraint_engine.set_current_site(site_id)
    
    # Get dynamic bounds
    dynamic_bounds = constraint_engine.get_dynamic_bounds()
    cement_bounds = dynamic_bounds['cement']
    chemical_bounds = dynamic_bounds['chemicals']
    steam_bounds = dynamic_bounds['steam']
    water_bounds = dynamic_bounds['water']
    
    x0 = [
        (cement_bounds[0] + cement_bounds[1]) / 2,  # Midpoint for cement
        (chemical_bounds[0] + chemical_bounds[1]) / 2,  # Midpoint for chemicals
        (steam_bounds[0] + steam_bounds[1]) / 2  # Midpoint for steam
    ]
    
    cons = {
        'type': 'ineq',
        'fun': strength_constraint,
        'args': (target_strength, target_time, temp, humidity)
    }
    
    bounds = [
        cement_bounds,
        chemical_bounds,
        steam_bounds
    ]

    res = minimize(
        objective_fn, x0,
        args=(target_time, temp, humidity),
        method='SLSQP', bounds=bounds, constraints=cons,
        options={'maxiter': 200, 'ftol': 1e-8}
    )

    if res.success:
        cement = round(res.x[0], 1)
        chemicals = round(res.x[1], 2)
        steam = round(res.x[2], 1)
        
        # Validate final recipe against all constraints
        is_valid, violations = constraint_engine.validate_proposed_recipe(
            cement, chemicals, steam, water_bounds[1]
        )
        
        if not is_valid:
            print(f"Constraint violations: {violations}")
            return None
            
        return cement, chemicals, steam
    return None


def build_strategy(name, label, objective_fn, target_strength, target_time, temp, humidity, baseline, site_id=None):
    """Run optimization and build a complete strategy response dict."""
    result = run_optimization(objective_fn, target_strength, target_time, temp, humidity, site_id)

    if result is None:
        return None

    cement, chemicals, steam = result
    pred = predict_strength(cement, chemicals, steam, target_time, temp, humidity)
    cost = calculate_cost(cement, chemicals, steam)
    co2 = calculate_co2(cement, steam)
    energy = calculate_energy(steam)
    risk_level, confidence = calculate_risk(pred, target_strength)

    cost_savings = ((baseline["cost"] - cost) / baseline["cost"]) * 100 if baseline["cost"] > 0 else 0
    co2_savings = ((baseline["co2"] - co2) / baseline["co2"]) * 100 if baseline["co2"] > 0 else 0
    energy_savings = ((baseline["energy"] - energy) / baseline["energy"]) * 100 if baseline["energy"] > 0 else 0

    return {
        "name": name,
        "label": label,
        "recommended_recipe": {
            "cement": cement,
            "chemicals": chemicals,
            "steam_hours": steam,
            "water": DEFAULT_WATER,
        },
        "predicted_strength": round(pred, 2),
        "confidence_score": round(confidence, 1),
        "risk_level": risk_level,
        "cost": round(cost, 2),
        "co2_kg": round(co2, 2),
        "energy_kwh": round(energy, 2),
        "cost_savings_percent": round(max(0, cost_savings), 1),
        "carbon_reduction_percent": round(max(0, co2_savings), 1),
        "energy_savings_percent": round(max(0, energy_savings), 1),
    }


def get_all_strategies(target_strength, target_time, temp, humidity, site_id=None):
    # If site_id is provided but doesn't exist, use default constraints
    if site_id:
        constraint_engine.set_current_site(site_id)
        # Check if the site exists, if not create a default one
        if not constraint_engine.get_current_constraints():
            # Create a default site profile if the specified site doesn't exist
            from constraints import SiteProfile, ConstraintBounds
            import os
            
            default_profile = SiteProfile(
                site_id=site_id,
                site_name=f"{site_id.replace('_', ' ').title()} Yard",
                location="Unknown",
                timezone="UTC",
                cement_bounds=ConstraintBounds(200, 550, "kg", "Cement content per m³"),
                chemical_bounds=ConstraintBounds(0, 15, "kg", "Chemical content"),
                steam_bounds=ConstraintBounds(0, 12, "hours", "Steam curing duration"),
                water_bounds=ConstraintBounds(150, 220, "kg", "Water content per m³"),
                max_batch_size=3.0,
                min_curing_temp=5,
                max_curing_temp=80,
                max_steam_pressure=8,
                cement_storage_capacity=50000,
                chemical_storage_capacity=5000,
                current_cement_stock=35000,
                current_chemical_stock=2800,
                primary_supplier="Default Supplier",
                backup_suppliers=["Backup Supplier"],
                min_order_quantity=1000,
                delivery_lead_time=24,
                quality_standards=["IS 8112"],
                working_hours=(6, 22),
                max_daily_production=50,
                safety_margins={"cement": 0.05, "chemicals": 0.10, "steam": 0.15},
                local_temperature_range=(5, 45),
                humidity_range=(20, 90),
                seasonal_adjustments={"summer": 1.1, "monsoon": 0.9, "winter": 0.95}
            )
            constraint_engine.site_profiles[site_id] = default_profile
            constraint_engine.set_current_site(site_id)
    """
    Run optimization for all 3 strategies and return results + baseline.
    """
    baseline = get_traditional_baseline(target_strength, target_time, temp, humidity)

    strategies = []
    for name, label, obj_fn in [
        ("cheapest", "💰 Cheapest", objective_cheapest),
        ("fastest", "⚡ Fastest", objective_fastest),
        ("eco", "🌱 Most Eco-Friendly", objective_eco),
    ]:
        s = build_strategy(name, label, obj_fn, target_strength, target_time, temp, humidity, baseline, site_id)
        if s:
            strategies.append(s)

    return strategies, baseline


def get_strength_curve(cement, chemicals, steam_hours, temp, humidity, max_hours=24):
    """Predict strength at each hour from 1 to max_hours for a given recipe."""
    curve = []
    for hour in range(1, max_hours + 1):
        pred = predict_strength(cement, chemicals, steam_hours, hour, temp, humidity)
        curve.append({"hour": hour, "strength": round(pred, 2)})
    return curve


def predict_whatif(cement, chemicals, steam_hours, time_hours, temp, humidity):
    """Predict for a what-if scenario and return full metrics."""
    pred = predict_strength(cement, chemicals, steam_hours, time_hours, temp, humidity)
    cost = calculate_cost(cement, chemicals, steam_hours)
    co2 = calculate_co2(cement, steam_hours)
    energy = calculate_energy(steam_hours)
    return {
        "predicted_strength": round(pred, 2),
        "cost": round(cost, 2),
        "co2_kg": round(co2, 2),
        "energy_kwh": round(energy, 2),
    }