from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
from datetime import datetime
from dataclasses import asdict
import pandas as pd
import joblib
import os
from optimizer import (
    get_all_strategies, get_strength_curve, predict_whatif,
    predict_strength, calculate_cost, calculate_co2, calculate_energy, model
)
from train_model import train_model
from realtime_data import real_time_data, SensorReading
from constraints import constraint_engine
from phase2_integration import setup_phase2_routes

app = FastAPI(title="CastOpt AI", version="2.0")


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class OptimizeRequest(BaseModel):
    target_strength: float
    target_time: float
    temp: float
    humidity: float
    site_id: Optional[str] = None
    use_real_time_data: Optional[bool] = False


class WhatIfRequest(BaseModel):
    cement: float
    chemicals: float
    steam_hours: float
    time_hours: float
    temp: float
    humidity: float


class RetrainRequest(BaseModel):
    cement: float
    chemicals: float
    steam_hours: float
    water: float
    age_hours: float
    temperature: float
    humidity: float
    curing_method: int
    actual_strength: float

@app.get("/")
async def root():
    return {"service": "CastOpt AI", "version": "2.0", "status": "running"}

@app.post("/optimize")
async def optimize_cycle(data: OptimizeRequest):
    """
    Main optimization endpoint.
    Returns 3 strategies (cheapest, fastest, eco) + baseline + strength curves.
    """
    try:
        # Use real-time data if requested
        temp = data.temp
        humidity = data.humidity
        
        if data.use_real_time_data and data.site_id:
            # Get real-time conditions
            conditions = real_time_data.get_current_conditions(data.site_id)
            if conditions['reading_count'] > 0:
                temp = conditions['temperature']
                humidity = conditions['humidity']
                print(f"Using real-time data: {temp:.1f}°C, {humidity:.1f}% RH")
        
        strategies, baseline = get_all_strategies(
            data.target_strength, data.target_time, temp, humidity, data.site_id
        )

        if not strategies:
            return {
                "status": "error",
                "message": "Could not find any optimal recipe. Target may be too aggressive for the given conditions."
            }


        for s in strategies:
            r = s["recommended_recipe"]
            s["strength_curve"] = get_strength_curve(
                r["cement"], r["chemicals"], r["steam_hours"],
                data.temp, data.humidity, max_hours=24
            )


        baseline["strength_curve"] = get_strength_curve(
            baseline["cement"], baseline["chemicals"], baseline["steam_hours"],
            data.temp, data.humidity, max_hours=24
        )

        # Add real-time data context
        context_info = {
            "used_real_time_data": data.use_real_time_data and data.site_id,
            "real_time_conditions": real_time_data.get_current_conditions(data.site_id) if data.site_id else None,
            "system_status": real_time_data.get_system_status(),
            "site_constraints": constraint_engine.get_current_constraints().__dict__ if data.site_id and constraint_engine.get_current_constraints() else None
        }
        
        return {
            "status": "success",
            "target_strength": data.target_strength,
            "target_time": data.target_time,
            "strategies": strategies,
            "baseline": baseline,
            "context": context_info
        }

    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.post("/what-if")
async def what_if_simulation(data: WhatIfRequest):
    """
    What-If endpoint: user manually sets recipe + conditions,
    gets back predicted strength, cost, CO₂.
    """
    try:
        result = predict_whatif(
            data.cement, data.chemicals, data.steam_hours,
            data.time_hours, data.temp, data.humidity
        )

        curve = get_strength_curve(
            data.cement, data.chemicals, data.steam_hours,
            data.temp, data.humidity, max_hours=24
        )
        result["strength_curve"] = curve
        return {"status": "success", **result}

    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.post("/retrain")
async def retrain_model_endpoint(data: RetrainRequest):
    """
    Continuous Learning Loop: receive actual field data and retrain the model.
    """
    try:
        curing_method = data.curing_method
        new_row = {
            'cement': data.cement,
            'slag': 0,
            'fly_ash': 0,
            'water': data.water,
            'superplasticizer': data.chemicals,
            'coarse_agg': 1000,
            'fine_agg': 700,
            'age_hours': data.age_hours,
            'temperature': data.temperature,
            'humidity': data.humidity,
            'curing_method': curing_method,
            'strength': data.actual_strength
        }

        csv_path = 'processed_concrete_data.csv'
        df = pd.read_csv(csv_path)
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        df.to_csv(csv_path, index=False)


        train_model()


        import optimizer
        optimizer.model = joblib.load('model.pkl')

        return {
            "status": "success",
            "message": "Model retrained with new field data. CastOpt AI is now smarter!",
            "total_samples": len(df)
        }

    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.get("/realtime/status")
async def get_realtime_status():
    """Get status of all real-time data sources."""
    return real_time_data.get_system_status()


@app.get("/realtime/conditions/{location}")
async def get_current_conditions(location: str):
    """Get current environmental conditions for a location."""
    conditions = real_time_data.get_current_conditions(location)
    return {
        "location": location,
        "conditions": conditions,
        "timestamp": datetime.now().isoformat()
    }


@app.get("/realtime/forecast/{location}")
async def get_weather_forecast(location: str, days: int = 3):
    """Get weather forecast for a location."""
    forecasts = await real_time_data.fetch_weather_forecast(location)
    return {
        "location": location,
        "forecasts": [asdict(f) for f in forecasts[:days*3]],  # 3 readings per day
        "days_requested": days
    }


@app.get("/realtime/prices")
async def get_material_prices():
    """Get current material prices."""
    prices = real_time_data.get_current_prices()
    return {
        "prices": {k: asdict(v) for k, v in prices.items()},
        "timestamp": datetime.now().isoformat()
    }


@app.get("/realtime/schedules/{location}")
async def get_production_schedules(location: str):
    """Get active production schedules for a location."""
    schedules = real_time_data.get_active_schedules(location)
    return {
        "location": location,
        "schedules": [asdict(s) for s in schedules],
        "count": len(schedules)
    }


@app.get("/constraints/sites")
async def get_available_sites():
    """Get list of available site profiles."""
    sites = constraint_engine.get_available_sites()
    return {
        "sites": sites,
        "count": len(sites)
    }


@app.get("/constraints/site/{site_id}")
async def get_site_constraints(site_id: str):
    """Get constraints for a specific site."""
    constraint_engine.set_current_site(site_id)
    constraints = constraint_engine.get_current_constraints()
    if constraints:
        return {
            "site_id": site_id,
            "constraints": asdict(constraints),
            "dynamic_bounds": constraint_engine.get_dynamic_bounds()
        }
    else:
        return {"status": "error", "message": f"Site {site_id} not found"}

@app.on_event("startup")
async def startup_event():
    """Initialize Phase 2 business logic components"""
    setup_phase2_routes(app)
    print("✅ Phase 2 business logic components initialized")
