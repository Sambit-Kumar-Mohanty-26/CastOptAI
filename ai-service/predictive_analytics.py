"""Predictive Analytics Module for CastOptAI Phase 3"""
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, Any
from datetime import datetime, timedelta
from dataclasses import dataclass
import logging
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib
import json

logger = logging.getLogger(__name__)

@dataclass
class PredictionResult:
    """Prediction result with confidence intervals"""
    predicted_value: float
    confidence_lower: float
    confidence_upper: float
    model_confidence: float  # 0-1 scale
    prediction_date: datetime

@dataclass
class DemandForecast:
    """Demand forecasting result"""
    forecasted_demand: float
    confidence_interval: Tuple[float, float]
    trend_direction: str  # 'increasing', 'decreasing', 'stable'
    seasonality_factor: float

class PredictiveAnalyticsEngine:
    """Advanced predictive analytics for concrete optimization"""
    
    def __init__(self):
        self.models = {}
        self.training_data = pd.DataFrame()
        self.is_trained = False
        
    def load_historical_data(self, data: pd.DataFrame):
        """Load historical optimization and usage data"""
        self.training_data = data
        logger.info(f"Loaded {len(data)} historical records for predictive analytics")
        
    def train_demand_forecasting_model(self, target_column: str = 'demand_volume'):
        """Train model to forecast future demand"""
        if len(self.training_data) < 10:
            logger.warning("Insufficient training data for demand forecasting, using sample model")
            # Create a simple model when there's no historical data
            self.models['demand_forecast'] = 'sample_model'
            self.is_trained = True
            return True
            
        # Prepare features for demand forecasting
        features = self._prepare_demand_features(self.training_data)
        
        if target_column not in self.training_data.columns:
            logger.error(f"Target column '{target_column}' not found in training data")
            return False
            
        X = features
        y = self.training_data[target_column]
        
        if X.empty or len(X) < 10:
            logger.warning("Insufficient features for training, using sample model")
            self.models['demand_forecast'] = 'sample_model'
            self.is_trained = True
            return True
        
        # Train random forest model
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X, y)
        
        self.models['demand_forecast'] = model
        self.is_trained = True
        
        logger.info("Demand forecasting model trained successfully")
        return True
    
    def _prepare_demand_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare features for demand forecasting"""
        features = pd.DataFrame()
        
        # Time-based features
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            features['month'] = df['date'].dt.month
            features['day_of_week'] = df['date'].dt.dayofweek
            features['quarter'] = df['date'].dt.quarter
            features['is_weekend'] = df['date'].dt.weekday >= 5
            
        # Lagged features if available
        if 'demand_volume' in df.columns:
            features['demand_lag_1'] = df['demand_volume'].shift(1)
            features['demand_lag_7'] = df['demand_volume'].shift(7)
            features['demand_ma_7'] = df['demand_volume'].rolling(window=7).mean()
            features['demand_ma_30'] = df['demand_volume'].rolling(window=30).mean()
        
        # Weather features if available
        weather_cols = ['temperature', 'humidity', 'rainfall']
        for col in weather_cols:
            if col in df.columns:
                features[f'{col}_avg_7'] = df[col].rolling(window=7).mean()
                features[f'{col}_current'] = df[col]
        
        # Fill NaN values
        features = features.fillna(0)
        
        return features
    
    def forecast_demand(self, periods: int = 30, confidence_level: float = 0.95) -> List[DemandForecast]:
        """Forecast demand for the next 'periods' days"""
        if not self.is_trained or 'demand_forecast' not in self.models:
            logger.error("Demand forecasting model not trained")
            return []
        
        forecasts = []
        model = self.models['demand_forecast']
        
        # Handle sample model case
        if model == 'sample_model':
            # Generate sample forecasts
            base_demand = 500.0
            for i in range(periods):
                future_date = datetime.now() + timedelta(days=i+1)
                
                # Add some randomness
                demand_variation = np.random.normal(0, 50)  # ±50 variance
                prediction = max(100, base_demand + demand_variation)  # Min 100
                
                # Calculate confidence interval
                std_error = 0.1 * prediction  # 10% of prediction as std error
                z_score = 1.96  # 95% confidence
                
                lower_bound = max(0, prediction - z_score * std_error)
                upper_bound = prediction + z_score * std_error
                
                # Determine trend
                trend = 'stable'
                
                forecast = DemandForecast(
                    forecasted_demand=prediction,
                    confidence_interval=(lower_bound, upper_bound),
                    trend_direction=trend,
                    seasonality_factor=self._calculate_seasonality_factor(future_date)
                )
                
                forecasts.append(forecast)
            return forecasts
        
        # Generate future dates for trained model
        last_date = datetime.now()
        for i in range(periods):
            future_date = last_date + timedelta(days=i+1)
            
            # Create features for prediction
            features = self._create_future_features(future_date)
            prediction = model.predict([features])[0]
            
            # Calculate confidence interval (simplified)
            # In practice, use quantile regression or bootstrap
            std_error = 0.1 * prediction  # 10% of prediction as std error
            z_score = 1.96  # 95% confidence
            
            lower_bound = max(0, prediction - z_score * std_error)
            upper_bound = prediction + z_score * std_error
            
            # Determine trend (simplified)
            trend = 'stable'
            if hasattr(self.training_data, 'columns') and 'demand_volume' in self.training_data.columns:
                if prediction > self.training_data['demand_volume'].mean() * 1.1:
                    trend = 'increasing'
                elif prediction < self.training_data['demand_volume'].mean() * 0.9:
                    trend = 'decreasing'
            
            forecast = DemandForecast(
                forecasted_demand=prediction,
                confidence_interval=(lower_bound, upper_bound),
                trend_direction=trend,
                seasonality_factor=self._calculate_seasonality_factor(future_date)
            )
            
            forecasts.append(forecast)
        
        return forecasts
    
    def _create_future_features(self, date: datetime) -> np.array:
        """Create features for a future date"""
        # This is a simplified version - in practice, you'd need weather forecasts, etc.
        # Return a fixed-size array to match training features
        features = np.array([
            date.month,
            date.weekday(),
            (date.month - 1) // 3 + 1,  # quarter
            int(date.weekday() >= 5),   # is_weekend
            self.training_data['demand_volume'].iloc[-1] if len(self.training_data) > 0 and 'demand_volume' in self.training_data.columns else 100,
            self.training_data['demand_volume'].iloc[-7:].mean() if len(self.training_data) > 7 and 'demand_volume' in self.training_data.columns else 100,
            self.training_data['demand_volume'].iloc[-30:].mean() if len(self.training_data) > 30 and 'demand_volume' in self.training_data.columns else 100,
            0,  # Placeholder for rainfall_avg_7
            0,  # Placeholder for rainfall_current
            0,  # Placeholder for temperature_avg_7
            0,  # Placeholder for temperature_current
            0,  # Placeholder for humidity_avg_7
        ])
        return features
    
    def _calculate_seasonality_factor(self, date: datetime) -> float:
        """Calculate seasonality factor based on historical patterns"""
        # Simplified seasonal factor based on month
        month = date.month
        seasonal_factors = {
            1: 0.8, 2: 0.7, 3: 0.9, 4: 1.1, 5: 1.2, 6: 1.1,
            7: 0.9, 8: 0.8, 9: 1.0, 10: 1.1, 11: 1.0, 12: 0.9
        }
        return seasonal_factors.get(month, 1.0)
    
    def predict_material_prices(self, material_type: str, days_ahead: int = 30) -> List[PredictionResult]:
        """Predict future material prices"""
        predictions = []
        
        # Simulate price prediction (in practice, use commodity price models)
        current_price = self._get_current_material_price(material_type)
        
        for day in range(1, days_ahead + 1):
            # Simulate price movement with some volatility
            volatility = 0.02  # 2% daily volatility
            drift = 0.0001 * day  # Small upward drift
            
            predicted_price = current_price * (1 + np.random.normal(drift, volatility))
            
            # Confidence interval (simplified)
            confidence_lower = predicted_price * 0.95
            confidence_upper = predicted_price * 1.05
            model_confidence = 0.85  # Fixed for demo
            
            pred_result = PredictionResult(
                predicted_value=predicted_price,
                confidence_lower=confidence_lower,
                confidence_upper=confidence_upper,
                model_confidence=model_confidence,
                prediction_date=datetime.now() + timedelta(days=day)
            )
            
            predictions.append(pred_result)
        
        return predictions
    
    def _get_current_material_price(self, material_type: str) -> float:
        """Get current material price (simulated)"""
        base_prices = {
            'cement': 8.0,
            'chemicals': 200.0,
            'aggregates': 1.5,
            'steel': 60.0
        }
        return base_prices.get(material_type, 10.0)
    
    def predict_quality_degradation(self, recipe: Dict[str, float], days: int = 365) -> Dict[str, List[float]]:
        """Predict how concrete properties degrade over time"""
        time_points = list(range(0, days + 1, 30))  # Monthly predictions
        strength_degradation = []
        durability_degradation = []
        
        base_strength = self._estimate_initial_strength(recipe)
        
        for day in time_points:
            # Strength degradation model (simplified)
            degradation_factor = 1.0 - (0.02 * day / 365)  # 2% annual degradation
            current_strength = base_strength * degradation_factor
            
            # Durability model (simplified)
            durability = 100 - (0.05 * day / 365)  # 5% annual durability loss
            
            strength_degradation.append(current_strength)
            durability_degradation.append(durability)
        
        return {
            'time_points': time_points,
            'strength_degradation': strength_degradation,
            'durability_degradation': durability_degradation
        }
    
    def _estimate_initial_strength(self, recipe: Dict[str, float]) -> float:
        """Estimate initial strength based on recipe (simplified model)"""
        return (
            0.05 * recipe.get('cement', 300) +
            2.5 * recipe.get('chemicals', 2.5) +
            0.8 * recipe.get('steam_hours', 4) +
            np.random.normal(10, 2)  # Add some noise
        )

def create_sample_historical_data() -> pd.DataFrame:
    """Create sample historical data for demonstration"""
    dates = pd.date_range(start='2023-01-01', end='2024-01-01', freq='D')
    n = len(dates)
    
    data = {
        'date': dates,
        'demand_volume': np.random.normal(500, 100, n).clip(200, 800),
        'cement_price': np.random.normal(8.0, 0.5, n).clip(6.0, 10.0),
        'chemical_price': np.random.normal(200.0, 20.0, n).clip(150.0, 250.0),
        'temperature': np.random.normal(25, 5, n).clip(10, 40),
        'humidity': np.random.normal(60, 15, n).clip(20, 90),
        'production_volume': np.random.normal(450, 80, n).clip(250, 700)
    }
    
    return pd.DataFrame(data)

# Example usage
def run_predictive_analytics_demo():
    """Demo function to showcase predictive analytics"""
    
    # Create sample data
    historical_data = create_sample_historical_data()
    
    # Initialize engine
    engine = PredictiveAnalyticsEngine()
    engine.load_historical_data(historical_data)
    
    # Train demand forecasting model
    engine.train_demand_forecasting_model('demand_volume')
    
    # Forecast demand
    demand_forecasts = engine.forecast_demand(periods=30)
    
    # Predict material prices
    cement_prices = engine.predict_material_prices('cement', 14)
    
    # Predict quality degradation
    recipe_example = {
        'cement': 350,
        'chemicals': 3.0,
        'steam_hours': 6,
        'water': 175
    }
    quality_prediction = engine.predict_quality_degradation(recipe_example)
    
    return {
        'demand_forecasts': demand_forecasts[:5],  # First 5 days
        'cement_price_predictions': cement_prices[:7],  # First week
        'quality_degradation': quality_prediction
    }