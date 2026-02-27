"""
Real-time Data Integration for CastOpt AI
Handles IoT sensor data, weather forecasts, material pricing, and production schedules
"""

import json
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import asyncio
import aiohttp
from dataclasses import dataclass, asdict


@dataclass
class SensorReading:
    """IoT sensor data reading"""
    sensor_id: str
    timestamp: datetime
    temperature: float
    humidity: float
    location: str
    chamber_id: Optional[str] = None
    pressure: Optional[float] = None
    vibration: Optional[float] = None


@dataclass
class WeatherForecast:
    """7-day weather forecast data"""
    location: str
    forecast_date: datetime
    temperature_min: float
    temperature_max: float
    humidity_avg: float
    precipitation_probability: float
    wind_speed: float
    condition: str


@dataclass
class MaterialPrice:
    """Real-time material pricing data"""
    material: str
    price_per_unit: float
    unit: str
    timestamp: datetime
    supplier: str
    location: str
    trend: str  # 'up', 'down', 'stable'


@dataclass
class ProductionSchedule:
    """Production scheduling data"""
    job_id: str
    product_type: str
    target_strength: float
    target_time: float
    scheduled_start: datetime
    scheduled_end: datetime
    priority: str
    status: str  # 'scheduled', 'in_progress', 'completed', 'delayed'
    location: str


class RealTimeDataIntegration:
    """Main real-time data integration system"""
    
    def __init__(self, openweather_api_key: Optional[str] = None):
        self.openweather_api_key = openweather_api_key
        self.sensor_data: List[SensorReading] = []
        self.weather_forecasts: List[WeatherForecast] = []
        self.material_prices: List[MaterialPrice] = []
        self.production_schedules: List[ProductionSchedule] = []
        self.last_update = datetime.now()
    
    def add_sensor_reading(self, reading: SensorReading):
        """Add a new sensor reading"""
        self.sensor_data.append(reading)
        # Keep only last 1000 readings for memory management
        if len(self.sensor_data) > 1000:
            self.sensor_data = self.sensor_data[-1000:]
        self.last_update = datetime.now()
    
    def get_current_conditions(self, location: str) -> Dict[str, float]:
        """Get current environmental conditions from sensors"""
        # Filter recent readings (last 1 hour)
        one_hour_ago = datetime.now() - timedelta(hours=1)
        recent_readings = [
            r for r in self.sensor_data 
            if r.location == location and r.timestamp >= one_hour_ago
        ]
        
        if not recent_readings:
            return {'temperature': 25.0, 'humidity': 60.0}  # Default values
        
        # Calculate averages
        temps = [r.temperature for r in recent_readings]
        humids = [r.humidity for r in recent_readings]
        
        return {
            'temperature': np.mean(temps),
            'humidity': np.mean(humids),
            'temperature_std': np.std(temps),
            'humidity_std': np.std(humids),
            'reading_count': len(recent_readings)
        }
    
    async def fetch_weather_forecast(self, city: str, country: str = "IN") -> List[WeatherForecast]:
        """Fetch 7-day weather forecast from OpenWeatherMap"""
        if not self.openweather_api_key:
            # Return mock data if no API key
            return self._generate_mock_forecast(city)
        
        try:
            url = f"http://api.openweathermap.org/data/2.5/forecast"
            params = {
                'q': f"{city},{country}",
                'appid': self.openweather_api_key,
                'units': 'metric'
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        forecasts = self._parse_weather_forecast(data, city)
                        self.weather_forecasts.extend(forecasts)
                        # Remove old forecasts
                        self._cleanup_old_data()
                        return forecasts
                    else:
                        print(f"Error fetching weather: {response.status}")
                        return self._generate_mock_forecast(city)
        except Exception as e:
            print(f"Error fetching weather forecast: {e}")
            return self._generate_mock_forecast(city)
    
    def _parse_weather_forecast(self, data: Dict, city: str) -> List[WeatherForecast]:
        """Parse OpenWeatherMap forecast data"""
        forecasts = []
        for item in data.get('list', [])[:8]:  # Next 24-48 hours (3-hour intervals)
            dt = datetime.fromtimestamp(item['dt'])
            main = item['main']
            weather = item['weather'][0] if item['weather'] else {}
            
            forecast = WeatherForecast(
                location=city,
                forecast_date=dt,
                temperature_min=main.get('temp_min', main.get('temp', 25)),
                temperature_max=main.get('temp_max', main.get('temp', 25)),
                humidity_avg=main.get('humidity', 60),
                precipitation_probability=item.get('pop', 0) * 100,
                wind_speed=item.get('wind', {}).get('speed', 0),
                condition=weather.get('description', 'clear sky')
            )
            forecasts.append(forecast)
        return forecasts
    
    def _generate_mock_forecast(self, city: str) -> List[WeatherForecast]:
        """Generate mock weather forecast data"""
        forecasts = []
        base_temp = 25.0
        base_humidity = 60.0
        
        # Add some seasonal variation
        month = datetime.now().month
        if month in [3, 4, 5]:  # Summer
            base_temp = 35.0
            base_humidity = 45.0
        elif month in [6, 7, 8, 9]:  # Monsoon
            base_temp = 28.0
            base_humidity = 80.0
        else:  # Winter
            base_temp = 20.0
            base_humidity = 55.0
        
        for i in range(7):
            forecast_date = datetime.now() + timedelta(days=i)
            # Add daily variation
            temp_variation = np.random.normal(0, 3)
            humidity_variation = np.random.normal(0, 10)
            
            forecast = WeatherForecast(
                location=city,
                forecast_date=forecast_date,
                temperature_min=base_temp + temp_variation - 2,
                temperature_max=base_temp + temp_variation + 2,
                humidity_avg=max(20, min(95, base_humidity + humidity_variation)),
                precipitation_probability=max(0, min(100, np.random.normal(20, 15))),
                wind_speed=max(0, np.random.normal(10, 5)),
                condition=np.random.choice(['clear sky', 'partly cloudy', 'scattered clouds', 'light rain'])
            )
            forecasts.append(forecast)
        return forecasts
    
    def get_weather_forecast(self, location: str, days_ahead: int = 3) -> List[WeatherForecast]:
        """Get weather forecast for a specific location"""
        # Filter forecasts for location and date range
        target_date = datetime.now() + timedelta(days=days_ahead)
        relevant_forecasts = [
            f for f in self.weather_forecasts
            if f.location == location and f.forecast_date.date() <= target_date.date()
        ]
        return relevant_forecasts[-days_ahead:] if relevant_forecasts else []
    
    def update_material_prices(self, prices: List[MaterialPrice]):
        """Update material pricing data"""
        self.material_prices.extend(prices)
        # Keep only recent prices (last 30 days)
        thirty_days_ago = datetime.now() - timedelta(days=30)
        self.material_prices = [
            p for p in self.material_prices 
            if p.timestamp >= thirty_days_ago
        ]
        self.last_update = datetime.now()
    
    def get_current_prices(self) -> Dict[str, MaterialPrice]:
        """Get current material prices"""
        current_prices = {}
        recent_prices = {}
        
        # Group by material and get most recent price
        for price in self.material_prices:
            material = price.material
            if material not in recent_prices or price.timestamp > recent_prices[material].timestamp:
                recent_prices[material] = price
        
        return recent_prices
    
    def add_production_schedule(self, schedule: ProductionSchedule):
        """Add a new production schedule"""
        self.production_schedules.append(schedule)
        # Remove old completed schedules
        self._cleanup_old_schedules()
        self.last_update = datetime.now()
    
    def get_active_schedules(self, location: Optional[str] = None) -> List[ProductionSchedule]:
        """Get currently active production schedules"""
        now = datetime.now()
        active_schedules = [
            s for s in self.production_schedules
            if s.status in ['scheduled', 'in_progress'] and s.scheduled_end >= now
        ]
        
        if location:
            active_schedules = [s for s in active_schedules if s.location == location]
            
        return sorted(active_schedules, key=lambda x: x.scheduled_start)
    
    def _cleanup_old_data(self):
        """Remove old data to manage memory"""
        # Remove old sensor data (older than 24 hours)
        one_day_ago = datetime.now() - timedelta(days=1)
        self.sensor_data = [r for r in self.sensor_data if r.timestamp >= one_day_ago]
        
        # Remove old forecasts (older than 1 day)
        self.weather_forecasts = [f for f in self.weather_forecasts if f.forecast_date >= one_day_ago]
    
    def _cleanup_old_schedules(self):
        """Remove old completed schedules"""
        thirty_days_ago = datetime.now() - timedelta(days=30)
        self.production_schedules = [
            s for s in self.production_schedules 
            if s.scheduled_end >= thirty_days_ago or s.status != 'completed'
        ]
    
    def get_system_status(self) -> Dict:
        """Get status of all real-time data sources"""
        return {
            'last_update': self.last_update.isoformat(),
            'sensor_readings_count': len(self.sensor_data),
            'weather_forecasts_count': len(self.weather_forecasts),
            'material_prices_count': len(self.material_prices),
            'active_schedules_count': len([s for s in self.production_schedules if s.status in ['scheduled', 'in_progress']]),
            'data_sources': {
                'sensors': 'active' if self.sensor_data else 'no_data',
                'weather': 'active' if self.openweather_api_key else 'mock_data',
                'pricing': 'active' if self.material_prices else 'no_data',
                'scheduling': 'active' if self.production_schedules else 'no_data'
            }
        }


class IoTDataSimulator:
    """Simulates IoT sensor data for testing"""
    
    def __init__(self):
        self.sensors = {
            'delhi_chamber_1': {'base_temp': 22, 'base_humidity': 55},
            'delhi_chamber_2': {'base_temp': 24, 'base_humidity': 58},
            'mumbai_chamber_1': {'base_temp': 28, 'base_humidity': 75},
            'chennai_chamber_1': {'base_temp': 26, 'base_humidity': 65}
        }
    
    def generate_readings(self, location: str, count: int = 10) -> List[SensorReading]:
        """Generate simulated sensor readings"""
        if location not in self.sensors:
            location = list(self.sensors.keys())[0]  # Default to first sensor
        
        sensor_config = self.sensors[location]
        readings = []
        
        for i in range(count):
            # Add some realistic variation
            temp_variation = np.random.normal(0, 1.5)  # ±1.5°C variation
            humidity_variation = np.random.normal(0, 5)  # ±5% humidity variation
            
            reading = SensorReading(
                sensor_id=f"{location}_sensor_{np.random.randint(1000, 9999)}",
                timestamp=datetime.now() - timedelta(minutes=i*15),  # Every 15 minutes
                temperature=max(10, min(50, sensor_config['base_temp'] + temp_variation)),
                humidity=max(20, min(95, sensor_config['base_humidity'] + humidity_variation)),
                location=location,
                chamber_id=f"chamber_{np.random.randint(1, 5)}"
            )
            readings.append(reading)
        
        return readings


# Global instances
real_time_data = RealTimeDataIntegration()
iot_simulator = IoTDataSimulator()

# Initialize with some mock data
def initialize_real_time_data():
    """Initialize real-time data system with sample data"""
    # Add some initial sensor readings
    for location in ['delhi_yard', 'mumbai_yard', 'chennai_yard']:
        readings = iot_simulator.generate_readings(location, 20)
        for reading in readings:
            real_time_data.add_sensor_reading(reading)
    
    # Add some mock material prices
    mock_prices = [
        MaterialPrice('cement', 320, 'per_50kg_bag', datetime.now(), 'ACC Limited', 'India', 'stable'),
        MaterialPrice('superplasticizer', 160, 'per_kg', datetime.now(), 'Sika', 'India', 'up'),
        MaterialPrice('water', 5, 'per_1000l', datetime.now(), 'Municipal', 'Local', 'stable')
    ]
    real_time_data.update_material_prices(mock_prices)
    
    # Add some mock production schedules
    mock_schedules = [
        ProductionSchedule(
            'CST-2024-0847', 'Precast Wall Panel', 25.0, 12.0,
            datetime.now() + timedelta(hours=2),
            datetime.now() + timedelta(hours=14),
            'high', 'scheduled', 'delhi_yard'
        ),
        ProductionSchedule(
            'CST-2024-0848', 'Hollow Core Slab', 30.0, 16.0,
            datetime.now() + timedelta(hours=6),
            datetime.now() + timedelta(hours=22),
            'medium', 'in_progress', 'mumbai_yard'
        )
    ]
    for schedule in mock_schedules:
        real_time_data.add_production_schedule(schedule)

# Initialize on import
initialize_real_time_data()