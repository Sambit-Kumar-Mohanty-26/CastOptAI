"""
Dynamic Constraint Engine for CastOpt AI
Manages site-specific constraints based on local conditions, equipment, and supply chain
"""

import json
import os
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
from datetime import datetime, timezone, timedelta
import numpy as np
try:
    from zoneinfo import ZoneInfo
except ImportError:
    from backports.zoneinfo import ZoneInfo


@dataclass
class ConstraintBounds:
    """Defines bounds for optimization variables"""
    min_value: float
    max_value: float
    unit: str
    description: str


@dataclass
class SiteProfile:
    """Complete site configuration with all constraints"""
    site_id: str
    site_name: str
    location: str
    timezone: str
    
    # Material constraints
    cement_bounds: ConstraintBounds
    chemical_bounds: ConstraintBounds
    steam_bounds: ConstraintBounds
    water_bounds: ConstraintBounds
    
    # Equipment constraints
    max_batch_size: float  # m³
    min_curing_temp: float  # °C
    max_curing_temp: float  # °C
    max_steam_pressure: float  # bar
    
    # Storage constraints
    cement_storage_capacity: float  # kg
    chemical_storage_capacity: float  # kg
    current_cement_stock: float  # kg
    current_chemical_stock: float  # kg
    
    # Supplier constraints
    primary_supplier: str
    backup_suppliers: List[str]
    min_order_quantity: float  # kg
    delivery_lead_time: int  # hours
    quality_standards: List[str]
    
    # Operational constraints
    working_hours: Tuple[int, int]  # (start_hour, end_hour)
    max_daily_production: float  # m³
    safety_margins: Dict[str, float]  # percentage buffers
    
    # Environmental constraints
    local_temperature_range: Tuple[float, float]  # (min, max) °C
    humidity_range: Tuple[float, float]  # (min, max) %
    seasonal_adjustments: Dict[str, float]  # month-based multipliers


class DynamicConstraintEngine:
    """Main constraint management system"""
    
    def __init__(self, config_dir: str = "constraints_config"):
        self.config_dir = config_dir
        self.site_profiles: Dict[str, SiteProfile] = {}
        self.current_site: Optional[str] = None
        self._load_site_profiles()
    
    def _load_site_profiles(self):
        """Load all site configuration files"""
        if not os.path.exists(self.config_dir):
            os.makedirs(self.config_dir)
            self._create_default_profiles()
        
        for filename in os.listdir(self.config_dir):
            if filename.endswith('.json'):
                site_id = filename.replace('.json', '')
                try:
                    with open(os.path.join(self.config_dir, filename), 'r') as f:
                        data = json.load(f)
                        # Convert dict bounds to ConstraintBounds objects
                        data['cement_bounds'] = ConstraintBounds(**data['cement_bounds'])
                        data['chemical_bounds'] = ConstraintBounds(**data['chemical_bounds'])
                        data['steam_bounds'] = ConstraintBounds(**data['steam_bounds'])
                        data['water_bounds'] = ConstraintBounds(**data['water_bounds'])
                        profile = SiteProfile(**data)
                        self.site_profiles[site_id] = profile
                except Exception as e:
                    print(f"Error loading profile {filename}: {e}")
    
    def _create_default_profiles(self):
        """Create default site profiles for common scenarios"""
        profiles = {
            "delhi_yard": {
                "site_id": "delhi_yard",
                "site_name": "Delhi NCR Precast Yard",
                "location": "Delhi, India",
                "timezone": "Asia/Kolkata",
                "cement_bounds": {"min_value": 250, "max_value": 500, "unit": "kg", "description": "Cement content per m³"},
                "chemical_bounds": {"min_value": 0.5, "max_value": 8.0, "unit": "kg", "description": "Superplasticizer content"},
                "steam_bounds": {"min_value": 0, "max_value": 10, "unit": "hours", "description": "Steam curing duration"},
                "water_bounds": {"min_value": 150, "max_value": 200, "unit": "kg", "description": "Water content per m³"},
                "max_batch_size": 2.5,
                "min_curing_temp": 5,
                "max_curing_temp": 80,
                "max_steam_pressure": 8,
                "cement_storage_capacity": 50000,
                "chemical_storage_capacity": 5000,
                "current_cement_stock": 35000,
                "current_chemical_stock": 2800,
                "primary_supplier": "ACC Limited",
                "backup_suppliers": ["UltraTech", "Ambuja"],
                "min_order_quantity": 1000,
                "delivery_lead_time": 24,
                "quality_standards": ["IS 8112", "IS 12269"],
                "working_hours": [6, 22],
                "max_daily_production": 50,
                "safety_margins": {"cement": 0.05, "chemicals": 0.10, "steam": 0.15},
                "local_temperature_range": [5, 45],
                "humidity_range": [20, 90],
                "seasonal_adjustments": {"summer": 1.1, "monsoon": 0.9, "winter": 0.95}
            },
            "mumbai_yard": {
                "site_id": "mumbai_yard",
                "site_name": "Mumbai Coastal Yard",
                "location": "Mumbai, India",
                "timezone": "Asia/Kolkata",
                "cement_bounds": {"min_value": 280, "max_value": 550, "unit": "kg", "description": "Higher cement for marine conditions"},
                "chemical_bounds": {"min_value": 1.0, "max_value": 12.0, "unit": "kg", "description": "Enhanced chemical resistance"},
                "steam_bounds": {"min_value": 2, "max_value": 12, "unit": "hours", "description": "Extended curing for durability"},
                "water_bounds": {"min_value": 140, "max_value": 180, "unit": "kg", "description": "Lower water for strength"},
                "max_batch_size": 3.0,
                "min_curing_temp": 15,
                "max_curing_temp": 70,
                "max_steam_pressure": 10,
                "cement_storage_capacity": 75000,
                "chemical_storage_capacity": 8000,
                "current_cement_stock": 45000,
                "current_chemical_stock": 5200,
                "primary_supplier": "UltraTech",
                "backup_suppliers": ["ACC", "Shree Cement"],
                "min_order_quantity": 2000,
                "delivery_lead_time": 18,
                "quality_standards": ["IS 8112", "IS 456", "BS EN 206"],
                "working_hours": [5, 23],
                "max_daily_production": 75,
                "safety_margins": {"cement": 0.10, "chemicals": 0.15, "steam": 0.20},
                "local_temperature_range": [20, 35],
                "humidity_range": [60, 95],
                "seasonal_adjustments": {"summer": 1.2, "monsoon": 1.3, "winter": 1.0}
            },
            "chennai_yard": {
                "site_id": "chennai_yard",
                "site_name": "Chennai Industrial Yard",
                "location": "Chennai, India",
                "timezone": "Asia/Kolkata",
                "cement_bounds": {"min_value": 220, "max_value": 480, "unit": "kg", "description": "Optimized for industrial applications"},
                "chemical_bounds": {"min_value": 0.3, "max_value": 6.0, "unit": "kg", "description": "Standard chemical content"},
                "steam_bounds": {"min_value": 0, "max_value": 8, "unit": "hours", "description": "Moderate steam curing"},
                "water_bounds": {"min_value": 160, "max_value": 210, "unit": "kg", "description": "Balanced water content"},
                "max_batch_size": 4.0,
                "min_curing_temp": 10,
                "max_curing_temp": 75,
                "max_steam_pressure": 6,
                "cement_storage_capacity": 100000,
                "chemical_storage_capacity": 12000,
                "current_cement_stock": 65000,
                "current_chemical_stock": 8500,
                "primary_supplier": "Ramco Cements",
                "backup_suppliers": ["India Cements", "Birla White"],
                "min_order_quantity": 3000,
                "delivery_lead_time": 12,
                "quality_standards": ["IS 8112", "IS 1489"],
                "working_hours": [6, 20],
                "max_daily_production": 100,
                "safety_margins": {"cement": 0.08, "chemicals": 0.12, "steam": 0.10},
                "local_temperature_range": [22, 40],
                "humidity_range": [50, 85],
                "seasonal_adjustments": {"summer": 1.05, "monsoon": 0.95, "winter": 0.9}
            }
        }
        
        # Save default profiles
        for site_id, profile_data in profiles.items():
            filepath = os.path.join(self.config_dir, f"{site_id}.json")
            with open(filepath, 'w') as f:
                json.dump(profile_data, f, indent=2)
    
    def set_current_site(self, site_id: str) -> bool:
        """Set the active site for constraint application"""
        if site_id in self.site_profiles:
            self.current_site = site_id
            return True
        return False
    
    def get_current_constraints(self) -> Optional[SiteProfile]:
        """Get constraints for current site"""
        if self.current_site and self.current_site in self.site_profiles:
            return self.site_profiles[self.current_site]
        return None
    
    def get_dynamic_bounds(self, current_time: Optional[datetime] = None) -> Dict[str, Tuple[float, float]]:
        """Get dynamically adjusted bounds based on current conditions"""
        profile = self.get_current_constraints()
        if not profile:
            # Return default bounds if no profile
            return {
                'cement': (200, 550),
                'chemicals': (0, 15),
                'steam': (0, 12),
                'water': (150, 220)
            }
        
        # Apply seasonal adjustments
        seasonal_multiplier = 1.0
        if current_time:
            month = current_time.month
            if month in [3, 4, 5]:  # Summer
                seasonal_multiplier = profile.seasonal_adjustments.get('summer', 1.0)
            elif month in [6, 7, 8, 9]:  # Monsoon
                seasonal_multiplier = profile.seasonal_adjustments.get('monsoon', 1.0)
            else:  # Winter
                seasonal_multiplier = profile.seasonal_adjustments.get('winter', 1.0)
        
        # Handle case where bounds might be dictionaries (from JSON loading)
        cement_min = profile.cement_bounds.min_value if hasattr(profile.cement_bounds, 'min_value') else profile.cement_bounds['min_value']
        cement_max = profile.cement_bounds.max_value if hasattr(profile.cement_bounds, 'max_value') else profile.cement_bounds['max_value']
        chemical_min = profile.chemical_bounds.min_value if hasattr(profile.chemical_bounds, 'min_value') else profile.chemical_bounds['min_value']
        chemical_max = profile.chemical_bounds.max_value if hasattr(profile.chemical_bounds, 'max_value') else profile.chemical_bounds['max_value']
        steam_min = profile.steam_bounds.min_value if hasattr(profile.steam_bounds, 'min_value') else profile.steam_bounds['min_value']
        steam_max = profile.steam_bounds.max_value if hasattr(profile.steam_bounds, 'max_value') else profile.steam_bounds['max_value']
        water_min = profile.water_bounds.min_value if hasattr(profile.water_bounds, 'min_value') else profile.water_bounds['min_value']
        water_max = profile.water_bounds.max_value if hasattr(profile.water_bounds, 'max_value') else profile.water_bounds['max_value']
        
        # Apply stock availability constraints
        cement_max = min(
            cement_max,
            profile.current_cement_stock * 0.8  # 80% of stock as safety margin
        )
        
        chemical_max = min(
            chemical_max,
            profile.current_chemical_stock * 0.7  # 70% of stock as safety margin
        )
        
        # Apply seasonal adjustments
        cement_max = cement_max * seasonal_multiplier
        chemical_max = chemical_max * seasonal_multiplier
        
        return {
            'cement': (cement_min, cement_max),
            'chemicals': (chemical_min, chemical_max),
            'steam': (steam_min, steam_max),
            'water': (water_min, water_max)
        }
    
    def validate_proposed_recipe(self, cement: float, chemicals: float, steam: float, 
                               water: float) -> Tuple[bool, List[str]]:
        """Validate if a recipe meets all site constraints"""
        profile = self.get_current_constraints()
        if not profile:
            return True, []
        
        violations = []
        bounds = self.get_dynamic_bounds()
        
        if not (bounds['cement'][0] <= cement <= bounds['cement'][1]):
            violations.append(f"Cement {cement} kg out of bounds {bounds['cement']}")
        
        if not (bounds['chemicals'][0] <= chemicals <= bounds['chemicals'][1]):
            violations.append(f"Chemicals {chemicals} kg out of bounds {bounds['chemicals']}")
        
        if not (bounds['steam'][0] <= steam <= bounds['steam'][1]):
            violations.append(f"Steam {steam} hours out of bounds {bounds['steam']}")
        
        if not (bounds['water'][0] <= water <= bounds['water'][1]):
            violations.append(f"Water {water} kg out of bounds {bounds['water']}")
        
        if steam > 0 and steam > profile.max_steam_pressure:
            violations.append(f"Steam pressure {steam} exceeds maximum {profile.max_steam_pressure}")
        
        if cement > profile.current_cement_stock:
            violations.append(f"Insufficient cement stock: {profile.current_cement_stock} kg available")
        
        if chemicals > profile.current_chemical_stock:
            violations.append(f"Insufficient chemical stock: {profile.current_chemical_stock} kg available")
        
        try:
            site_tz = ZoneInfo(profile.timezone)
        except Exception:
            site_tz = timezone(timedelta(hours=5, minutes=30))
        current_time = datetime.now(site_tz)
        current_hour = current_time.hour
        if not (profile.working_hours[0] <= current_hour <= profile.working_hours[1]):
            violations.append(f"Outside working hours: {profile.working_hours[0]}-{profile.working_hours[1]}")
        
        return len(violations) == 0, violations
    
    def get_available_sites(self) -> List[Dict[str, str]]:
        """Get list of all available site profiles"""
        return [
            {
                'site_id': site_id,
                'site_name': profile.site_name,
                'location': profile.location
            }
            for site_id, profile in self.site_profiles.items()
        ]
    
    def update_stock_levels(self, site_id: str, cement_stock: float, chemical_stock: float) -> bool:
        """Update stock levels for a site"""
        if site_id not in self.site_profiles:
            return False
        
        profile = self.site_profiles[site_id]
        profile.current_cement_stock = cement_stock
        profile.current_chemical_stock = chemical_stock
        
        # Save updated profile
        filepath = os.path.join(self.config_dir, f"{site_id}.json")
        with open(filepath, 'w') as f:
            json.dump(asdict(profile), f, indent=2)
        
        return True


# Global constraint engine instance
constraint_engine = DynamicConstraintEngine()