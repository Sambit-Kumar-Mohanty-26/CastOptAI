# Constraint Configuration Files

This directory contains site-specific constraint profiles for the CastOpt AI system.

## File Structure
Each JSON file represents a complete site profile with the following structure:

```json
{
  "site_id": "unique_identifier",
  "site_name": "Human readable name",
  "location": "City, Country",
  "timezone": "Timezone identifier",
  
  "cement_bounds": {
    "min_value": 200,
    "max_value": 500,
    "unit": "kg",
    "description": "Cement content per m³"
  },
  
  "chemical_bounds": {
    "min_value": 0.5,
    "max_value": 8.0,
    "unit": "kg",
    "description": "Superplasticizer content"
  },
  
  "steam_bounds": {
    "min_value": 0,
    "max_value": 10,
    "unit": "hours",
    "description": "Steam curing duration"
  },
  
  "water_bounds": {
    "min_value": 150,
    "max_value": 200,
    "unit": "kg",
    "description": "Water content per m³"
  },
  
  "max_batch_size": 2.5,
  "min_curing_temp": 5,
  "max_curing_temp": 80,
  "max_steam_pressure": 8,
  
  "cement_storage_capacity": 50000,
  "chemical_storage_capacity": 5000,
  "current_cement_stock": 35000,
  "current_chemical_stock": 2800,
  
  "primary_supplier": "Supplier Name",
  "backup_suppliers": ["Supplier 1", "Supplier 2"],
  "min_order_quantity": 1000,
  "delivery_lead_time": 24,
  "quality_standards": ["IS 8112", "IS 12269"],
  
  "working_hours": [6, 22],
  "max_daily_production": 50,
  "safety_margins": {
    "cement": 0.05,
    "chemicals": 0.10,
    "steam": 0.15
  },
  
  "local_temperature_range": [5, 45],
  "humidity_range": [20, 90],
  "seasonal_adjustments": {
    "summer": 1.1,
    "monsoon": 0.9,
    "winter": 0.95
  }
}
```

## Available Profiles

1. **delhi_yard.json** - Delhi NCR Precast Yard
2. **mumbai_yard.json** - Mumbai Coastal Yard  
3. **chennai_yard.json** - Chennai Industrial Yard

## Usage

The constraint engine automatically loads all `.json` files in this directory at startup. New profiles can be added by creating additional JSON files following the above structure.

## Dynamic Adjustments

Constraints are automatically adjusted based on:
- Current stock levels
- Seasonal factors
- Working hours
- Equipment capabilities
- Environmental conditions