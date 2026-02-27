"""Supply Chain Intelligence System for CastOptAI Phase 2"""
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
import json
import logging
from collections import defaultdict

logger = logging.getLogger(__name__)

@dataclass
class SupplierInfo:
    """Supplier information and capabilities"""
    id: str
    name: str
    material_type: str
    base_price: float
    lead_time: int  # days
    reliability_score: float  # 0-1
    quality_rating: float     # 0-5
    capacity: float          # monthly capacity
    current_utilization: float  # 0-1
    location: str
    certifications: List[str]

@dataclass
class MaterialRequirement:
    """Material requirements for optimization"""
    material_type: str
    quantity: float
    delivery_date: datetime
    priority: str  # high, medium, low

@dataclass
class SupplyChainRecommendation:
    """Supply chain optimization recommendation"""
    suppliers: List[Dict[str, Any]]
    total_cost: float
    lead_time: int
    risk_score: float
    carbon_footprint: float
    alternative_options: List[Dict[str, Any]]

class SupplyChainIntelligence:
    """Intelligent supply chain optimization system"""
    
    def __init__(self):
        self.suppliers: List[SupplierInfo] = []
        self.historical_prices = defaultdict(list)
        self.delivery_routes = {}
        self.inventory_levels = {}
    
    def add_suppliers(self, supplier_data: List[Dict[str, Any]]):
        """Add supplier information to the system"""
        for supplier_info in supplier_data:
            supplier = SupplierInfo(**supplier_info)
            self.suppliers.append(supplier)
        logger.info(f"Added {len(supplier_data)} suppliers to supply chain system")
    
    def optimize_material_sourcing(self,
                                 requirements: List[MaterialRequirement],
                                 budget_constraints: Dict[str, float] = None) -> SupplyChainRecommendation:
        """Optimize material sourcing considering cost, lead time, and reliability"""
        
        recommendations = []
        total_cost = 0
        max_lead_time = 0
        total_risk = 0
        total_carbon = 0
        
        for requirement in requirements:
            # Find best suppliers for this requirement
            suitable_suppliers = self._find_suitable_suppliers(
                requirement.material_type,
                requirement.quantity,
                requirement.delivery_date
            )
            
            # Score suppliers based on multiple criteria
            scored_suppliers = self._score_suppliers(
                suitable_suppliers, 
                requirement, 
                budget_constraints
            )
            
            # Select best option
            if scored_suppliers:
                best_supplier = scored_suppliers[0]
                recommendations.append({
                    'material': requirement.material_type,
                    'supplier': best_supplier['supplier'].name,
                    'supplier_id': best_supplier['supplier'].id,
                    'quantity': requirement.quantity,
                    'unit_price': best_supplier['unit_price'],
                    'total_cost': best_supplier['total_cost'],
                    'lead_time': best_supplier['lead_time'],
                    'delivery_date': best_supplier['delivery_date'],
                    'reliability': best_supplier['supplier'].reliability_score,
                    'risk_score': best_supplier['risk_score']
                })
                
                total_cost += best_supplier['total_cost']
                max_lead_time = max(max_lead_time, best_supplier['lead_time'])
                total_risk += best_supplier['risk_score'] * best_supplier['total_cost']
                total_carbon += best_supplier['carbon_footprint']
        
        # Calculate overall risk score
        overall_risk = total_risk / max(total_cost, 1) if total_cost > 0 else 0
        
        # Generate alternative options
        alternatives = self._generate_alternatives(requirements, recommendations)
        
        return SupplyChainRecommendation(
            suppliers=recommendations,
            total_cost=total_cost,
            lead_time=max_lead_time,
            risk_score=overall_risk,
            carbon_footprint=total_carbon,
            alternative_options=alternatives
        )
    
    def _find_suitable_suppliers(self, 
                               material_type: str,
                               quantity: float,
                               delivery_date: datetime) -> List[SupplierInfo]:
        """Find suppliers that can meet the requirements"""
        
        suitable = []
        for supplier in self.suppliers:
            if (supplier.material_type == material_type and
                supplier.capacity * (1 - supplier.current_utilization) >= quantity and
                supplier.lead_time <= (delivery_date - datetime.now()).days):
                suitable.append(supplier)
        
        return suitable
    
    def _score_suppliers(self,
                        suppliers: List[SupplierInfo],
                        requirement: MaterialRequirement,
                        budget_constraints: Dict[str, float] = None) -> List[Dict[str, Any]]:
        """Score suppliers based on multiple criteria"""
        
        scored = []
        
        for supplier in suppliers:
            # Calculate dynamic pricing
            unit_price = self._calculate_dynamic_price(supplier, requirement.quantity)
            
            # Calculate risk score
            risk_score = self._calculate_supplier_risk(supplier, requirement.delivery_date)
            
            # Calculate carbon footprint
            carbon_footprint = self._calculate_carbon_footprint(supplier, requirement.quantity)
            
            # Calculate delivery date
            delivery_date = datetime.now() + timedelta(days=supplier.lead_time)
            
            scored.append({
                'supplier': supplier,
                'unit_price': unit_price,
                'total_cost': unit_price * requirement.quantity,
                'lead_time': supplier.lead_time,
                'delivery_date': delivery_date,
                'risk_score': risk_score,
                'carbon_footprint': carbon_footprint,
                'composite_score': self._calculate_composite_score(
                    supplier, unit_price, risk_score, carbon_footprint
                )
            })
        
        # Sort by composite score (lower is better)
        return sorted(scored, key=lambda x: x['composite_score'])
    
    def _calculate_dynamic_price(self, supplier: SupplierInfo, quantity: float) -> float:
        """Calculate dynamic pricing based on market conditions and quantity"""
        
        base_price = supplier.base_price
        
        # Volume discount
        if quantity > 10000:
            volume_discount = 0.15
        elif quantity > 5000:
            volume_discount = 0.10
        elif quantity > 1000:
            volume_discount = 0.05
        else:
            volume_discount = 0
        
        # Market condition adjustment (simplified)
        market_factor = 1.0 + np.random.normal(0, 0.05)  # ±5% market fluctuation
        
        # Capacity utilization effect
        capacity_pressure = supplier.current_utilization * 0.2  # Up to 20% premium at full capacity
        
        final_price = base_price * (1 - volume_discount) * market_factor * (1 + capacity_pressure)
        
        return max(final_price, base_price * 0.8)  # Minimum 80% of base price
    
    def _calculate_supplier_risk(self, supplier: SupplierInfo, delivery_date: datetime) -> float:
        """Calculate risk score for supplier"""
        
        # Base risk from reliability score
        base_risk = 1.0 - supplier.reliability_score
        
        # Lead time risk
        time_to_delivery = (delivery_date - datetime.now()).days
        lead_time_risk = max(0, (supplier.lead_time - time_to_delivery) / 30.0)
        
        # Capacity risk
        capacity_risk = supplier.current_utilization * 0.3
        
        # Quality risk
        quality_risk = (5.0 - supplier.quality_rating) / 5.0 * 0.4
        
        total_risk = min(1.0, base_risk + lead_time_risk + capacity_risk + quality_risk)
        return total_risk
    
    def _calculate_carbon_footprint(self, supplier: SupplierInfo, quantity: float) -> float:
        """Calculate carbon footprint for transportation and production"""
        
        # Distance-based transportation (simplified)
        # In practice, use actual distances and transportation modes
        distance_factor = {
            'local': 50,      # km
            'regional': 200,  # km
            'national': 500,  # km
            'international': 2000  # km
        }
        
        # Transportation emissions (kg CO2 per ton-km)
        transport_emissions = 0.1
        
        # Production emissions (kg CO2 per kg material)
        production_emissions = {
            'cement': 0.9,
            'chemicals': 2.5,
            'aggregates': 0.02
        }
        
        material_type = supplier.material_type
        distance = distance_factor.get(supplier.location, 200)
        
        transport_co2 = (quantity / 1000) * distance * transport_emissions  # Convert to tons
        production_co2 = quantity * production_emissions.get(material_type, 0.5)
        
        return transport_co2 + production_co2
    
    def _calculate_composite_score(self, 
                                 supplier: SupplierInfo,
                                 unit_price: float,
                                 risk_score: float,
                                 carbon_footprint: float) -> float:
        """Calculate composite score for supplier ranking"""
        
        # Normalize components (lower is better)
        price_score = unit_price / 1000  # Normalize price
        risk_component = risk_score * 2.0  # Risk weight
        carbon_component = carbon_footprint / 1000  # Normalize carbon
        
        # Weighted composite score
        composite = (
            0.5 * price_score +      # 50% price weight
            0.3 * risk_component +   # 30% risk weight
            0.2 * carbon_component   # 20% environmental weight
        )
        
        return composite
    
    def _generate_alternatives(self,
                             requirements: List[MaterialRequirement],
                             primary_recommendations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate alternative sourcing options"""
        
        alternatives = []
        
        for req in requirements:
            suitable_suppliers = self._find_suitable_suppliers(
                req.material_type, req.quantity, req.delivery_date
            )
            
            # Get top 3 alternatives (excluding primary choice)
            primary_supplier_ids = {rec['supplier_id'] for rec in primary_recommendations}
            alternative_suppliers = [
                s for s in suitable_suppliers 
                if s.id not in primary_supplier_ids
            ][:3]
            
            for supplier in alternative_suppliers:
                unit_price = self._calculate_dynamic_price(supplier, req.quantity)
                alternatives.append({
                    'material': req.material_type,
                    'supplier': supplier.name,
                    'supplier_id': supplier.id,
                    'unit_price': unit_price,
                    'total_cost': unit_price * req.quantity,
                    'lead_time': supplier.lead_time,
                    'reliability': supplier.reliability_score,
                    'reason': self._get_alternative_reason(supplier)
                })
        
        return alternatives
    
    def _get_alternative_reason(self, supplier: SupplierInfo) -> str:
        """Generate reason for suggesting this alternative"""
        
        reasons = []
        
        if supplier.reliability_score > 0.9:
            reasons.append("High reliability")
        if supplier.lead_time < 5:
            reasons.append("Fast delivery")
        if supplier.quality_rating > 4.5:
            reasons.append("Premium quality")
        if supplier.current_utilization < 0.7:
            reasons.append("Low capacity pressure")
            
        return "; ".join(reasons) if reasons else "Alternative option available"
    
    def inventory_monitoring(self, current_inventory: Dict[str, float]) -> Dict[str, Any]:
        """Monitor inventory levels and generate alerts"""
        
        alerts = {
            'low_stock_alerts': [],
            'overstock_warnings': [],
            'reorder_recommendations': [],
            'inventory_summary': {}
        }
        
        safety_stock_levels = {
            'cement': 1000,      # kg
            'chemicals': 100,    # kg
            'aggregates': 5000   # kg
        }
        
        max_stock_levels = {
            'cement': 50000,
            'chemicals': 5000,
            'aggregates': 100000
        }
        
        for material, current_level in current_inventory.items():
            safety_level = safety_stock_levels.get(material, 500)
            max_level = max_stock_levels.get(material, 10000)
            
            # Low stock alert
            if current_level < safety_level:
                alerts['low_stock_alerts'].append({
                    'material': material,
                    'current_level': current_level,
                    'safety_level': safety_level,
                    'urgency': 'high' if current_level < safety_level * 0.5 else 'medium'
                })
            
            # Overstock warning
            if current_level > max_level * 0.9:
                alerts['overstock_warnings'].append({
                    'material': material,
                    'current_level': current_level,
                    'max_level': max_level,
                    'excess_percentage': ((current_level - max_level) / max_level) * 100
                })
            
            # Reorder recommendations
            if current_level < safety_level * 1.5:
                recommended_order = safety_level * 2 - current_level
                alerts['reorder_recommendations'].append({
                    'material': material,
                    'recommended_quantity': max(0, recommended_order),
                    'target_level': safety_level * 2
                })
        
        alerts['inventory_summary'] = {
            'total_materials': len(current_inventory),
            'low_stock_items': len(alerts['low_stock_alerts']),
            'overstock_items': len(alerts['overstock_warnings'])
        }
        
        return alerts
    
    def cost_benefit_analysis(self,
                            current_sourcing: Dict[str, Any],
                            alternative_sourcing: Dict[str, Any]) -> Dict[str, float]:
        """Perform cost-benefit analysis for sourcing changes"""
        
        analysis = {
            'cost_difference': 0,
            'risk_difference': 0,
            'carbon_difference': 0,
            'net_benefit': 0,
            'recommendation': ''
        }
        
        # Calculate differences
        cost_diff = alternative_sourcing['total_cost'] - current_sourcing['total_cost']
        risk_diff = alternative_sourcing['risk_score'] - current_sourcing['risk_score']
        carbon_diff = alternative_sourcing['carbon_footprint'] - current_sourcing['carbon_footprint']
        
        analysis['cost_difference'] = cost_diff
        analysis['risk_difference'] = risk_diff
        analysis['carbon_difference'] = carbon_diff
        
        # Calculate net benefit (simplified)
        # Positive values indicate improvement
        net_benefit = (
            -cost_diff * 0.5 +        # Cost savings weight
            -risk_diff * 1000 +       # Risk reduction benefit
            -carbon_diff * 0.1        # Environmental benefit
        )
        
        analysis['net_benefit'] = net_benefit
        
        # Generate recommendation
        if net_benefit > 100:
            analysis['recommendation'] = 'Strongly recommended to switch'
        elif net_benefit > 0:
            analysis['recommendation'] = 'Recommended to switch'
        elif net_benefit > -100:
            analysis['recommendation'] = 'Marginal benefit - consider other factors'
        else:
            analysis['recommendation'] = 'Not recommended to switch'
        
        return analysis

# Example usage
def run_supply_chain_optimization(material_requirements: List[Dict[str, Any]],
                                supplier_data: List[Dict[str, Any]],
                                current_inventory: Dict[str, float]) -> Dict[str, Any]:
    """Run complete supply chain optimization"""
    
    sc_system = SupplyChainIntelligence()
    
    # Add suppliers
    sc_system.add_suppliers(supplier_data)
    
    # Convert requirements
    requirements = [
        MaterialRequirement(
            material_type=req['material_type'],
            quantity=req['quantity'],
            delivery_date=datetime.now() + timedelta(days=req.get('lead_time_days', 7)),
            priority=req.get('priority', 'medium')
        )
        for req in material_requirements
    ]
    
    # Optimize sourcing
    optimization_result = sc_system.optimize_material_sourcing(requirements)
    
    # Inventory monitoring
    inventory_alerts = sc_system.inventory_monitoring(current_inventory)
    
    # Cost-benefit analysis (example with current vs alternative)
    current_sourcing = {
        'total_cost': 50000,
        'risk_score': 0.25,
        'carbon_footprint': 2500
    }
    
    alternative_sourcing = {
        'total_cost': 48000,
        'risk_score': 0.30,
        'carbon_footprint': 2300
    }
    
    cost_benefit = sc_system.cost_benefit_analysis(current_sourcing, alternative_sourcing)
    
    return {
        'optimization_result': {
            'suppliers': optimization_result.suppliers,
            'total_cost': optimization_result.total_cost,
            'lead_time': optimization_result.lead_time,
            'risk_score': optimization_result.risk_score,
            'carbon_footprint': optimization_result.carbon_footprint
        },
        'inventory_monitoring': inventory_alerts,
        'cost_benefit_analysis': cost_benefit,
        'alternative_options': optimization_result.alternative_options
    }