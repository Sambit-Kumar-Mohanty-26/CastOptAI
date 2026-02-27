"""Quick test script to verify Phase 2 components"""
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_phase2_components():
    """Test all Phase 2 components"""
    print("🧪 Testing Phase 2 Components...")
    
    try:
        # Test Pareto Optimizer
        from pareto_optimizer import ParetoOptimizer, ObjectiveWeights
        optimizer = ParetoOptimizer()
        optimizer.set_objective_weights(ObjectiveWeights(cost_weight=0.5, time_weight=0.3, strength_weight=0.15, environmental_weight=0.05))
        print("✅ Pareto Optimizer: OK")
        
        # Test Risk Management
        from risk_management import RiskManagementSystem
        risk_system = RiskManagementSystem()
        print("✅ Risk Management System: OK")
        
        # Test Supply Chain
        from supply_chain import SupplyChainIntelligence
        sc_system = SupplyChainIntelligence()
        print("✅ Supply Chain Intelligence: OK")
        
        # Test Integration
        from phase2_integration import Phase2Response
        print("✅ Phase 2 Integration: OK")
        
        print("\n🎉 All Phase 2 components loaded successfully!")
        print("\n📋 Phase 2 Features Available:")
        print("• Multi-objective Pareto Optimization")
        print("• Custom weight sliders and trade-off visualization")
        print("• Monte Carlo simulations for uncertainty quantification")
        print("• Defect probability modeling")
        print("• Supplier reliability scoring")
        print("• Supply chain optimization with cost-benefit analysis")
        print("• Inventory monitoring and alerts")
        print("• Emergency protocol triggers")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing Phase 2 components: {e}")
        return False

if __name__ == "__main__":
    test_phase2_components()