"""Advanced Analytics Engine for CastOptAI Phase 3"""
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, Any
from datetime import datetime, timedelta
import json
import logging
from dataclasses import dataclass
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

logger = logging.getLogger(__name__)

@dataclass
class AnalyticsInsight:
    """Structured insight from analytics"""
    title: str
    description: str
    severity: str  # 'high', 'medium', 'low'
    recommendations: List[str]
    data_points: Dict[str, Any]

@dataclass
class TrendAnalysis:
    """Trend analysis result"""
    metric: str
    trend_direction: str  # 'increasing', 'decreasing', 'stable'
    trend_strength: float  # -1 to 1 scale
    period: str
    confidence: float

class AdvancedAnalyticsEngine:
    """Advanced analytics and insights engine"""
    
    def __init__(self):
        self.data_store = {}
        self.insights_history = []
        
    def analyze_optimization_patterns(self, optimization_history: List[Dict]) -> List[AnalyticsInsight]:
        """Analyze patterns in optimization results"""
        insights = []
        
        if not optimization_history:
            return insights
        
        # Convert to DataFrame for analysis
        df = pd.DataFrame(optimization_history)
        
        # Cost trend analysis
        if 'cost' in df.columns:
            cost_trend = self._analyze_cost_trends(df)
            if cost_trend:
                insights.append(cost_trend)
        
        # Recipe optimization patterns
        recipe_cols = ['cement', 'chemicals', 'steam_hours', 'water']
        recipe_cols_present = [col for col in recipe_cols if col in df.columns]
        
        if recipe_cols_present:
            recipe_insights = self._analyze_recipe_patterns(df, recipe_cols_present)
            insights.extend(recipe_insights)
        
        # Performance analysis
        perf_cols = ['predicted_strength', 'confidence_score', 'cost_savings_percent']
        perf_cols_present = [col for col in perf_cols if col in df.columns]
        
        if perf_cols_present:
            perf_insights = self._analyze_performance_patterns(df, perf_cols_present)
            insights.extend(perf_insights)
        
        return insights
    
    def _analyze_cost_trends(self, df: pd.DataFrame) -> Optional[AnalyticsInsight]:
        """Analyze cost trends and patterns"""
        if 'cost' not in df.columns or len(df) < 2:
            return None
        
        costs = df['cost'].dropna()
        if len(costs) < 2:
            return None
        
        # Calculate trend
        recent_avg = costs.tail(10).mean() if len(costs) >= 10 else costs.mean()
        overall_avg = costs.mean()
        trend_pct = ((recent_avg - overall_avg) / overall_avg) * 100
        
        # Determine severity
        if abs(trend_pct) > 10:
            severity = 'high'
        elif abs(trend_pct) > 5:
            severity = 'medium'
        else:
            severity = 'low'
        
        # Create insight
        if trend_pct > 0:
            title = "Cost Increase Trend Detected"
            description = f"Recent costs are {trend_pct:.1f}% higher than historical average"
            recommendations = [
                "Review material pricing and supplier contracts",
                "Consider alternative optimization strategies",
                "Check for seasonal pricing effects"
            ]
        else:
            title = "Cost Optimization Trend"
            description = f"Recent costs are {abs(trend_pct):.1f}% lower than historical average"
            recommendations = [
                "Maintain current optimization strategies",
                "Document successful approaches for replication",
                "Monitor for quality impacts"
            ]
        
        return AnalyticsInsight(
            title=title,
            description=description,
            severity=severity,
            recommendations=recommendations,
            data_points={
                'recent_average': recent_avg,
                'historical_average': overall_avg,
                'trend_percentage': trend_pct,
                'sample_size': len(costs)
            }
        )
    
    def _analyze_recipe_patterns(self, df: pd.DataFrame, recipe_cols: List[str]) -> List[AnalyticsInsight]:
        """Analyze patterns in recipe compositions"""
        insights = []
        
        for col in recipe_cols:
            if col in df.columns:
                series = df[col].dropna()
                if len(series) > 5:  # Need minimum data points
                    avg_val = series.mean()
                    std_val = series.std()
                    current_val = series.iloc[-1] if len(series) > 0 else avg_val
                    
                    deviation = abs(current_val - avg_val) / std_val if std_val > 0 else 0
                    
                    if deviation > 2:  # More than 2 standard deviations
                        severity = 'high'
                        recs = [f"Review {col} usage - current value is significantly different from typical"]
                    elif deviation > 1:
                        severity = 'medium'
                        recs = [f"Monitor {col} usage - current value differs from typical"]
                    else:
                        severity = 'low'
                        recs = [f"{col} usage is within normal range"]
                    
                    insights.append(AnalyticsInsight(
                        title=f"{col.title()} Usage Pattern",
                        description=f"Current {col} value ({current_val:.1f}) vs. average ({avg_val:.1f})",
                        severity=severity,
                        recommendations=recs,
                        data_points={
                            'current_value': current_val,
                            'average_value': avg_val,
                            'std_deviation': std_val,
                            'deviation_sigma': deviation
                        }
                    ))
        
        return insights
    
    def _analyze_performance_patterns(self, df: pd.DataFrame, perf_cols: List[str]) -> List[AnalyticsInsight]:
        """Analyze performance patterns"""
        insights = []
        
        for col in perf_cols:
            if col in df.columns:
                series = df[col].dropna()
                if len(series) > 5:
                    avg_val = series.mean()
                    current_val = series.iloc[-1] if len(series) > 0 else avg_val
                    
                    if 'savings' in col.lower():
                        # Higher is better for savings
                        performance_change = ((current_val - avg_val) / avg_val) * 100 if avg_val != 0 else 0
                    else:
                        # Lower might be better for costs, higher for strengths
                        performance_change = ((current_val - avg_val) / avg_val) * 100 if avg_val != 0 else 0
                    
                    if abs(performance_change) > 10:
                        severity = 'high'
                    elif abs(performance_change) > 5:
                        severity = 'medium'
                    else:
                        severity = 'low'
                    
                    insights.append(AnalyticsInsight(
                        title=f"{col.replace('_', ' ').title()} Trend",
                        description=f"Current {col}: {current_val:.2f}, Average: {avg_val:.2f} ({performance_change:+.1f}%)",
                        severity=severity,
                        recommendations=[f"Continue monitoring {col} performance"],
                        data_points={
                            'current_value': current_val,
                            'average_value': avg_val,
                            'change_percentage': performance_change
                        }
                    ))
        
        return insights
    
    def cluster_similar_projects(self, projects_data: List[Dict]) -> Dict[str, Any]:
        """Cluster similar projects based on requirements and outcomes"""
        if not projects_data:
            return {}
        
        df = pd.DataFrame(projects_data)
        
        # Select clustering features
        feature_cols = []
        for col in ['target_strength', 'target_time', 'cement', 'chemicals', 'steam_hours', 'cost']:
            if col in df.columns:
                feature_cols.append(col)
        
        if not feature_cols:
            return {}
        
        # Prepare features
        feature_df = df[feature_cols].fillna(0)
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(feature_df)
        
        # Determine optimal clusters (simplified)
        n_clusters = min(5, len(feature_df), max(2, len(feature_df) // 10))
        
        # Perform clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(scaled_features)
        
        # Create cluster analysis
        clusters = {}
        for i in range(n_clusters):
            cluster_mask = cluster_labels == i
            cluster_data = df[cluster_mask]
            
            clusters[f'cluster_{i}'] = {
                'size': int(cluster_data.shape[0]),
                'avg_values': {col: float(cluster_data[col].mean()) for col in feature_cols},
                'characteristics': self._describe_cluster_characteristics(cluster_data, feature_cols)
            }
        
        return {
            'clusters': clusters,
            'n_clusters': n_clusters,
            'feature_importance': feature_cols
        }
    
    def _describe_cluster_characteristics(self, cluster_data: pd.DataFrame, feature_cols: List[str]) -> str:
        """Describe characteristics of a cluster"""
        avg_vals = [cluster_data[col].mean() for col in feature_cols]
        feature_avg_pairs = list(zip(feature_cols, avg_vals))
        feature_avg_pairs.sort(key=lambda x: x[1], reverse=True)  # Sort by value
        
        top_features = [f"{col} ({val:.1f})" for col, val in feature_avg_pairs[:3]]
        return f"Characterized by: {', '.join(top_features)}"
    
    def generate_performance_report(self, optimization_data: List[Dict]) -> Dict[str, Any]:
        """Generate comprehensive performance report"""
        if not optimization_data:
            return {'error': 'No data available for report'}
        
        df = pd.DataFrame(optimization_data)
        
        report = {
            'summary': {
                'total_optimizations': len(optimization_data),
                'date_range': {
                    'start': str(df.get('date', pd.Series()).min()) if 'date' in df.columns else 'N/A',
                    'end': str(df.get('date', pd.Series()).max()) if 'date' in df.columns else 'N/A'
                }
            },
            'cost_analysis': self._generate_cost_analysis(df),
            'efficiency_metrics': self._generate_efficiency_metrics(df),
            'trend_analysis': self._generate_trend_analysis(df),
            'recommendations': self._generate_recommendations(df)
        }
        
        return report
    
    def _generate_cost_analysis(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Generate cost analysis section"""
        if 'cost' not in df.columns:
            return {'error': 'Cost data not available'}
        
        costs = df['cost'].dropna()
        if len(costs) == 0:
            return {'error': 'No cost data available'}
        
        return {
            'average_cost': float(costs.mean()),
            'cost_std_dev': float(costs.std()),
            'min_cost': float(costs.min()),
            'max_cost': float(costs.max()),
            'cost_savings_opportunity': float(costs.max() - costs.min()) if len(costs) > 1 else 0
        }
    
    def _generate_efficiency_metrics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Generate efficiency metrics section"""
        metrics = {}
        
        # Cost per strength unit
        if 'cost' in df.columns and 'predicted_strength' in df.columns:
            df_copy = df.copy()
            df_copy = df_copy[(df_copy['predicted_strength'] > 0) & (df_copy['cost'] > 0)]
            if len(df_copy) > 0:
                df_copy = df_copy.copy()
                df_copy['cost_per_strength'] = df_copy['cost'] / df_copy['predicted_strength']
                metrics['avg_cost_per_strength'] = float(df_copy['cost_per_strength'].mean())
        
        # Savings percentage
        if 'cost_savings_percent' in df.columns:
            savings = df['cost_savings_percent'].dropna()
            if len(savings) > 0:
                metrics['avg_savings_percent'] = float(savings.mean())
                metrics['savings_range'] = {
                    'min': float(savings.min()),
                    'max': float(savings.max())
                }
        
        return metrics
    
    def _generate_trend_analysis(self, df: pd.DataFrame) -> List[TrendAnalysis]:
        """Generate trend analysis for key metrics"""
        trends = []
        
        # Look for time-based columns
        time_cols = [col for col in df.columns if 'date' in col.lower() or 'time' in col.lower()]
        
        if time_cols and len(df) > 1:
            # Sort by date if available
            df_sorted = df.copy().sort_values(by=time_cols[0]) if time_cols else df.copy()
            
            # Analyze trends for numeric columns
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            
            for col in numeric_cols[:5]:  # Limit to first 5 numeric columns
                series = df_sorted[col].dropna()
                if len(series) > 2:  # Need at least 3 points for trend
                    # Simple linear trend analysis
                    x = np.arange(len(series))
                    slope, _ = np.polyfit(x, series.values, 1)
                    trend_strength = slope / series.mean() if series.mean() != 0 else 0
                    
                    if trend_strength > 0.01:
                        direction = 'increasing'
                    elif trend_strength < -0.01:
                        direction = 'decreasing'
                    else:
                        direction = 'stable'
                    
                    trends.append(TrendAnalysis(
                        metric=col,
                        trend_direction=direction,
                        trend_strength=float(trend_strength),
                        period='full_data_period',
                        confidence=min(1.0, len(series) / 10.0)  # Simple confidence based on data size
                    ))
        
        return trends
    
    def _generate_recommendations(self, df: pd.DataFrame) -> List[str]:
        """Generate high-level recommendations"""
        recommendations = []
        
        # Cost recommendations
        if 'cost' in df.columns:
            costs = df['cost'].dropna()
            if len(costs) > 5:
                recent_costs = costs.tail(5).mean()
                historical_costs = costs.head(-5).mean() if len(costs) > 10 else costs.mean()
                
                if recent_costs > historical_costs * 1.1:
                    recommendations.append("Recent costs are elevated - investigate optimization strategies")
                elif recent_costs < historical_costs * 0.9:
                    recommendations.append("Cost improvements noted - document successful approaches")
        
        # Performance recommendations
        if 'cost_savings_percent' in df.columns:
            savings = df['cost_savings_percent'].dropna()
            if len(savings) > 0 and savings.mean() < 10:
                recommendations.append("Average savings below target - consider model retraining")
            elif savings.mean() > 25:
                recommendations.append("Excellent savings achieved - maintain current strategies")
        
        if not recommendations:
            recommendations.append("Continue current optimization approach")
        
        return recommendations

# Example usage function
def run_advanced_analytics_demo():
    """Demo function to showcase advanced analytics"""
    
    # Sample optimization history
    sample_data = [
        {
            'date': '2024-01-01',
            'target_strength': 20,
            'target_time': 24,
            'cement': 300,
            'chemicals': 2.5,
            'steam_hours': 4,
            'cost': 4500,
            'predicted_strength': 22.5,
            'confidence_score': 92,
            'cost_savings_percent': 25
        },
        {
            'date': '2024-01-02',
            'target_strength': 25,
            'target_time': 48,
            'cement': 350,
            'chemicals': 3.2,
            'steam_hours': 6,
            'cost': 5200,
            'predicted_strength': 27.8,
            'confidence_score': 94,
            'cost_savings_percent': 18
        },
        {
            'date': '2024-01-03',
            'target_strength': 18,
            'target_time': 12,
            'cement': 280,
            'chemicals': 1.8,
            'steam_hours': 2,
            'cost': 3800,
            'predicted_strength': 19.2,
            'confidence_score': 89,
            'cost_savings_percent': 32
        }
    ] * 20  # Repeat for more data points
    
    # Initialize engine
    engine = AdvancedAnalyticsEngine()
    
    # Analyze optimization patterns
    insights = engine.analyze_optimization_patterns(sample_data)
    
    # Cluster similar projects
    clusters = engine.cluster_similar_projects(sample_data)
    
    # Generate performance report
    report = engine.generate_performance_report(sample_data)
    
    return {
        'insights': [i.__dict__ for i in insights],
        'clusters': clusters,
        'performance_report': report
    }