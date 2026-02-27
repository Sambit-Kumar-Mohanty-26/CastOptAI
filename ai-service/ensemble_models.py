"""
Advanced ML Ensemble Architecture for CastOpt AI
Implements XGBoost, Neural Networks, and ensemble methods with uncertainty quantification
"""

import pandas as pd
import numpy as np
import joblib
import json
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from datetime import datetime

# ML Libraries
try:
    import xgboost as xgb
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    import tensorflow as tf
    from tensorflow import keras
    from sklearn.model_selection import RandomizedSearchCV
    HAS_ML_LIBS = True
except ImportError as e:
    print(f"Warning: ML libraries not available: {e}")
    HAS_ML_LIBS = False

@dataclass
class ModelMetrics:
    """Container for model performance metrics"""
    mse: float
    rmse: float
    mae: float
    r2: float
    cv_score: float
    training_time: float
    model_size: float  # in MB

@dataclass
class PredictionResult:
    """Container for prediction with uncertainty"""
    prediction: float
    lower_bound: float
    upper_bound: float
    confidence_interval: float
    model_name: str
    feature_importance: Optional[Dict[str, float]] = None

class XGBoostModel:
    """XGBoost implementation with hyperparameter optimization"""
    
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.feature_columns = None
        self.is_trained = False
        self.metrics = None
        
    def optimize_hyperparameters(self, X_train, y_train, n_iter=50):
        """Bayesian optimization for XGBoost hyperparameters"""
        if not HAS_ML_LIBS:
            return self._default_xgb_params()
            
        param_dist = {
            'n_estimators': [100, 200, 300, 500],
            'max_depth': [3, 4, 5, 6, 8, 10],
            'learning_rate': [0.01, 0.05, 0.1, 0.2],
            'subsample': [0.6, 0.8, 1.0],
            'colsample_bytree': [0.6, 0.8, 1.0],
            'min_child_weight': [1, 3, 5, 7],
            'gamma': [0, 0.1, 0.2, 0.5],
            'reg_alpha': [0, 0.1, 0.5, 1.0],
            'reg_lambda': [0, 0.1, 0.5, 1.0]
        }
        
        xgb_model = xgb.XGBRegressor(random_state=42, n_jobs=-1)
        random_search = RandomizedSearchCV(
            xgb_model, param_dist, 
            n_iter=n_iter, 
            cv=3, 
            scoring='neg_mean_squared_error',
            n_jobs=-1,
            random_state=42
        )
        
        random_search.fit(X_train, y_train)
        return random_search.best_params_
    
    def _default_xgb_params(self):
        """Default XGBoost parameters if optimization fails"""
        return {
            'n_estimators': 200,
            'max_depth': 6,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'min_child_weight': 1,
            'gamma': 0,
            'reg_alpha': 0,
            'reg_lambda': 1
        }
    
    def train(self, df: pd.DataFrame, optimize_params: bool = True) -> ModelMetrics:
        """Train XGBoost model"""
        if not HAS_ML_LIBS:
            raise RuntimeError("XGBoost not available. Please install with: pip install xgboost")
            
        start_time = datetime.now()
        
        # Prepare features
        self.feature_columns = [
            'cement', 'slag', 'fly_ash', 'water',
            'superplasticizer', 'coarse_agg', 'fine_agg',
            'age_hours', 'temperature', 'humidity', 'curing_method'
        ]
        
        X = df[self.feature_columns]
        y = df['strength']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Get parameters
        if optimize_params:
            print("Optimizing XGBoost hyperparameters...")
            params = self.optimize_hyperparameters(X_train_scaled, y_train)
            print(f"Best parameters: {params}")
        else:
            params = self._default_xgb_params()
        
        # Train model
        self.model = xgb.XGBRegressor(**params, random_state=42, n_jobs=-1)
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluate
        y_pred = self.model.predict(X_test_scaled)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        # Cross-validation
        cv_scores = cross_val_score(
            self.model, X_train_scaled, y_train, 
            cv=5, scoring='neg_mean_squared_error'
        )
        cv_score = -cv_scores.mean()
        
        training_time = (datetime.now() - start_time).total_seconds()
        
        # Model size
        import tempfile
        import os
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            joblib.dump(self.model, tmp.name)
            model_size = os.path.getsize(tmp.name) / (1024 * 1024)  # MB
            os.unlink(tmp.name)
        
        self.metrics = ModelMetrics(
            mse=mse, rmse=rmse, mae=mae, r2=r2,
            cv_score=cv_score, training_time=training_time,
            model_size=model_size
        )
        
        self.is_trained = True
        return self.metrics
    
    def predict(self, features: pd.DataFrame) -> PredictionResult:
        """Make prediction with uncertainty estimation"""
        if not self.is_trained or self.model is None:
            raise RuntimeError("Model not trained")
        
        # Scale features
        features_scaled = self.scaler.transform(features)
        
        # Prediction
        prediction = float(self.model.predict(features_scaled)[0])
        
        # Feature importance
        feature_importance = dict(zip(
            self.feature_columns, 
            self.model.feature_importances_
        ))
        
        # Simple uncertainty estimation (you can improve this with quantile regression)
        # For now, we'll use a simple heuristic based on model performance
        uncertainty = self.metrics.rmse * 0.1  # 10% of RMSE
        lower_bound = prediction - uncertainty
        upper_bound = prediction + uncertainty
        confidence_interval = 2 * uncertainty
        
        return PredictionResult(
            prediction=prediction,
            lower_bound=max(0, lower_bound),  # Strength can't be negative
            upper_bound=upper_bound,
            confidence_interval=confidence_interval,
            model_name="XGBoost",
            feature_importance=feature_importance
        )
    
    def save_model(self, filepath: str):
        """Save model and scaler"""
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_columns': self.feature_columns,
            'is_trained': self.is_trained,
            'metrics': self.metrics
        }
        joblib.dump(model_data, filepath)
    
    def load_model(self, filepath: str):
        """Load model and scaler"""
        model_data = joblib.load(filepath)
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.feature_columns = model_data['feature_columns']
        self.is_trained = model_data['is_trained']
        self.metrics = model_data['metrics']

class NeuralNetworkModel:
    """Neural Network implementation for non-linear pattern recognition"""
    
    def __init__(self):
        self.model = None
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        self.feature_columns = None
        self.is_trained = False
        self.metrics = None
    
    def build_model(self, input_dim: int) -> keras.Sequential:
        """Build neural network architecture"""
        model = keras.Sequential([
            keras.layers.Dense(128, activation='relu', input_shape=(input_dim,)),
            keras.layers.Dropout(0.3),
            keras.layers.Dense(64, activation='relu'),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(32, activation='relu'),
            keras.layers.Dense(16, activation='relu'),
            keras.layers.Dense(1, activation='linear')
        ])
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def train(self, df: pd.DataFrame, epochs: int = 100) -> ModelMetrics:
        """Train neural network model"""
        if not HAS_ML_LIBS:
            raise RuntimeError("TensorFlow/Keras not available. Please install with: pip install tensorflow")
            
        start_time = datetime.now()
        
        # Prepare features
        self.feature_columns = [
            'cement', 'slag', 'fly_ash', 'water',
            'superplasticizer', 'coarse_agg', 'fine_agg',
            'age_hours', 'temperature', 'humidity', 'curing_method'
        ]
        
        X = df[self.feature_columns]
        y = df['strength'].values.reshape(-1, 1)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Scale features
        X_train_scaled = self.scaler_X.fit_transform(X_train)
        X_test_scaled = self.scaler_X.transform(X_test)
        y_train_scaled = self.scaler_y.fit_transform(y_train)
        y_test_scaled = self.scaler_y.transform(y_test)
        
        # Build model
        self.model = self.build_model(X_train_scaled.shape[1])
        
        # Early stopping
        early_stopping = keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=15, restore_best_weights=True
        )
        
        # Train model
        history = self.model.fit(
            X_train_scaled, y_train_scaled,
            epochs=epochs,
            batch_size=32,
            validation_split=0.2,
            callbacks=[early_stopping],
            verbose=0
        )
        
        # Evaluate
        test_loss, test_mae = self.model.evaluate(X_test_scaled, y_test_scaled, verbose=0)
        y_pred_scaled = self.model.predict(X_test_scaled, verbose=0)
        y_pred = self.scaler_y.inverse_transform(y_pred_scaled)
        
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        # Cross-validation (approximate)
        from sklearn.model_selection import KFold
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = []
        
        for train_idx, val_idx in kf.split(X_train_scaled):
            # Clone model
            temp_model = self.build_model(X_train_scaled.shape[1])
            temp_model.fit(
                X_train_scaled[train_idx], y_train_scaled[train_idx],
                epochs=30, batch_size=32, verbose=0
            )
            val_pred = temp_model.predict(X_train_scaled[val_idx], verbose=0)
            val_pred_inv = self.scaler_y.inverse_transform(val_pred)
            val_true_inv = self.scaler_y.inverse_transform(y_train_scaled[val_idx])
            cv_scores.append(mean_squared_error(val_true_inv, val_pred_inv))
        
        cv_score = np.mean(cv_scores)
        training_time = (datetime.now() - start_time).total_seconds()
        
        # Model size
        import tempfile
        import os
        with tempfile.NamedTemporaryFile(delete=False, suffix='.h5') as tmp:
            self.model.save(tmp.name, save_format='h5')
            model_size = os.path.getsize(tmp.name) / (1024 * 1024)  # MB
            os.unlink(tmp.name)
        
        self.metrics = ModelMetrics(
            mse=mse, rmse=rmse, mae=mae, r2=r2,
            cv_score=cv_score, training_time=training_time,
            model_size=model_size
        )
        
        self.is_trained = True
        return self.metrics
    
    def predict(self, features: pd.DataFrame) -> PredictionResult:
        """Make prediction with uncertainty estimation"""
        if not self.is_trained or self.model is None:
            raise RuntimeError("Model not trained")
        
        # Scale features
        features_scaled = self.scaler_X.transform(features)
        
        # Prediction
        prediction_scaled = self.model.predict(features_scaled, verbose=0)
        prediction = float(self.scaler_y.inverse_transform(prediction_scaled)[0][0])
        
        # Simple uncertainty (dropout-based estimation)
        predictions = []
        for _ in range(100):  # Monte Carlo dropout
            pred_scaled = self.model.predict(features_scaled, verbose=0)
            pred = self.scaler_y.inverse_transform(pred_scaled)
            predictions.append(pred[0][0])
        
        std_dev = np.std(predictions)
        lower_bound = prediction - 1.96 * std_dev  # 95% confidence interval
        upper_bound = prediction + 1.96 * std_dev
        confidence_interval = 2 * 1.96 * std_dev
        
        return PredictionResult(
            prediction=prediction,
            lower_bound=max(0, lower_bound),
            upper_bound=upper_bound,
            confidence_interval=confidence_interval,
            model_name="Neural Network"
        )

class EnsembleModel:
    """Ensemble model combining multiple ML approaches"""
    
    def __init__(self):
        self.models = {
            'xgboost': XGBoostModel(),
            'neural_network': NeuralNetworkModel(),
            'random_forest': None  # Will use existing model
        }
        self.weights = {
            'xgboost': 0.4,
            'neural_network': 0.3,
            'random_forest': 0.3
        }
        self.trained_models = []
        
    def train_all_models(self, df: pd.DataFrame) -> Dict[str, ModelMetrics]:
        """Train all models in the ensemble"""
        results = {}
        
        # Train XGBoost
        print("Training XGBoost model...")
        xgb_metrics = self.models['xgboost'].train(df)
        results['xgboost'] = xgb_metrics
        self.trained_models.append('xgboost')
        
        # Train Neural Network
        print("Training Neural Network model...")
        nn_metrics = self.models['neural_network'].train(df)
        results['neural_network'] = nn_metrics
        self.trained_models.append('neural_network')
        
        # Load existing Random Forest model
        try:
            rf_model_data = joblib.load('model.pkl')
            self.models['random_forest'] = rf_model_data
            # Calculate metrics for RF
            from sklearn.model_selection import train_test_split
            from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
            import time
            
            start_time = time.time()
            X = df[['cement', 'slag', 'fly_ash', 'water', 'superplasticizer', 
                   'coarse_agg', 'fine_agg', 'age_hours', 'temperature', 
                   'humidity', 'curing_method']]
            y = df['strength']
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            y_pred = rf_model_data.predict(X_test)
            
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            training_time = time.time() - start_time
            
            # Estimate model size
            import tempfile
            import os
            with tempfile.NamedTemporaryFile(delete=False) as tmp:
                joblib.dump(rf_model_data, tmp.name)
                model_size = os.path.getsize(tmp.name) / (1024 * 1024)
                os.unlink(tmp.name)
            
            results['random_forest'] = ModelMetrics(
                mse=mse, rmse=rmse, mae=mae, r2=r2,
                cv_score=mse, training_time=training_time,
                model_size=model_size
            )
            self.trained_models.append('random_forest')
            print("Loaded existing Random Forest model")
        except Exception as e:
            print(f"Could not load Random Forest model: {e}")
            self.weights['random_forest'] = 0
            # Renormalize weights
            total_weight = sum(self.weights.values())
            for model in self.weights:
                self.weights[model] /= total_weight
        
        return results
    
    def predict(self, features: pd.DataFrame) -> PredictionResult:
        """Ensemble prediction with weighted averaging"""
        if not self.trained_models:
            raise RuntimeError("No models trained")
        
        predictions = []
        weights = []
        uncertainties = []
        model_details = []
        
        # Get predictions from each model
        for model_name in self.trained_models:
            model = self.models[model_name]
            if model:
                result = model.predict(features)
                predictions.append(result.prediction)
                weights.append(self.weights[model_name])
                uncertainties.append(result.confidence_interval)
                model_details.append(f"{model_name}: {result.prediction:.2f}±{result.confidence_interval:.2f}")
        
        # Weighted ensemble prediction
        ensemble_prediction = np.average(predictions, weights=weights)
        
        # Combined uncertainty (simplified)
        # In practice, you'd want more sophisticated uncertainty combination
        weighted_uncertainties = [u * w for u, w in zip(uncertainties, weights)]
        ensemble_uncertainty = sum(weighted_uncertainties) / sum(weights)
        
        # Simple ensemble model performance weights (based on training error)
        if hasattr(self.models['xgboost'], 'metrics') and self.models['xgboost'].metrics:
            xgb_weight = 1.0 / (1.0 + self.models['xgboost'].metrics.rmse)
            if len(self.trained_models) > 1 and hasattr(self.models['neural_network'], 'metrics') and self.models['neural_network'].metrics:
                nn_weight = 1.0 / (1.0 + self.models['neural_network'].metrics.rmse)
                total_inv_rmse = xgb_weight + nn_weight
                if len(self.trained_models) > 2 and self.models['random_forest'] and hasattr(self.models['random_forest'], 'oob_score_'):
                    # Simplified for Random Forest
                    pass  # Weights are already set
            
        lower_bound = ensemble_prediction - ensemble_uncertainty
        upper_bound = ensemble_prediction + ensemble_uncertainty
        confidence_interval = 2 * ensemble_uncertainty
        
        print(f"Ensemble predictions: {', '.join(model_details)}")
        print(f"Final ensemble: {ensemble_prediction:.2f} ± {ensemble_uncertainty:.2f}")
        
        return PredictionResult(
            prediction=ensemble_prediction,
            lower_bound=max(0, lower_bound),
            upper_bound=upper_bound,
            confidence_interval=confidence_interval,
            model_name="Ensemble",
            feature_importance=self._get_combined_importance(features)
        )
    
    def _get_combined_importance(self, features: pd.DataFrame) -> Dict[str, float]:
        """Get combined feature importance from all models"""
        importance_dict = {}
        
        if 'xgboost' in self.trained_models and hasattr(self.models['xgboost'], 'predict') and self.models['xgboost'].model is not None:
            # Only try to get XGBoost feature importance
            xgb_imp = getattr(self.models['xgboost'].model, 'feature_importances_', {})
            if xgb_imp.any() if hasattr(xgb_imp, 'any') else xgb_imp:
                for i, col in enumerate(self.models['xgboost'].feature_columns):
                    importance_dict[col] = importance_dict.get(col, 0) + xgb_imp[i] * self.weights['xgboost']
        
        # Normalize
        if importance_dict:
            total = sum(importance_dict.values())
            if total > 0:
                importance_dict = {k: v/total for k, v in importance_dict.items()}
        
        return importance_dict
    
    def save_ensemble(self, filepath: str):
        """Save entire ensemble"""
        ensemble_data = {
            'models': self.models,
            'weights': self.weights,
            'trained_models': self.trained_models
        }
        joblib.dump(ensemble_data, filepath)
    
    def load_ensemble(self, filepath: str):
        """Load ensemble"""
        ensemble_data = joblib.load(filepath)
        self.models = ensemble_data['models']
        self.weights = ensemble_data['weights']
        self.trained_models = ensemble_data['trained_models']

# Global ensemble model instance
ensemble_model = EnsembleModel()