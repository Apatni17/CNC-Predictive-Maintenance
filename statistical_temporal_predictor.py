import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import mahalanobis
from scipy.stats import chi2
from pathlib import Path
import joblib
import warnings
warnings.filterwarnings('ignore')

class StatisticalTemporalWearPredictor:
    def __init__(self):
        self.model = None
        self.feature_names = None
        self.is_trained = False
        self.scaler = StandardScaler()
        
        # Base features for temporal analysis
        self.base_features = [
            'X1_CurrentFeedback', 'X1_DCBusVoltage', 'M1_CURRENT_FEEDRATE', 'X1_OutputPower',
            'X1_ActualVelocity', 'X1_OutputCurrent', 'Y1_CurrentFeedback', 'Z1_CurrentFeedback',
            'X1_ActualPosition', 'X1_CommandVelocity', 'S1_CurrentFeedback', 'X1_OutputVoltage'
        ]
        
        # Statistical analysis results (from previous analysis)
        self.statistical_ranges = {
            'fresh_start': 0.516,    # 51.6% - statistically derived
            'fresh_end': 0.814,      # 81.4% - statistically derived
            'worn_start': 0.507,     # 50.7% - statistically derived  
            'worn_end': 0.936        # 93.6% - statistically derived
        }
        
        # Mahalanobis distance parameters
        self.mahal_threshold = 21.026  # 95% confidence threshold
        self.fresh_baseline_mean = None
        self.fresh_baseline_cov = None
        self.fresh_cov_inv = None
        
        # Experiment classifications
        self.fresh_experiments = [1, 2, 3, 4, 5, 11, 12, 17]
        self.worn_experiments = [6, 7, 8, 9, 10, 13, 14, 15, 16, 18]
        
        # XGBoost parameters optimized for regression
        self.model_params = {
            'objective': 'reg:squarederror',
            'n_estimators': 200,
            'max_depth': 5,
            'learning_rate': 0.08,
            'subsample': 0.9,
            'colsample_bytree': 0.8,
            'random_state': 42,
            'eval_metric': 'rmse'
        }
    
    def calculate_baseline_statistics(self, data_dir="data/CNC mill wear /"):
        """Calculate baseline statistics for Mahalanobis distance"""
        print("📊 Calculating baseline statistics for Mahalanobis distance...")
        
        fresh_data = []
        
        # Load fresh tool experiments for baseline
        for exp_id in self.fresh_experiments:
            filepath = Path(data_dir) / f"experiment_{exp_id:02d}.csv"
            if filepath.exists():
                data = pd.read_csv(filepath)
                # Sample to reduce computation
                if len(data) > 500:
                    data = data.sample(n=500, random_state=42)
                data = data[self.base_features].fillna(0)
                fresh_data.append(data)
        
        fresh_df = pd.concat(fresh_data, ignore_index=True)
        
        # Calculate baseline statistics
        self.fresh_baseline_mean = fresh_df.mean().values
        self.fresh_baseline_cov = fresh_df.cov().values
        
        # Calculate pseudo-inverse for numerical stability
        try:
            self.fresh_cov_inv = np.linalg.inv(self.fresh_baseline_cov)
        except np.linalg.LinAlgError:
            self.fresh_cov_inv = np.linalg.pinv(self.fresh_baseline_cov)
        
        print(f"✅ Baseline calculated from {len(fresh_df)} fresh tool samples")
        return self.fresh_baseline_mean, self.fresh_baseline_cov
    
    def calculate_mahalanobis_distance(self, sample):
        """Calculate Mahalanobis distance for a sample"""
        if self.fresh_baseline_mean is None:
            raise ValueError("Baseline statistics not calculated. Run calculate_baseline_statistics() first.")
        
        return mahalanobis(sample, self.fresh_baseline_mean, self.fresh_cov_inv)
    
    def create_statistical_temporal_features(self, data, experiment_id):
        """Create temporal features with statistical foundation"""
        data = data.copy()
        data = data.sort_index()
        
        # Add time-based features
        data['time_step'] = range(len(data))
        data['time_progress'] = data['time_step'] / (len(data) - 1)
        
        # Statistical wear progression based on data analysis
        if experiment_id in self.fresh_experiments:
            starting_wear = self.statistical_ranges['fresh_start']
            ending_wear = self.statistical_ranges['fresh_end']
            tool_type = 'fresh'
        elif experiment_id in self.worn_experiments:
            starting_wear = self.statistical_ranges['worn_start']
            ending_wear = self.statistical_ranges['worn_end']
            tool_type = 'worn'
        else:
            # Unknown experiment - use fresh tool assumptions
            starting_wear = self.statistical_ranges['fresh_start']
            ending_wear = self.statistical_ranges['fresh_end']
            tool_type = 'unknown'
        
        # Calculate statistical wear progression
        wear_range = ending_wear - starting_wear
        data['wear_progression'] = starting_wear + (data['time_progress'] * wear_range)
        
        # Add Mahalanobis distance features
        if self.fresh_baseline_mean is not None:
            mahal_distances = []
            for _, row in data[self.base_features].fillna(0).iterrows():
                distance = self.calculate_mahalanobis_distance(row.values)
                mahal_distances.append(distance)
            
            data['mahalanobis_distance'] = mahal_distances
            data['mahal_anomaly'] = (data['mahalanobis_distance'] > self.mahal_threshold).astype(int)
            
            # Mahalanobis-based wear indicators
            data['mahal_wear_indicator'] = np.clip(data['mahalanobis_distance'] / self.mahal_threshold, 0, 2)
        
        # Rolling statistics for temporal trends
        window_sizes = [10, 25, 50]
        
        for feature in self.base_features:
            if feature in data.columns:
                # Rolling statistics
                for window in window_sizes:
                    data[f'{feature}_rolling_mean_{window}'] = data[feature].rolling(window=window, min_periods=1).mean()
                    data[f'{feature}_rolling_std_{window}'] = data[feature].rolling(window=window, min_periods=1).std()
                
                # Trend features (slope)
                data[f'{feature}_trend_10'] = data[feature].rolling(window=10, min_periods=5).apply(
                    lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) >= 2 else 0, raw=False
                )
                
                # Change from beginning
                data[f'{feature}_change_from_start'] = data[feature] - data[feature].iloc[0]
                
                # Cumulative change
                data[f'{feature}_cumulative_change'] = (data[feature] - data[feature].shift(1)).fillna(0).cumsum()
        
        # Statistical features
        data['cumulative_machining_time'] = data['time_step']
        data['experiment_id'] = experiment_id
        data['tool_type'] = tool_type
        
        print(f"   📈 Created features for {tool_type} tool (Exp {experiment_id}): {starting_wear:.1%} → {ending_wear:.1%}")
        
        return data
    
    def prepare_statistical_temporal_data(self, data_dir="data/CNC mill wear /"):
        """Prepare temporal training data with statistical foundation"""
        print("🧠 Preparing Statistical Temporal Data...")
        print("=" * 45)
        
        # Calculate baseline statistics first
        self.calculate_baseline_statistics(data_dir)
        
        all_temporal_data = []
        
        # Process experiments 1-16 for training
        for exp_id in range(1, 17):
            filename = f"experiment_{exp_id:02d}.csv"
            filepath = Path(data_dir) / filename
            
            if filepath.exists():
                print(f"📁 Processing {filename} for statistical temporal analysis...")
                
                # Load experiment data
                exp_data = pd.read_csv(filepath)
                
                # Create statistical temporal features
                temporal_data = self.create_statistical_temporal_features(exp_data, exp_id)
                
                # Sample data to manage size (every 5th sample)
                if len(temporal_data) > 300:
                    step = max(1, len(temporal_data) // 300)
                    temporal_data = temporal_data.iloc[::step]
                
                all_temporal_data.append(temporal_data)
                print(f"   ✅ Processed {len(exp_data)} → {len(temporal_data)} statistical temporal samples")
        
        # Combine all data
        combined_data = pd.concat(all_temporal_data, ignore_index=True)
        
        print(f"\n📊 Statistical Temporal Data Summary:")
        print(f"   Total samples: {len(combined_data)}")
        print(f"   Experiments processed: 16")
        print(f"   Features created: {len(combined_data.columns)}")
        print(f"   Fresh tool samples: {len(combined_data[combined_data['tool_type'] == 'fresh'])}")
        print(f"   Worn tool samples: {len(combined_data[combined_data['tool_type'] == 'worn'])}")
        
        return combined_data
    
    def train_statistical_temporal_model(self, data_dir="data/CNC mill wear /"):
        """Train XGBoost model with statistical foundation"""
        print("🎯 Training Statistical Temporal Wear Predictor...")
        
        # Prepare statistical temporal data
        temporal_data = self.prepare_statistical_temporal_data(data_dir)
        
        # Get feature columns
        exclude_cols = ['wear_progression', 'experiment_id', 'time_step', 'time_progress', 'tool_type']
        feature_cols = [col for col in temporal_data.columns if col not in exclude_cols and not col.startswith('Machining_Process')]
        
        # Select numeric features
        numeric_features = []
        for col in feature_cols:
            if temporal_data[col].dtype in ['int64', 'float64']:
                numeric_features.append(col)
        
        print(f"\n🔧 Using {len(numeric_features)} statistical temporal features")
        
        # Prepare features and target
        X = temporal_data[numeric_features].fillna(0)
        y = temporal_data['wear_progression']
        
        # Scale features for better performance
        X_scaled = self.scaler.fit_transform(X)
        
        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42, stratify=temporal_data['experiment_id']
        )
        
        print(f"\n📈 Statistical Temporal Train/Test Split:")
        print(f"   Training samples: {len(X_train)}")
        print(f"   Test samples: {len(X_test)}")
        
        # Train XGBoost regressor
        print(f"\n🤖 Training Statistical XGBoost Model...")
        self.model = xgb.XGBRegressor(**self.model_params)
        self.model.fit(X_train, y_train)
        
        # Evaluate model
        y_train_pred = self.model.predict(X_train)
        y_test_pred = self.model.predict(X_test)
        
        train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
        test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
        
        # Classification metrics (using 0.7 threshold for high wear)
        y_train_class = (y_train > 0.7).astype(int)
        y_train_pred_class = (y_train_pred > 0.7).astype(int)
        y_test_class = (y_test > 0.7).astype(int)
        y_test_pred_class = (y_test_pred > 0.7).astype(int)
        
        train_accuracy = accuracy_score(y_train_class, y_train_pred_class)
        train_precision = precision_score(y_train_class, y_train_pred_class, zero_division=0)
        train_recall = recall_score(y_train_class, y_train_pred_class, zero_division=0)
        train_f1 = f1_score(y_train_class, y_train_pred_class, zero_division=0)
        
        test_accuracy = accuracy_score(y_test_class, y_test_pred_class)
        test_precision = precision_score(y_test_class, y_test_pred_class, zero_division=0)
        test_recall = recall_score(y_test_class, y_test_pred_class, zero_division=0)
        test_f1 = f1_score(y_test_class, y_test_pred_class, zero_division=0)
        
        print(f"\n📊 Statistical Temporal Model Performance:")
        print(f"   Training RMSE: {train_rmse:.3f}")
        print(f"   Test RMSE: {test_rmse:.3f}")
        
        print(f"\n🏋️ Training Classification Metrics (>0.7 threshold):")
        print(f"   Accuracy: {train_accuracy:.3f}")
        print(f"   Precision: {train_precision:.3f}")
        print(f"   Recall: {train_recall:.3f}")
        print(f"   F1-Score: {train_f1:.3f}")
        
        print(f"\n🧪 Test Classification Metrics (>0.7 threshold):")
        print(f"   Accuracy: {test_accuracy:.3f}")
        print(f"   Precision: {test_precision:.3f}")
        print(f"   Recall: {test_recall:.3f}")
        print(f"   F1-Score: {test_f1:.3f}")
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': numeric_features,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print(f"\n🔍 Top 10 Most Important Statistical Features:")
        print(feature_importance.head(10))
        
        # Save model and metrics
        self.feature_names = numeric_features
        self.is_trained = True
        
        metrics = {
            'train_rmse': train_rmse,
            'test_rmse': test_rmse,
            'train_accuracy': train_accuracy,
            'train_precision': train_precision,
            'train_recall': train_recall,
            'train_f1': train_f1,
            'test_accuracy': test_accuracy,
            'test_precision': test_precision,
            'test_recall': test_recall,
            'test_f1': test_f1,
            'y_train_class': y_train_class,
            'y_train_pred_class': y_train_pred_class,
            'y_test_class': y_test_class,
            'y_test_pred_class': y_test_pred_class,
            'statistical_ranges': self.statistical_ranges,
            'mahal_threshold': self.mahal_threshold
        }
        
        joblib.dump(self.model, 'statistical_temporal_model.pkl')
        joblib.dump(self.feature_names, 'statistical_temporal_features.pkl')
        joblib.dump(self.scaler, 'statistical_temporal_scaler.pkl')
        joblib.dump(metrics, 'statistical_temporal_metrics.pkl')
        joblib.dump({
            'mean': self.fresh_baseline_mean,
            'cov': self.fresh_baseline_cov,
            'cov_inv': self.fresh_cov_inv
        }, 'statistical_baseline.pkl')
        
        print("✅ Statistical temporal model training completed!")
        return test_rmse, feature_importance
    
    def predict_statistical_wear_progression(self, data, experiment_id=None):
        """Predict wear progression using statistical model"""
        if self.model is None:
            try:
                self.model = joblib.load('statistical_temporal_model.pkl')
                self.feature_names = joblib.load('statistical_temporal_features.pkl')
                self.scaler = joblib.load('statistical_temporal_scaler.pkl')
                baseline_data = joblib.load('statistical_baseline.pkl')
                self.fresh_baseline_mean = baseline_data['mean']
                self.fresh_baseline_cov = baseline_data['cov']
                self.fresh_cov_inv = baseline_data['cov_inv']
                self.is_trained = True
            except FileNotFoundError:
                raise ValueError("Statistical temporal model not found. Please train the model first.")
        
        # Create temporal features
        if isinstance(data, str):
            data = pd.read_csv(data)
        
        # Auto-detect experiment ID if not provided
        if experiment_id is None:
            import re
            if hasattr(data, 'name') and data.name:
                match = re.search(r'experiment_(\\d+)', str(data.name))
                if match:
                    experiment_id = int(match.group(1))
            if experiment_id is None:
                experiment_id = 999  # Default
        
        temporal_data = self.create_statistical_temporal_features(data, experiment_id)
        
        # Prepare features
        X = temporal_data[self.feature_names].fillna(0)
        X_scaled = self.scaler.transform(X)
        
        # Predict wear progression
        wear_progression = self.model.predict(X_scaled)
        
        # Statistical risk categories
        risk_categories = []
        for wear in wear_progression:
            if wear > 0.8:
                risk_categories.append("🔴 HIGH RISK")
            elif wear > 0.5:
                risk_categories.append("🟡 MEDIUM RISK")
            else:
                risk_categories.append("🟢 LOW RISK")
        
        return wear_progression, risk_categories, temporal_data
    
    def analyze_statistical_health(self, data, wear_progression, risk_categories, temporal_data):
        """Analyze machine health with statistical foundation"""
        
        # Calculate metrics
        avg_wear = np.mean(wear_progression)
        max_wear = np.max(wear_progression)
        final_wear = wear_progression[-1] if len(wear_progression) > 0 else 0
        wear_rate = (wear_progression[-1] - wear_progression[0]) if len(wear_progression) > 1 else 0
        
        # Statistical operation performance assessment
        if avg_wear > 0.8:
            overall_risk = "🔴 POOR PERFORMANCE"
            risk_description = "Tool experienced heavy wear - statistical analysis indicates performance degradation"
        elif avg_wear > 0.6:
            overall_risk = "🟡 MODERATE PERFORMANCE"
            risk_description = "Tool showed moderate wear progression - within statistical expected range"
        else:
            overall_risk = "🟢 GOOD PERFORMANCE"
            risk_description = "Tool maintained low wear levels - excellent statistical performance"
        
        # Mahalanobis-based anomaly detection
        mahal_anomalies = 0
        if 'mahal_anomaly' in temporal_data.columns:
            mahal_anomalies = temporal_data['mahal_anomaly'].sum()
        
        # End-stage warning
        end_stage_warning = None
        if final_wear > 0.8:
            end_stage_warning = {
                'level': 'HIGH',
                'message': f"⚠️ CRITICAL: Statistical analysis shows {final_wear:.1%} wear by end of operation!",
                'recommendation': "Tool replacement required - exceeds statistical safety threshold"
            }
        elif final_wear > 0.65:
            end_stage_warning = {
                'level': 'MEDIUM',
                'message': f"⚠️ WARNING: Statistical analysis shows {final_wear:.1%} wear by end of operation",
                'recommendation': "Monitor closely - approaching statistical wear threshold"
            }
        
        # Statistical issues and recommendations
        issues = []
        recommendations = []
        
        if mahal_anomalies > len(temporal_data) * 0.1:  # >10% anomalies
            issues.append(f"⚠️ Mahalanobis anomalies detected: {mahal_anomalies} samples exceed statistical threshold")
            recommendations.append("Investigate machining parameters - sensor patterns deviate from baseline")
        
        if wear_rate > 0.3:
            issues.append(f"⚠️ High wear progression rate detected: {wear_rate:.3f}")
            recommendations.append("Reduce cutting parameters to control wear rate")
        
        return {
            'overall_risk': overall_risk,
            'risk_description': risk_description,
            'avg_wear_progression': avg_wear,
            'max_wear_progression': max_wear,
            'final_wear_progression': final_wear,
            'wear_rate': wear_rate,
            'end_stage_warning': end_stage_warning,
            'mahalanobis_anomalies': mahal_anomalies,
            'statistical_foundation': True,
            'issues': issues,
            'recommendations': recommendations,
            'wear_progression': wear_progression,
            'risk_categories': risk_categories,
            'temporal_analysis': True
        }

def main():
    print("🧠 Statistical Temporal Wear Progression Predictor")
    print("=" * 55)
    
    predictor = StatisticalTemporalWearPredictor()
    
    # Train the statistical model
    test_rmse, feature_importance = predictor.train_statistical_temporal_model()
    
    print(f"\n🎉 Statistical Temporal Model Training Complete!")
    print(f"📊 Test RMSE: {test_rmse:.3f}")
    print(f"🧠 Statistical foundation: Mahalanobis distance + data-driven ranges")
    print(f"🎯 Ready for statistically-grounded wear progression analysis!")

if __name__ == "__main__":
    main()