import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, mean_squared_error
from sklearn.model_selection import train_test_split
from pathlib import Path
import joblib
import warnings
warnings.filterwarnings('ignore')

class TemporalWearPredictor:
    def __init__(self):
        self.model = None
        self.feature_names = None
        self.is_trained = False
        self.scaler = None
        
        # Base features for temporal analysis
        self.base_features = [
            'X1_CurrentFeedback', 'X1_DCBusVoltage', 'M1_CURRENT_FEEDRATE', 'X1_OutputPower',
            'X1_ActualVelocity', 'X1_OutputCurrent', 'Y1_CurrentFeedback', 'Z1_CurrentFeedback',
            'X1_ActualPosition', 'X1_CommandVelocity', 'S1_CurrentFeedback', 'X1_OutputVoltage'
        ]
        
        # XGBoost parameters optimized for regression (wear progression)
        self.model_params = {
            'objective': 'reg:squarederror',  # Regression for wear percentage
            'n_estimators': 150,
            'max_depth': 4,
            'learning_rate': 0.08,
            'subsample': 0.9,
            'colsample_bytree': 0.8,
            'random_state': 42,
            'eval_metric': 'rmse'
        }
    
    def create_temporal_features(self, data, experiment_id):
        """Create time-based features that capture wear progression"""
        data = data.copy()
        data = data.sort_index()  # Ensure chronological order
        
        # Add time-based features
        data['time_step'] = range(len(data))
        data['time_progress'] = data['time_step'] / (len(data) - 1)  # 0 to 1 progression
        
        # Determine starting wear level based on experiment type
        fresh_tool_experiments = [1, 2, 3, 4, 5, 11, 12, 17]  # Fresh/unworn tools
        worn_tool_experiments = [6, 7, 8, 9, 10, 13, 14, 15, 16, 18]  # Pre-worn tools
        
        if experiment_id in fresh_tool_experiments:
            starting_wear = 0.0  # Fresh tool starts at 0% wear
            ending_wear = 0.6    # Fresh tool reaches moderate wear by end
        elif experiment_id in worn_tool_experiments:
            starting_wear = 0.7  # Pre-worn tool starts at 70% wear
            ending_wear = 1.0    # Pre-worn tool reaches full wear by end
        else:
            # Default case for unknown experiments
            starting_wear = 0.0
            ending_wear = 1.0
        
        # Create realistic wear progression target based on tool starting condition
        wear_range = ending_wear - starting_wear
        data['wear_progression'] = starting_wear + (data['time_progress'] * wear_range)
        
        # Rolling statistics to capture temporal trends
        window_sizes = [10, 25, 50]
        
        for feature in self.base_features:
            if feature in data.columns:
                # Rolling means
                for window in window_sizes:
                    data[f'{feature}_rolling_mean_{window}'] = data[feature].rolling(window=window, min_periods=1).mean()
                    data[f'{feature}_rolling_std_{window}'] = data[feature].rolling(window=window, min_periods=1).std()
                
                # Trend features (slope of recent values)
                data[f'{feature}_trend_10'] = data[feature].rolling(window=10, min_periods=5).apply(
                    lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) >= 2 else 0, raw=False
                )
                
                # Change from beginning
                data[f'{feature}_change_from_start'] = data[feature] - data[feature].iloc[0]
                
                # Cumulative change
                data[f'{feature}_cumulative_change'] = (data[feature] - data[feature].shift(1)).fillna(0).cumsum()
        
        # Machining process duration effect
        data['cumulative_machining_time'] = data['time_step']
        
        # Experiment characteristics
        data['experiment_id'] = experiment_id
        
        return data
    
    def prepare_temporal_data(self, data_dir="data/CNC mill wear /"):
        """Prepare temporal training data from all experiments"""
        print("🕒 Preparing Temporal Wear Progression Data...")
        print("=" * 50)
        
        all_temporal_data = []
        
        # Process each experiment (1-16 for training)
        for exp_id in range(1, 17):
            filename = f"experiment_{exp_id:02d}.csv"
            filepath = Path(data_dir) / filename
            
            if filepath.exists():
                print(f"📁 Processing {filename} for temporal analysis...")
                
                # Load experiment data
                exp_data = pd.read_csv(filepath)
                
                # Create temporal features
                temporal_data = self.create_temporal_features(exp_data, exp_id)
                
                # Sample every N rows to reduce data size but maintain temporal structure
                sample_interval = max(1, len(temporal_data) // 200)  # ~200 samples per experiment
                sampled_data = temporal_data.iloc[::sample_interval].copy()
                
                print(f"   ✅ Processed {len(temporal_data)} → {len(sampled_data)} temporal samples")
                
                all_temporal_data.append(sampled_data)
        
        # Combine all temporal data
        combined_data = pd.concat(all_temporal_data, ignore_index=True)
        
        print(f"\n📊 Temporal Data Summary:")
        print(f"   Total temporal samples: {len(combined_data)}")
        print(f"   Experiments processed: {combined_data['experiment_id'].nunique()}")
        print(f"   Features created: {len(combined_data.columns)}")
        
        return combined_data
    
    def train_temporal_model(self, data_dir="data/CNC mill wear /"):
        """Train XGBoost model on temporal wear progression"""
        print("🎯 Training Temporal Wear Progression Predictor...")
        
        # Prepare temporal data
        temporal_data = self.prepare_temporal_data(data_dir)
        
        # Get feature columns (exclude metadata and target)
        exclude_cols = ['wear_progression', 'experiment_id', 'time_step', 'time_progress']
        feature_cols = [col for col in temporal_data.columns if col not in exclude_cols and not col.startswith('Machining_Process')]
        
        # Remove any remaining non-numeric columns
        numeric_features = []
        for col in feature_cols:
            if temporal_data[col].dtype in ['int64', 'float64']:
                numeric_features.append(col)
        
        print(f"\n🔧 Using {len(numeric_features)} temporal features")
        
        # Prepare features and target
        X = temporal_data[numeric_features].fillna(0)
        y = temporal_data['wear_progression']
        
        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=temporal_data['experiment_id']
        )
        
        print(f"\n📈 Temporal Train/Test Split:")
        print(f"   Training samples: {len(X_train)}")
        print(f"   Test samples: {len(X_test)}")
        
        # Train XGBoost regressor
        print(f"\n🤖 Training Temporal XGBoost Model...")
        self.model = xgb.XGBRegressor(**self.model_params)
        self.model.fit(X_train, y_train)
        
        # Evaluate model
        y_train_pred = self.model.predict(X_train)
        y_test_pred = self.model.predict(X_test)
        
        train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
        test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
        
        # Convert to classification metrics for comparison
        # Threshold: >0.7 = worn, 0.3-0.7 = wearing, <0.3 = unworn
        y_train_class = (y_train > 0.7).astype(int)
        y_train_pred_class = (y_train_pred > 0.7).astype(int)
        y_test_class = (y_test > 0.7).astype(int)
        y_test_pred_class = (y_test_pred > 0.7).astype(int)
        
        # Calculate classification metrics
        train_accuracy = accuracy_score(y_train_class, y_train_pred_class)
        train_precision = precision_score(y_train_class, y_train_pred_class, zero_division=0)
        train_recall = recall_score(y_train_class, y_train_pred_class, zero_division=0)
        train_f1 = f1_score(y_train_class, y_train_pred_class, zero_division=0)
        
        test_accuracy = accuracy_score(y_test_class, y_test_pred_class)
        test_precision = precision_score(y_test_class, y_test_pred_class, zero_division=0)
        test_recall = recall_score(y_test_class, y_test_pred_class, zero_division=0)
        test_f1 = f1_score(y_test_class, y_test_pred_class, zero_division=0)
        
        classification_accuracy = test_accuracy
        
        print(f"\n📊 Temporal Model Performance:")
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
        
        print(f"\n🔍 Top 10 Most Important Temporal Features:")
        print(feature_importance.head(10))
        
        # Save model and metrics
        self.feature_names = numeric_features
        self.is_trained = True
        
        # Save all metrics
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
            'y_test_pred_class': y_test_pred_class
        }
        
        joblib.dump(self.model, 'temporal_wear_model.pkl')
        joblib.dump(self.feature_names, 'temporal_features.pkl')
        joblib.dump(metrics, 'temporal_metrics.pkl')
        
        # Create visualizations
        self.create_temporal_visualizations(y_test, y_test_pred, feature_importance, temporal_data, metrics)
        
        print("✅ Temporal model training completed!")
        return test_rmse, feature_importance
    
    def predict_wear_progression(self, data, experiment_id=None):
        """Predict wear progression for new data"""
        if self.model is None:
            try:
                self.model = joblib.load('temporal_wear_model.pkl')
                self.feature_names = joblib.load('temporal_features.pkl')
                self.is_trained = True
            except FileNotFoundError:
                raise ValueError("Temporal model not found. Please train the model first.")
        
        # Create temporal features for the input data
        if isinstance(data, str):
            data = pd.read_csv(data)
        
        # Try to detect experiment ID from filename if not provided
        if experiment_id is None:
            # Try to extract from common filename patterns
            import re
            if hasattr(data, 'name') and data.name:
                match = re.search(r'experiment_(\d+)', str(data.name))
                if match:
                    experiment_id = int(match.group(1))
            
            # Default for unknown experiments (assume fresh tool)
            if experiment_id is None:
                experiment_id = 999
        
        temporal_data = self.create_temporal_features(data, experiment_id)
        
        # Prepare features
        X = temporal_data[self.feature_names].fillna(0)
        
        # Predict wear progression
        wear_progression = self.model.predict(X)
        
        # Convert to risk categories
        risk_categories = []
        for wear in wear_progression:
            if wear > 0.8:
                risk_categories.append("🔴 HIGH RISK")
            elif wear > 0.5:
                risk_categories.append("🟡 MEDIUM RISK")
            else:
                risk_categories.append("🟢 LOW RISK")
        
        return wear_progression, risk_categories, temporal_data
    
    def analyze_temporal_health(self, data, wear_progression, risk_categories, temporal_data):
        """Analyze machine health based on temporal wear progression"""
        
        # Calculate temporal metrics
        avg_wear = np.mean(wear_progression)
        max_wear = np.max(wear_progression)
        final_wear = wear_progression[-1] if len(wear_progression) > 0 else 0
        wear_rate = (wear_progression[-1] - wear_progression[0]) if len(wear_progression) > 1 else 0
        
        # Determine overall operation performance (based on average)
        if avg_wear > 0.8:
            overall_risk = "🔴 POOR PERFORMANCE"
            risk_description = "Tool experienced heavy wear throughout operation - performance degraded significantly"
        elif avg_wear > 0.5:
            overall_risk = "🟡 MODERATE PERFORMANCE"
            risk_description = "Tool showed moderate wear progression during operation - acceptable performance"
        else:
            overall_risk = "🟢 GOOD PERFORMANCE"
            risk_description = "Tool maintained low wear levels throughout most of operation - good performance"
        
        # Check for end-stage high wear warning
        end_stage_warning = None
        if final_wear > 0.8:
            end_stage_warning = {
                'level': 'HIGH',
                'message': f"⚠️ CRITICAL: Tool reached {final_wear:.1%} wear by end of operation!",
                'recommendation': "Tool replacement required before next operation"
            }
        elif final_wear > 0.6:
            end_stage_warning = {
                'level': 'MEDIUM',
                'message': f"⚠️ WARNING: Tool reached {final_wear:.1%} wear by end of operation",
                'recommendation': "Monitor closely and plan replacement soon"
            }
        
        # Analyze wear trends
        issues = []
        recommendations = []
        
        if wear_rate > 0.5:
            issues.append(f"⚠️ Rapid wear progression detected (rate: {wear_rate:.3f})")
            recommendations.append("Reduce cutting parameters to slow wear progression")
        
        if max_wear > 0.9:
            issues.append(f"⚠️ Critical wear level reached ({max_wear:.1%})")
            recommendations.append("Schedule immediate tool replacement")
        
        # Check for temporal patterns in base features
        if 'M1_CURRENT_FEEDRATE_trend_10' in temporal_data.columns:
            feedrate_trend = temporal_data['M1_CURRENT_FEEDRATE_trend_10'].mean()
            if abs(feedrate_trend) > 1.0:
                issues.append(f"⚠️ Feedrate instability detected (trend: {feedrate_trend:.3f})")
                recommendations.append("Stabilize feedrate for consistent tool wear")
        
        return {
            'overall_risk': overall_risk,
            'risk_description': risk_description,
            'avg_wear_progression': avg_wear,
            'max_wear_progression': max_wear,
            'final_wear_progression': final_wear,
            'wear_rate': wear_rate,
            'end_stage_warning': end_stage_warning,
            'issues': issues,
            'recommendations': recommendations,
            'wear_progression': wear_progression,
            'risk_categories': risk_categories,
            'temporal_analysis': True
        }
    
    def create_temporal_visualizations(self, y_test, y_test_pred, feature_importance, temporal_data, metrics):
        """Create temporal analysis visualizations"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # 1. Prediction vs Actual
        axes[0,0].scatter(y_test, y_test_pred, alpha=0.6)
        axes[0,0].plot([0, 1], [0, 1], 'r--', lw=2)
        axes[0,0].set_xlabel('Actual Wear Progression')
        axes[0,0].set_ylabel('Predicted Wear Progression')
        axes[0,0].set_title('Temporal Model: Prediction vs Actual')
        axes[0,0].grid(True, alpha=0.3)
        
        # 2. Feature importance (top 10)
        top_features = feature_importance.head(10)
        axes[0,1].barh(range(len(top_features)), top_features['importance'])
        axes[0,1].set_yticks(range(len(top_features)))
        axes[0,1].set_yticklabels(top_features['feature'], fontsize=8)
        axes[0,1].set_xlabel('Importance')
        axes[0,1].set_title('Top 10 Temporal Features')
        axes[0,1].invert_yaxis()
        
        # 3. Training Confusion Matrix
        train_cm = confusion_matrix(metrics['y_train_class'], metrics['y_train_pred_class'])
        sns.heatmap(train_cm, annot=True, fmt='d', cmap='Blues', ax=axes[0,2])
        axes[0,2].set_title('Training Confusion Matrix')
        axes[0,2].set_xlabel('Predicted')
        axes[0,2].set_ylabel('Actual')
        axes[0,2].set_xticklabels(['Low Wear', 'High Wear'])
        axes[0,2].set_yticklabels(['Low Wear', 'High Wear'])
        
        # 4. Test Confusion Matrix
        test_cm = confusion_matrix(metrics['y_test_class'], metrics['y_test_pred_class'])
        sns.heatmap(test_cm, annot=True, fmt='d', cmap='Oranges', ax=axes[1,0])
        axes[1,0].set_title('Test Confusion Matrix')
        axes[1,0].set_xlabel('Predicted')
        axes[1,0].set_ylabel('Actual')
        axes[1,0].set_xticklabels(['Low Wear', 'High Wear'])
        axes[1,0].set_yticklabels(['Low Wear', 'High Wear'])
        
        # 5. Performance Metrics Comparison
        train_metrics = [metrics['train_accuracy'], metrics['train_precision'], metrics['train_recall'], metrics['train_f1']]
        test_metrics = [metrics['test_accuracy'], metrics['test_precision'], metrics['test_recall'], metrics['test_f1']]
        metric_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        
        x_pos = np.arange(len(metric_names))
        width = 0.35
        
        axes[1,1].bar(x_pos - width/2, train_metrics, width, label='Training', color='skyblue')
        axes[1,1].bar(x_pos + width/2, test_metrics, width, label='Test', color='lightcoral')
        axes[1,1].set_xlabel('Metrics')
        axes[1,1].set_ylabel('Score')
        axes[1,1].set_title('Classification Performance Metrics')
        axes[1,1].set_xticks(x_pos)
        axes[1,1].set_xticklabels(metric_names, rotation=45)
        axes[1,1].legend()
        axes[1,1].set_ylim(0, 1)
        axes[1,1].grid(True, alpha=0.3)
        
        # 6. Model summary
        summary_text = f"""
        🕒 Temporal Wear Progression Model
        
        📊 RMSE Performance:
        • Training: {metrics['train_rmse']:.3f}
        • Test: {metrics['test_rmse']:.3f}
        
        🎯 Classification (>0.7 threshold):
        • Test Accuracy: {metrics['test_accuracy']:.3f}
        • Test Precision: {metrics['test_precision']:.3f}
        • Test Recall: {metrics['test_recall']:.3f}
        • Test F1-Score: {metrics['test_f1']:.3f}
        
        🔧 Features: {len(feature_importance)} temporal
        🕒 Approach: Realistic wear progression
        
        🛠️ Tool Types:
        • Fresh: 0% → 60% wear progression
        • Worn: 70% → 100% wear progression
        """
        
        axes[1,2].text(0.1, 0.5, summary_text, transform=axes[1,2].transAxes,
                      fontsize=10, verticalalignment='center',
                      bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcyan"))
        axes[1,2].set_title('Model Performance Summary')
        axes[1,2].axis('off')
        
        plt.tight_layout()
        plt.savefig('temporal_wear_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("📊 Temporal analysis visualization saved as 'temporal_wear_analysis.png'")

def main():
    """Train the temporal wear progression model"""
    print("🕒 Temporal Wear Progression Predictor")
    print("=" * 60)
    
    predictor = TemporalWearPredictor()
    test_rmse, feature_importance = predictor.train_temporal_model()
    
    print(f"\n🎉 Temporal Model Training Complete!")
    print(f"📊 Test RMSE: {test_rmse:.3f}")
    print(f"🎯 Ready for temporal wear progression analysis!")

if __name__ == "__main__":
    main()