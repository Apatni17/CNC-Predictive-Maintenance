import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from pathlib import Path
import joblib
import random

class OptimizedXGBoostPredictor:
    def __init__(self):
        self.model = None
        self.feature_names = None
        self.is_trained = False
        self.training_files = []
        self.test_accuracy = None
        self.validation_files = ['experiment_17.csv', 'experiment_18.csv']
        
        # Optimized risk thresholds from Step 4
        self.medium_risk_threshold = 0.5
        self.high_risk_threshold = 0.8
        
        # Best features identified
        self.best_features = [
            'X1_CurrentFeedback', 'X1_DCBusVoltage', 'M1_CURRENT_FEEDRATE', 'X1_OutputPower',
            'X1_ActualVelocity', 'X1_OutputCurrent', 'Y1_CurrentFeedback', 'Z1_CurrentFeedback',
            'X1_ActualPosition', 'X1_CommandVelocity', 'S1_CurrentFeedback', 'X1_OutputVoltage'
        ]
        
        # Optimized hyperparameters from Step 4
        self.model_params = {
            'objective': 'binary:logistic',
            'n_estimators': 100,
            'max_depth': 3,
            'learning_rate': 0.05,
            'subsample': 1.0,
            'colsample_bytree': 0.8,
            'random_state': 42,
            'eval_metric': 'logloss'
        }
        
    def train_model(self, data_dir="data/CNC mill wear /"):
        """Train the optimized XGBoost model"""
        print("🎯 Training Optimized XGBoost Tool Wear Predictor...")
        
        # Training files: 1-16 (experiment-based split)
        self.training_files = [f"experiment_{i:02d}.csv" for i in range(1, 17)]
        
        print(f"📁 Training on files: {self.training_files}")
        print(f"🔍 Validation files (reserved): {self.validation_files}")
        
        # Load and prepare training data
        training_data = []
        
        for filename in self.training_files:
            filepath = f"{data_dir}{filename}"
            try:
                data = pd.read_csv(filepath)
                exp_num = int(filename.split('_')[1].split('.')[0])
                data['experiment_id'] = exp_num
                data['tool_wear'] = 1 if exp_num > 5 else 0  # 0=unworn (1-5), 1=worn (6-16)
                print(f"✅ Loaded {filename} (Experiment {exp_num})")
                training_data.append(data)
            except FileNotFoundError:
                print(f"❌ Warning: {filepath} not found")
                continue
        
        if not training_data:
            raise ValueError("No training data files found!")
        
        # Combine all data
        combined_data = pd.concat(training_data, ignore_index=True)
        print(f"📊 Combined data shape: {combined_data.shape}")
        
        # Get available features
        self.available_features = [f for f in self.best_features if f in combined_data.columns]
        print(f"🔧 Using {len(self.available_features)} best features for prediction")
        
        # Experiment-based train/test split (same as Step 4)
        random.seed(42)
        unworn_experiments = list(range(1, 6))
        worn_experiments = list(range(6, 17))
        
        n_unworn_train = int(len(unworn_experiments) * 0.7)
        unworn_train_exp = random.sample(unworn_experiments, n_unworn_train)
        unworn_test_exp = [exp for exp in unworn_experiments if exp not in unworn_train_exp]
        
        n_worn_train = int(len(worn_experiments) * 0.7)
        worn_train_exp = random.sample(worn_experiments, n_worn_train)
        worn_test_exp = [exp for exp in worn_experiments if exp not in worn_train_exp]
        
        train_experiments = unworn_train_exp + worn_train_exp
        test_experiments = unworn_test_exp + worn_test_exp
        
        train_data = combined_data[combined_data['experiment_id'].isin(train_experiments)]
        test_data = combined_data[combined_data['experiment_id'].isin(test_experiments)]
        
        print(f"\n📊 Training Data Summary:")
        print(f"   Total samples: {len(combined_data)}")
        print(f"   Training samples: {len(train_data)}")
        print(f"   Test samples: {len(test_data)}")
        print(f"   Training experiments: {train_experiments}")
        print(f"   Test experiments: {test_experiments}")
        
        # Prepare features and labels
        X_train = train_data[self.available_features].fillna(0)
        y_train = train_data['tool_wear']
        X_test = test_data[self.available_features].fillna(0)
        y_test = test_data['tool_wear']
        
        print(f"\n📈 Train/Test Split:")
        print(f"   Training samples: {len(X_train)} (Unworn: {sum(y_train == 0)}, Worn: {sum(y_train == 1)})")
        print(f"   Test samples: {len(X_test)} (Unworn: {sum(y_test == 0)}, Worn: {sum(y_test == 1)})")
        
        # Train optimized XGBoost model
        print(f"\n🤖 Training Optimized XGBoost Model...")
        print(f"   Parameters: {self.model_params}")
        
        self.model = xgb.XGBClassifier(**self.model_params)
        self.model.fit(X_train, y_train)
        
        # Evaluate model performance
        y_test_pred = self.model.predict(X_test)
        y_test_proba = self.model.predict_proba(X_test)[:, 1]
        
        self.test_accuracy = accuracy_score(y_test, y_test_pred)
        test_precision = precision_score(y_test, y_test_pred)
        test_recall = recall_score(y_test, y_test_pred)
        test_f1 = f1_score(y_test, y_test_pred)
        
        print(f"\n📈 Test Set Performance:")
        print(f"   Accuracy: {self.test_accuracy:.3f}")
        print(f"   Precision: {test_precision:.3f}")
        print(f"   Recall: {test_recall:.3f}")
        print(f"   F1-Score: {test_f1:.3f}")
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': self.available_features,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print(f"\n🔍 Top 5 Most Important Features:")
        print(feature_importance.head())
        
        # Risk threshold analysis
        test_risk_categories = self.categorize_risk_batch(y_test_proba)
        risk_dist = pd.Series(test_risk_categories).value_counts()
        
        print(f"\n🎯 Risk Category Distribution (Optimized Thresholds):")
        for risk, count in risk_dist.items():
            print(f"   {risk}: {count} ({count/len(test_risk_categories)*100:.1f}%)")
        
        # Save model and components
        joblib.dump(self.model, 'optimized_xgboost_model.pkl')
        joblib.dump(self.available_features, 'optimized_xgboost_features.pkl')
        joblib.dump(self.training_files, 'optimized_training_files.pkl')
        joblib.dump(self.test_accuracy, 'optimized_test_accuracy.pkl')
        joblib.dump({
            'medium_threshold': self.medium_risk_threshold,
            'high_threshold': self.high_risk_threshold
        }, 'optimized_risk_thresholds.pkl')
        
        self.is_trained = True
        
        # Create performance visualization
        self.create_performance_plots(y_test, y_test_pred, y_test_proba, feature_importance)
        
        print("✅ Optimized XGBoost model training completed!")
        print(f"🎯 Ready for validation with experiments 17-18!")
        return self.test_accuracy, feature_importance
    
    def categorize_risk_batch(self, probabilities):
        """Categorize risk for batch of probabilities using optimized thresholds"""
        risk_categories = []
        for prob in probabilities:
            if prob >= self.high_risk_threshold:
                risk_categories.append("High Risk")
            elif prob >= self.medium_risk_threshold:
                risk_categories.append("Medium Risk")
            else:
                risk_categories.append("Low Risk")
        return risk_categories
    
    def categorize_risk(self, probability):
        """Categorize risk for a single probability using optimized thresholds"""
        if probability >= self.high_risk_threshold:
            return "🔴 HIGH RISK"
        elif probability >= self.medium_risk_threshold:
            return "🟡 MEDIUM RISK"
        else:
            return "🟢 LOW RISK"
    
    def predict_tool_wear(self, data):
        """Predict tool wear risk for new machine data"""
        # Load model if not already loaded
        if self.model is None:
            try:
                self.model = joblib.load('optimized_xgboost_model.pkl')
                self.available_features = joblib.load('optimized_xgboost_features.pkl')
                self.test_accuracy = joblib.load('optimized_test_accuracy.pkl')
                thresholds = joblib.load('optimized_risk_thresholds.pkl')
                self.medium_risk_threshold = thresholds['medium_threshold']
                self.high_risk_threshold = thresholds['high_threshold']
                self.is_trained = True
            except FileNotFoundError:
                raise ValueError("Optimized XGBoost model files not found. Please train the model first.")
        
        if not self.is_trained:
            raise ValueError("Model not trained. Please train the model first.")
        
        # Prepare input data
        if isinstance(data, str):
            data = pd.read_csv(data)
        
        # Select available features
        available_data = data[self.available_features].fillna(0)
        
        # Make predictions
        wear_probabilities = self.model.predict_proba(available_data)[:, 1]
        wear_predictions = self.model.predict(available_data)
        
        return wear_probabilities, wear_predictions
    
    def analyze_machine_health(self, data, wear_probabilities):
        """Analyze machine health using optimized risk assessment"""
        if isinstance(data, str):
            data = pd.read_csv(data)
        
        # Calculate key metrics
        avg_wear_prob = np.mean(wear_probabilities)
        max_wear_prob = np.max(wear_probabilities)
        risk_std = np.std(wear_probabilities)
        
        # Determine risk level using optimized thresholds
        risk_level = self.categorize_risk(avg_wear_prob)
        
        # Risk descriptions
        if avg_wear_prob >= self.high_risk_threshold:
            risk_description = "Tool shows high wear probability - consider replacement soon"
        elif avg_wear_prob >= self.medium_risk_threshold:
            risk_description = "Tool shows moderate wear indicators - monitor closely and plan maintenance"
        else:
            risk_description = "Tool appears to be in good condition - continue normal operation"
        
        # Analyze operational indicators
        issues = []
        recommendations = []
        
        # Enhanced operational analysis with better thresholds
        if 'M1_CURRENT_FEEDRATE' in data.columns:
            feedrate = data['M1_CURRENT_FEEDRATE'].mean()
            if feedrate > 30:  # Adjusted threshold
                issues.append(f"⚠️ High feedrate detected ({feedrate:.1f})")
                recommendations.append("Consider reducing feedrate for better tool life")
        
        if 'X1_OutputCurrent' in data.columns:
            current = data['X1_OutputCurrent'].mean()
            if current > 328:  # Adjusted threshold based on analysis
                issues.append(f"⚠️ High output current detected ({current:.1f})")
                recommendations.append("Monitor power consumption - may indicate increased cutting resistance")
        
        if 'X1_ActualPosition' in data.columns:
            position_std = data['X1_ActualPosition'].std()
            if position_std > 50:  # Position instability
                issues.append(f"⚠️ Position instability detected (std: {position_std:.2f})")
                recommendations.append("Check machine rigidity and positioning accuracy")
        
        # Model confidence assessment
        model_confidence_issue = False
        if avg_wear_prob >= self.high_risk_threshold and len(issues) == 0:
            model_confidence_issue = True
            recommendations.append("⚠️ High wear prediction with no operational issues - recommend manual inspection")
        
        return {
            'risk_level': risk_level,
            'risk_description': risk_description,
            'avg_wear_probability': avg_wear_prob,
            'max_wear_probability': max_wear_prob,
            'risk_stability': 'Stable' if risk_std < 0.1 else 'Variable',
            'issues': issues,
            'recommendations': recommendations,
            'wear_probabilities': wear_probabilities,
            'model_confidence_issue': model_confidence_issue,
            'thresholds': {
                'medium': self.medium_risk_threshold,
                'high': self.high_risk_threshold
            }
        }
    
    def create_performance_plots(self, y_test, y_test_pred, y_test_proba, feature_importance):
        """Create performance visualization plots"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Confusion Matrix
        cm = confusion_matrix(y_test, y_test_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0,0])
        axes[0,0].set_title('Confusion Matrix (Optimized XGBoost)')
        axes[0,0].set_xlabel('Predicted')
        axes[0,0].set_ylabel('Actual')
        
        # 2. Feature Importance
        top_8_features = feature_importance.head(8)
        axes[0,1].barh(range(len(top_8_features)), top_8_features['importance'])
        axes[0,1].set_yticks(range(len(top_8_features)))
        axes[0,1].set_yticklabels(top_8_features['feature'])
        axes[0,1].set_xlabel('Feature Importance')
        axes[0,1].set_title('Feature Importance (Optimized XGBoost)')
        axes[0,1].invert_yaxis()
        
        # 3. Risk Distribution with Optimized Thresholds
        risk_categories = self.categorize_risk_batch(y_test_proba)
        risk_dist = pd.Series(risk_categories).value_counts()
        
        axes[1,0].pie(risk_dist.values, labels=risk_dist.index, autopct='%1.1f%%',
                     colors=['lightcoral', 'gold', 'lightgreen'])
        axes[1,0].set_title('Risk Distribution (Optimized Thresholds)')
        
        # 4. Model Summary
        summary_text = f"""
        🎯 Optimized XGBoost Predictor
        
        📊 Model Performance:
        • Test Accuracy: {self.test_accuracy:.3f}
        • Optimization: F1-Score focused
        
        🎚️ Optimized Risk Thresholds:
        • Medium Risk: ≥ {self.medium_risk_threshold}
        • High Risk: ≥ {self.high_risk_threshold}
        
        🔧 Key Features:
        • Feedrate (Primary)
        • Output Current
        • Position Accuracy
        
        🎯 Ready for Real-World Testing
        """
        
        axes[1,1].text(0.1, 0.5, summary_text, transform=axes[1,1].transAxes, 
                      fontsize=10, verticalalignment='center',
                      bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
        axes[1,1].set_title('Optimized Model Summary')
        axes[1,1].axis('off')
        
        plt.tight_layout()
        plt.savefig('optimized_xgboost_performance.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("📊 Performance plots saved as 'optimized_xgboost_performance.png'")

def main():
    """Train the optimized XGBoost model"""
    print("🎯 Optimized XGBoost Tool Wear Predictor")
    print("=" * 50)
    
    # Create predictor
    predictor = OptimizedXGBoostPredictor()
    
    # Train model
    test_accuracy, feature_importance = predictor.train_model()
    
    print(f"\n🎉 Optimized XGBoost Training Complete!")
    print(f"📊 Test accuracy: {test_accuracy:.3f}")
    print(f"🎯 Ready for real-world testing with experiments 17-18!")

if __name__ == "__main__":
    main()