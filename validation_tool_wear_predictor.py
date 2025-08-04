import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib
import warnings
warnings.filterwarnings('ignore')

class ValidationToolWearPredictor:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = None
        self.is_trained = False
        self.training_files = []
        self.test_accuracy = None
        self.validation_files = ['experiment_17.csv', 'experiment_18.csv']
        
    def train_model(self, data_dir="data/CNC mill wear /"):
        """Train the model on experiments 1-16 with 80/20 train/test split"""
        print("🎯 Training Validation Tool Wear Predictor...")
        
        # Training files: 1-16 (will be split 80/20)
        self.training_files = [f"experiment_{i:02d}.csv" for i in range(1, 17)]
        
        print(f"📁 Training on files: {self.training_files}")
        print(f"🔍 Validation files (reserved): {self.validation_files}")
        
        # Load and prepare training data
        all_data = []
        
        for filename in self.training_files:
            filepath = f"{data_dir}{filename}"
            try:
                data = pd.read_csv(filepath)
                # Extract experiment number from filename
                exp_num = int(filename.split('_')[1].split('.')[0])
                data['experiment_id'] = exp_num
                print(f"✅ Loaded {filename} (Experiment {exp_num})")
                all_data.append(data)
            except FileNotFoundError:
                print(f"❌ Warning: {filepath} not found")
                continue
        
        if not all_data:
            raise ValueError("No training data files found!")
        
        # Combine all data
        combined_data = pd.concat(all_data, ignore_index=True)
        print(f"📊 Combined data shape: {combined_data.shape}")
        
        # Sample 300 data points from each experiment
        sampled_data = []
        for exp_id in combined_data['experiment_id'].unique():
            exp_data = combined_data[combined_data['experiment_id'] == exp_id]
            if len(exp_data) >= 300:
                sampled = exp_data.sample(n=300, random_state=42)
            else:
                sampled = exp_data
            sampled_data.append(sampled)
            print(f"📈 Experiment {exp_id}: {len(sampled)} samples")
        
        self.training_data = pd.concat(sampled_data, ignore_index=True)
        
        # Create tool wear labels (1-5 unworn, 6-16 worn)
        self.training_data['tool_wear'] = (
            self.training_data['experiment_id'].apply(lambda x: 0 if x <= 5 else 1)
        )
        
        # Select key features that affect tool wear (based on our analysis)
        self.feature_names = [
            'X1_CurrentFeedback',      # Cutting forces - most important
            'X1_DCBusVoltage',         # Power supply stability
            'M1_CURRENT_FEEDRATE',     # Feedrate - major factor
            'X1_OutputPower',          # Power consumption
            'X1_ActualVelocity',       # Velocity stability
            'X1_OutputCurrent',        # Current draw
            'Y1_CurrentFeedback',      # Y-axis forces
            'Z1_CurrentFeedback',      # Z-axis forces
            'X1_ActualPosition',       # Position accuracy
            'X1_CommandVelocity',      # Command vs actual velocity
            'S1_CurrentFeedback',      # Spindle forces
            'X1_OutputVoltage'         # Voltage output
        ]
        
        # Filter to available features
        available_features = [f for f in self.feature_names if f in self.training_data.columns]
        print(f"🔧 Using {len(available_features)} key features for prediction")
        
        # Prepare training data
        X = self.training_data[available_features].fillna(0)
        y = self.training_data['tool_wear']
        
        print(f"\n📊 Training Data Summary:")
        print(f"   Total samples: {len(X)}")
        print(f"   Unworn tools: {sum(y == 0)}")
        print(f"   Worn tools: {sum(y == 1)}")
        print(f"   Features: {len(available_features)}")
        
        # Split into train and test (80/20)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"\n📈 Train/Test Split:")
        print(f"   Training samples: {len(X_train)}")
        print(f"   Test samples: {len(X_test)}")
        print(f"   Training unworn: {sum(y_train == 0)}")
        print(f"   Training worn: {sum(y_train == 1)}")
        print(f"   Test unworn: {sum(y_test == 0)}")
        print(f"   Test worn: {sum(y_test == 1)}")
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train Random Forest model
        print("\n🤖 Training Random Forest Model...")
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluate model performance on test set
        y_test_pred = self.model.predict(X_test_scaled)
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
            'feature': available_features,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print(f"\n🔍 Top 5 Most Important Features:")
        print(feature_importance.head())
        
        # Save model and components
        joblib.dump(self.model, 'validation_tool_wear_model.pkl')
        joblib.dump(self.scaler, 'validation_tool_wear_scaler.pkl')
        joblib.dump(available_features, 'validation_tool_wear_features.pkl')
        joblib.dump(self.training_files, 'validation_training_files.pkl')
        joblib.dump(self.test_accuracy, 'validation_test_accuracy.pkl')
        
        self.is_trained = True
        self.available_features = available_features
        
        # Create performance visualization
        self.create_performance_plots(y_train, y_test, y_test_pred, feature_importance)
        
        print("✅ Model training completed!")
        print(f"🎯 Ready for validation with experiments 17-18!")
        return self.test_accuracy, feature_importance
    
    def create_performance_plots(self, y_train, y_test, y_test_pred, feature_importance):
        """Create performance visualization plots"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # 1. Confusion Matrix (Test Set)
        cm = confusion_matrix(y_test, y_test_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0,0])
        axes[0,0].set_title('Confusion Matrix (Test Set)')
        axes[0,0].set_xlabel('Predicted')
        axes[0,0].set_ylabel('Actual')
        
        # 2. Feature Importance
        top_10_features = feature_importance.head(10)
        axes[0,1].barh(range(len(top_10_features)), top_10_features['importance'])
        axes[0,1].set_yticks(range(len(top_10_features)))
        axes[0,1].set_yticklabels(top_10_features['feature'])
        axes[0,1].set_xlabel('Feature Importance')
        axes[0,1].set_title('Top 10 Feature Importance')
        axes[0,1].invert_yaxis()
        
        # 3. Training Data Distribution
        axes[0,2].pie([sum(y_train == 0), sum(y_train == 1)], 
                     labels=['Unworn Tools', 'Worn Tools'], 
                     autopct='%1.1f%%', colors=['lightblue', 'lightcoral'])
        axes[0,2].set_title('Training Data Distribution')
        
        # 4. Test Data Distribution
        axes[1,0].pie([sum(y_test == 0), sum(y_test == 1)], 
                     labels=['Unworn Tools', 'Worn Tools'], 
                     autopct='%1.1f%%', colors=['lightblue', 'lightcoral'])
        axes[1,0].set_title('Test Data Distribution')
        
        # 5. Model Summary
        summary_text = f"""
        🎯 Validation Tool Wear Predictor
        
        📁 Training Files: {len(self.training_files)} (1-16)
        🔍 Validation Files: {len(self.validation_files)} (17-18)
        📊 Training Samples: {len(y_train)}
        🧪 Test Samples: {len(y_test)}
        🔧 Features Used: {len(self.available_features)}
        
        📈 Test Performance:
        • Accuracy: {self.test_accuracy:.3f}
        • Precision: {precision_score(y_test, y_test_pred):.3f}
        • Recall: {recall_score(y_test, y_test_pred):.3f}
        • F1-Score: {f1_score(y_test, y_test_pred):.3f}
        
        🎯 Ready for True Validation
        """
        
        axes[1,1].text(0.1, 0.5, summary_text, transform=axes[1,1].transAxes, 
                      fontsize=10, verticalalignment='center',
                      bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
        axes[1,1].set_title('Model Summary')
        axes[1,1].axis('off')
        
        # 6. Train/Test Split Visualization
        split_data = [len(y_train), len(y_test)]
        axes[1,2].pie(split_data, labels=['Training (80%)', 'Test (20%)'], 
                     autopct='%1.1f%%', colors=['lightgreen', 'lightyellow'])
        axes[1,2].set_title('Train/Test Split')
        
        plt.tight_layout()
        plt.savefig('validation_model_performance.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("📊 Performance plots saved as 'validation_model_performance.png'")
    
    def predict_tool_wear(self, data):
        """Predict tool wear risk for new machine data"""
        # Load model if not already loaded
        if self.model is None:
            try:
                self.model = joblib.load('validation_tool_wear_model.pkl')
                self.scaler = joblib.load('validation_tool_wear_scaler.pkl')
                self.available_features = joblib.load('validation_tool_wear_features.pkl')
                self.test_accuracy = joblib.load('validation_test_accuracy.pkl')
                self.is_trained = True
            except FileNotFoundError:
                raise ValueError("Validation model files not found. Please train the model first.")
        
        if not self.is_trained:
            raise ValueError("Model not trained. Please train the model first.")
        
        # Prepare input data
        if isinstance(data, str):
            data = pd.read_csv(data)
        
        # Select available features
        available_data = data[self.available_features].fillna(0)
        
        # Scale features
        data_scaled = self.scaler.transform(available_data)
        
        # Make predictions
        wear_probabilities = self.model.predict_proba(data_scaled)[:, 1]
        wear_predictions = self.model.predict(data_scaled)
        
        return wear_probabilities, wear_predictions
    
    def analyze_machine_health(self, data, wear_probabilities):
        """Analyze machine health and provide prevention advice"""
        if isinstance(data, str):
            data = pd.read_csv(data)
        
        # Calculate key metrics
        avg_wear_prob = np.mean(wear_probabilities)
        max_wear_prob = np.max(wear_probabilities)
        risk_std = np.std(wear_probabilities)
        
        # Determine risk level
        if avg_wear_prob > 0.7:
            risk_level = "🔴 HIGH RISK"
            risk_description = "Tool shows significant wear indicators - immediate attention required"
        elif avg_wear_prob > 0.4:
            risk_level = "🟡 MEDIUM RISK"
            risk_description = "Tool shows moderate wear indicators - preventive action recommended"
        else:
            risk_level = "🟢 LOW RISK"
            risk_description = "Tool appears to be in good condition - continue monitoring"
        
        # Analyze key indicators
        issues = []
        recommendations = []
        
        # Check cutting forces
        if 'X1_CurrentFeedback' in data.columns:
            current_feedback = data['X1_CurrentFeedback'].mean()
            if current_feedback > -0.3:
                issues.append(f"⚠️ High cutting forces detected ({current_feedback:.3f})")
                recommendations.append("Reduce feedrate by 15-20% to decrease cutting forces")
                recommendations.append("Check tool sharpness and consider tool replacement")
        
        # Check power supply
        if 'X1_DCBusVoltage' in data.columns:
            voltage = data['X1_DCBusVoltage'].mean()
            if voltage < 0.03:
                issues.append(f"⚠️ Low voltage detected ({voltage:.3f})")
                recommendations.append("Check electrical connections and power supply")
                recommendations.append("Monitor voltage stability during operation")
        
        # Check feedrate
        if 'M1_CURRENT_FEEDRATE' in data.columns:
            feedrate = data['M1_CURRENT_FEEDRATE'].mean()
            if feedrate > 25:
                issues.append(f"⚠️ High feedrate detected ({feedrate:.1f})")
                recommendations.append("Optimize feedrate for your material and tool")
                recommendations.append("Consult tool manufacturer guidelines")
        
        # Check velocity stability
        if 'X1_ActualVelocity' in data.columns:
            velocity_std = data['X1_ActualVelocity'].std()
            if velocity_std > 6:
                issues.append(f"⚠️ Velocity instability detected (std: {velocity_std:.2f})")
                recommendations.append("Check for mechanical issues or tool chatter")
                recommendations.append("Verify machine rigidity and alignment")
        
        # Check power consumption
        if 'X1_OutputPower' in data.columns:
            power = data['X1_OutputPower'].mean()
            if power > 0.006:
                issues.append(f"⚠️ High power consumption ({power:.6f})")
                recommendations.append("Optimize cutting parameters for efficiency")
                recommendations.append("Check tool geometry and material compatibility")
        
        # Generate prevention strategies
        prevention_strategies = []
        
        if avg_wear_prob > 0.6:
            prevention_strategies.append({
                'priority': 'Immediate',
                'action': 'Schedule tool replacement within 24-48 hours',
                'reason': 'High wear probability indicates imminent tool failure'
            })
        
        if len(issues) > 2:
            prevention_strategies.append({
                'priority': 'High',
                'action': 'Review and optimize all cutting parameters',
                'reason': 'Multiple issues detected - comprehensive optimization needed'
            })
        
        if avg_wear_prob < 0.3 and len(issues) == 0:
            prevention_strategies.append({
                'priority': 'Low',
                'action': 'Continue current maintenance schedule',
                'reason': 'Tool appears to be operating within normal parameters'
            })
        
        # Add general maintenance recommendations
        general_recommendations = [
            "Monitor tool wear indicators daily",
            "Keep detailed records of cutting parameters",
            "Implement predictive maintenance schedule",
            "Train operators on early warning signs",
            "Maintain clean and well-lubricated machine components"
        ]
        
        return {
            'risk_level': risk_level,
            'risk_description': risk_description,
            'avg_wear_probability': avg_wear_prob,
            'max_wear_probability': max_wear_prob,
            'risk_stability': 'Stable' if risk_std < 0.1 else 'Variable',
            'issues': issues,
            'recommendations': recommendations,
            'prevention_strategies': prevention_strategies,
            'general_recommendations': general_recommendations,
            'wear_probabilities': wear_probabilities
        }
    
    def create_validation_report(self, data, wear_probabilities, analysis_results, machine_name):
        """Create a validation-focused prediction report"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. Risk Assessment
        risk_text = f"""
        🏭 {machine_name}
        
        📊 Risk Assessment:
        • Risk Level: {analysis_results['risk_level']}
        • Average Risk: {analysis_results['avg_wear_probability']:.1%}
        • Peak Risk: {analysis_results['max_wear_probability']:.1%}
        • Risk Stability: {analysis_results['risk_stability']}
        
        ⚠️ Issues Found: {len(analysis_results['issues'])}
        💡 Recommendations: {len(analysis_results['recommendations'])}
        
        🎯 Model Test Accuracy: {self.test_accuracy:.1%}
        """
        
        axes[0,0].text(0.1, 0.5, risk_text, transform=axes[0,0].transAxes, 
                      fontsize=11, verticalalignment='center',
                      bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
        axes[0,0].set_title('Risk Assessment')
        axes[0,0].axis('off')
        
        # 2. Risk Timeline
        axes[0,1].plot(wear_probabilities, alpha=0.7, color='red', linewidth=2)
        axes[0,1].axhline(y=0.7, color='red', linestyle='--', label='High Risk')
        axes[0,1].axhline(y=0.4, color='orange', linestyle='--', label='Medium Risk')
        axes[0,1].set_xlabel('Data Point')
        axes[0,1].set_ylabel('Wear Probability')
        axes[0,1].set_title('Tool Wear Risk Timeline')
        axes[0,1].legend()
        axes[0,1].grid(True, alpha=0.3)
        
        # 3. Risk Distribution
        axes[1,0].hist(wear_probabilities, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        axes[1,0].axvline(np.mean(wear_probabilities), color='red', linestyle='--', 
                         label=f'Mean: {np.mean(wear_probabilities):.3f}')
        axes[1,0].set_xlabel('Wear Probability')
        axes[1,0].set_ylabel('Frequency')
        axes[1,0].set_title('Risk Distribution')
        axes[1,0].legend()
        
        # 4. Validation Context
        validation_text = f"""
        🎯 Validation Context:
        
        📚 Training: Experiments 1-16
        📊 Train/Test Split: 80%/20%
        🧪 Test Accuracy: {self.test_accuracy:.1%}
        
        🔍 True Validation:
        • Experiment 17 (unworn)
        • Experiment 18 (worn)
        
        📈 This prediction shows how well
        the model generalizes to truly
        unseen data.
        """
        
        axes[1,1].text(0.1, 0.5, validation_text, transform=axes[1,1].transAxes, 
                      fontsize=10, verticalalignment='center',
                      bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen"))
        axes[1,1].set_title('Validation Context')
        axes[1,1].axis('off')
        
        plt.tight_layout()
        plt.savefig('validation_prediction_report.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("📊 Validation report saved as 'validation_prediction_report.png'")

def main():
    """Train the validation model"""
    print("🎯 Validation Tool Wear Predictor")
    print("=" * 50)
    
    # Create predictor
    predictor = ValidationToolWearPredictor()
    
    # Train model on experiments 1-16 with 80/20 split
    test_accuracy, feature_importance = predictor.train_model()
    
    print(f"\n🎉 Validation Model Training Complete!")
    print(f"📁 Training files: {len(predictor.training_files)} (1-16)")
    print(f"🔍 Validation files: {len(predictor.validation_files)} (17-18)")
    print(f"📊 Test accuracy: {test_accuracy:.3f}")
    print(f"🎯 Ready for true validation with experiments 17-18!")

if __name__ == "__main__":
    main() 