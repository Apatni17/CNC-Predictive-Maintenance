import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

class ToolWearPredictor:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_importance = None
        self.class_weights = None
        
    def load_and_prepare_data(self):
        """Load data from experiments 18, 11, and 5 and prepare for modeling"""
        print("Loading and preparing data...")
        
        # Define tool conditions based on train.csv
        # Experiment 18: worn, Experiment 11: unworn, Experiment 5: unworn
        experiment_conditions = {18: 'worn', 11: 'unworn', 5: 'unworn'}
        
        # Load data from experiments 18, 11, and 5 - last 500 points
        all_data = []
        experiments = [18, 11, 5]
        
        for exp_num in experiments:
            try:
                df = pd.read_csv(f"data/CNC data /experiment_{exp_num:02d}.csv")
                # Take only the last 500 points from each experiment
                df = df.tail(500)
                df['tool_condition'] = experiment_conditions[exp_num]
                df['experiment_id'] = exp_num
                all_data.append(df)
                print(f"Loaded last 500 points from experiment {exp_num} ({experiment_conditions[exp_num]})")
            except Exception as e:
                print(f"Error loading experiment {exp_num}: {e}")
        
        # Combine all data
        combined_data = pd.concat(all_data, ignore_index=True)
        print(f"Loaded {len(combined_data)} samples (last 500 points from each experiment)")
        print(f"Unworn samples: {len(combined_data[combined_data['tool_condition'] == 'unworn'])}")
        print(f"Worn samples: {len(combined_data[combined_data['tool_condition'] == 'worn'])}")
        
        return combined_data
    
    def load_and_prepare_data_new(self):
        """Load data from all experiments except 5 and 6, with balanced sampling"""
        print("Loading and preparing data from all experiments except 5 and 6...")
        
        # Define tool conditions based on train.csv - CORRECTED MAPPING
        # Based on the actual train.csv file, here's the correct mapping:
        experiment_conditions = {
            1: 'unworn', 2: 'unworn', 3: 'unworn', 4: 'unworn',
            7: 'worn', 8: 'worn', 9: 'worn', 10: 'worn',
            11: 'unworn', 12: 'unworn', 13: 'worn', 14: 'worn',
            15: 'worn', 16: 'worn', 17: 'unworn', 18: 'worn'
        }
        
        # Load data from all experiments except 5 and 6
        all_data = []
        experiments = [1, 2, 3, 4, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]
        
        for exp_num in experiments:
            try:
                df = pd.read_csv(f"data/CNC data /experiment_{exp_num:02d}.csv")
                # Take only the last 500 points from each experiment
                df = df.tail(500)
                df['tool_condition'] = experiment_conditions[exp_num]
                df['experiment_id'] = exp_num
                all_data.append(df)
                print(f"Loaded last 500 points from experiment {exp_num} ({experiment_conditions[exp_num]})")
            except Exception as e:
                print(f"Error loading experiment {exp_num}: {e}")
        
        # Combine all data
        combined_data = pd.concat(all_data, ignore_index=True)
        
        # Balance the dataset to prevent bias
        unworn_data = combined_data[combined_data['tool_condition'] == 'unworn']
        worn_data = combined_data[combined_data['tool_condition'] == 'worn']
        
        # Sample equal numbers from each class to prevent bias
        min_samples = min(len(unworn_data), len(worn_data))
        unworn_balanced = unworn_data.sample(n=min_samples, random_state=42)
        worn_balanced = worn_data.sample(n=min_samples, random_state=42)
        
        # Combine balanced data
        balanced_data = pd.concat([unworn_balanced, worn_balanced], ignore_index=True)
        balanced_data = balanced_data.sample(frac=1, random_state=42).reset_index(drop=True)
        
        print(f"Original data: {len(combined_data)} samples")
        print(f"Unworn samples: {len(combined_data[combined_data['tool_condition'] == 'unworn'])}")
        print(f"Worn samples: {len(combined_data[combined_data['tool_condition'] == 'worn'])}")
        print(f"Balanced data: {len(balanced_data)} samples")
        print(f"Balanced unworn samples: {len(balanced_data[balanced_data['tool_condition'] == 'unworn'])}")
        print(f"Balanced worn samples: {len(balanced_data[balanced_data['tool_condition'] == 'worn'])}")
        
        return balanced_data
    
    def extract_features(self, df):
        """Extract features with improved engineering to prevent false positives"""
        print("Extracting features with improved engineering...")
        
        # Select only the most critical sensor columns based on our analysis
        sensor_columns = [
            'X1_CurrentFeedback', 'Y1_CurrentFeedback', 'S1_CurrentFeedback',
            'X1_OutputPower', 'Y1_OutputPower', 'S1_OutputPower',
            'X1_ActualPosition', 'Y1_ActualPosition', 'Z1_ActualPosition',
            'X1_CommandPosition', 'Y1_CommandPosition', 'Z1_CommandPosition'
        ]
        
        # Create feature dataframe with only the most important features
        features_df = df[sensor_columns].copy()
        
        # IMPROVED FEATURE ENGINEERING - More robust features that won't cause false positives
        
        # 1. Current stability (improved calculation)
        features_df['current_stability'] = features_df[['X1_CurrentFeedback', 'Y1_CurrentFeedback', 'S1_CurrentFeedback']].std(axis=1)
        
        # 2. Position tracking error (normalized)
        features_df['position_error_x'] = abs(features_df['X1_ActualPosition'] - features_df['X1_CommandPosition'])
        features_df['position_error_y'] = abs(features_df['Y1_ActualPosition'] - features_df['Y1_CommandPosition'])
        features_df['position_error_z'] = abs(features_df['Z1_ActualPosition'] - features_df['Z1_CommandPosition'])
        features_df['total_position_error'] = features_df['position_error_x'] + features_df['position_error_y'] + features_df['position_error_z']
        
        # 3. Power efficiency (improved calculation)
        current_mean = features_df[['X1_CurrentFeedback', 'Y1_CurrentFeedback', 'S1_CurrentFeedback']].mean(axis=1)
        power_mean = features_df[['X1_OutputPower', 'Y1_OutputPower', 'S1_OutputPower']].mean(axis=1)
        features_df['power_efficiency'] = np.where(current_mean > 0, power_mean / current_mean, 0)
        
        # 4. Current ratios (more robust)
        features_df['x_y_current_ratio'] = np.where(
            features_df['Y1_CurrentFeedback'] > 0, 
            features_df['X1_CurrentFeedback'] / features_df['Y1_CurrentFeedback'], 
            1.0
        )
        
        # 5. Current change (smoothed)
        features_df['x_current_change'] = features_df['X1_CurrentFeedback'].diff().abs().fillna(0)
        features_df['y_current_change'] = features_df['Y1_CurrentFeedback'].diff().abs().fillna(0)
        features_df['s_current_change'] = features_df['S1_CurrentFeedback'].diff().abs().fillna(0)
        
        # 6. Power stability
        features_df['power_stability'] = features_df[['X1_OutputPower', 'Y1_OutputPower', 'S1_OutputPower']].std(axis=1)
        
        # 7. Position stability
        features_df['position_stability'] = features_df[['X1_ActualPosition', 'Y1_ActualPosition', 'Z1_ActualPosition']].std(axis=1)
        
        # 8. Completion indicator (improved calculation)
        features_df['completion_indicator'] = 100 * np.exp(-features_df['current_stability'] / 100)
        
        # Remove any infinite or NaN values
        features_df = features_df.replace([np.inf, -np.inf], np.nan)
        features_df = features_df.fillna(0)
        
        print(f"Extracted {features_df.shape[1]} features")
        return features_df
    
    def prepare_training_data(self, df):
        """Prepare data for training with class weights"""
        print("Preparing training data with class balancing...")
        
        # Extract features
        features = self.extract_features(df)
        
        # Prepare target variable
        target = df['tool_condition'].values
        
        # Encode target variable
        target_encoded = self.label_encoder.fit_transform(target)
        
        # Calculate class weights to handle imbalance
        classes = np.unique(target_encoded)
        self.class_weights = compute_class_weight('balanced', classes=classes, y=target_encoded)
        class_weight_dict = dict(zip(classes, self.class_weights))
        
        print(f"Class weights: {class_weight_dict}")
        
        # Split data with 75/25 train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            features, target_encoded, test_size=0.25, random_state=42, stratify=target_encoded
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        print(f"Training set: {X_train_scaled.shape}")
        print(f"Test set: {X_test_scaled.shape}")
        
        return X_train_scaled, X_test_scaled, y_train, y_test, features.columns
    
    def train_xgboost_model(self, X_train, y_train, feature_names):
        """Train XGBoost model with improved parameters and class weights"""
        print("Training XGBoost model with class weights...")
        
        # Initialize XGBoost classifier with improved parameters
        self.model = xgb.XGBClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            objective='binary:logistic',
            random_state=42,
            eval_metric='logloss',
            scale_pos_weight=self.class_weights[1] / self.class_weights[0] if len(self.class_weights) > 1 else 1
        )
        
        # Train the model
        self.model.fit(X_train, y_train)
        
        # Get feature importance
        self.feature_importance = pd.DataFrame({
            'feature': feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("Model training completed successfully!")
        
        return self.model
    
    def evaluate_model(self, X_test, y_test):
        """Evaluate the trained model with detailed analysis"""
        print("Evaluating model...")
        
        # Make predictions
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-Score: {f1:.4f}")
        
        # Print classification report
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=['Unworn', 'Worn']))
        
        # Create confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['Unworn', 'Worn'], 
                    yticklabels=['Unworn', 'Worn'])
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Analyze false positives (unworn tools predicted as worn)
        tn, fp, fn, tp = cm.ravel()
        false_positive_rate = fp / (fp + tn) if (fp + tn) > 0 else 0
        false_negative_rate = fn / (fn + tp) if (fn + tp) > 0 else 0
        
        print(f"\nFalse Positive Rate (Unworn predicted as Worn): {false_positive_rate:.4f}")
        print(f"False Negative Rate (Worn predicted as Unworn): {false_negative_rate:.4f}")
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'predictions': y_pred,
            'probabilities': y_pred_proba,
            'false_positive_rate': false_positive_rate,
            'false_negative_rate': false_negative_rate
        }
    
    def plot_feature_importance(self, top_n=15):
        """Plot feature importance"""
        print("Plotting feature importance...")
        
        top_features = self.feature_importance.head(top_n)
        
        plt.figure(figsize=(12, 8))
        sns.barplot(data=top_features, x='importance', y='feature')
        plt.title(f'Top {top_n} Feature Importance for Tool Wear Prediction')
        plt.xlabel('Importance')
        plt.ylabel('Feature')
        plt.tight_layout()
        plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("\nTop 10 Most Important Features:")
        for idx, row in self.feature_importance.head(10).iterrows():
            print(f"{row['feature']}: {row['importance']:.4f}")
    
    def predict_new_data(self, new_data):
        """Predict tool wear for new data with improved accuracy"""
        if self.model is None:
            raise ValueError("Model not trained yet. Please train the model first.")
        
        # Extract features from new data
        features = self.extract_features(new_data)
        
        # Scale features
        features_scaled = self.scaler.transform(features)
        
        # Make predictions based on model type
        if hasattr(self.model, 'predict_proba'):
            # Both XGBoost and Random Forest support predict_proba
            predictions = self.model.predict(features_scaled)
            probabilities = self.model.predict_proba(features_scaled)[:, 1]
        else:
            # Fallback for models that don't support predict_proba
            predictions = self.model.predict(features_scaled)
            probabilities = np.zeros(len(predictions))  # Default probabilities
        
        # Convert back to original labels
        predictions_labels = self.label_encoder.inverse_transform(predictions)
        
        # Apply confidence threshold to reduce false positives
        confidence_threshold = 0.6  # Only predict 'worn' if confidence > 60%
        adjusted_predictions = []
        adjusted_probabilities = []
        
        for i, (pred, prob) in enumerate(zip(predictions_labels, probabilities)):
            if pred == 'worn' and prob < confidence_threshold:
                # If predicted as worn but confidence is low, classify as unworn
                adjusted_predictions.append('unworn')
                adjusted_probabilities.append(1 - prob)  # Invert the probability
            else:
                adjusted_predictions.append(pred)
                adjusted_probabilities.append(prob)
        
        return np.array(adjusted_predictions), np.array(adjusted_probabilities)
    
    def save_model(self, filename='tool_wear_model.pkl'):
        """Save the trained model"""
        import pickle
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'label_encoder': self.label_encoder,
            'feature_importance': self.feature_importance
        }
        
        with open(filename, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"Model saved to {filename}")
    
    def load_model(self, filename='tool_wear_model.pkl'):
        """Load a trained model"""
        import pickle
        
        with open(filename, 'rb') as f:
            model_data = pickle.load(f)
        
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.label_encoder = model_data['label_encoder']
        self.feature_importance = model_data['feature_importance']
        
        print(f"Model loaded from {filename}")

def main():
    """Main function to run the tool wear prediction pipeline"""
    print("=== Tool Wear Prediction Model ===")
    print("Based on comprehensive analysis of CNC sensor data")
    
    # Initialize predictor
    predictor = ToolWearPredictor()
    
    # Load and prepare data
    data = predictor.load_and_prepare_data()
    
    # Prepare training data
    X_train, X_test, y_train, y_test, feature_names = predictor.prepare_training_data(data)
    
    # Train model
    model = predictor.train_xgboost_model(X_train, y_train, feature_names)
    
    # Evaluate model
    results = predictor.evaluate_model(X_test, y_test)
    
    # Plot feature importance
    predictor.plot_feature_importance()
    
    # Save model
    predictor.save_model()
    
    print("\n=== Model Training Complete ===")
    print("The model can now predict tool wear based on sensor data patterns.")
    print("Key insights from our analysis have been incorporated into the feature engineering.")

if __name__ == "__main__":
    main() 