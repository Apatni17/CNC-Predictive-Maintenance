import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier
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
        """Load data from all experiments except 5 and 6, with 75/25 train/test split"""
        print("Loading and preparing data from all experiments except 5 and 6...")
        
        # Define tool conditions based on train.csv
        # All experiments except 5 and 6: 1,2,3,4,7,8,9,10,11,12,13,14,15,16,17,18
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
        print(f"Loaded {len(combined_data)} samples (last 500 points from each experiment)")
        print(f"Unworn samples: {len(combined_data[combined_data['tool_condition'] == 'unworn'])}")
        print(f"Worn samples: {len(combined_data[combined_data['tool_condition'] == 'worn'])}")
        
        return combined_data
    
    def extract_features(self, df):
        """Extract only the most important features based on our analysis"""
        print("Extracting most important features...")
        
        # Select only the most critical sensor columns based on our analysis
        sensor_columns = [
            'X1_CurrentFeedback', 'Y1_CurrentFeedback', 'S1_CurrentFeedback',
            'X1_OutputPower', 'Y1_OutputPower', 'S1_OutputPower',
            'X1_ActualPosition', 'Y1_ActualPosition', 'Z1_ActualPosition',
            'X1_CommandPosition', 'Y1_CommandPosition', 'Z1_CommandPosition'
        ]
        
        # Create feature dataframe with only the most important features
        features_df = df[sensor_columns].copy()
        
        # Add only the most critical engineered features based on our analysis
        # 1. Y-axis current ratio (strongest indicator from our analysis)
        features_df['y_current_ratio'] = features_df['Y1_CurrentFeedback'] / (features_df['X1_CurrentFeedback'] + 1e-6)
        
        # 2. X-axis current change (second strongest indicator)
        features_df['x_current_change'] = features_df['X1_CurrentFeedback'].diff().abs()
        
        # 3. Position tracking error (key from our analysis)
        features_df['position_error_x'] = abs(features_df['X1_ActualPosition'] - features_df['X1_CommandPosition'])
        features_df['position_error_y'] = abs(features_df['Y1_ActualPosition'] - features_df['Y1_CommandPosition'])
        features_df['position_error_z'] = abs(features_df['Z1_ActualPosition'] - features_df['Z1_CommandPosition'])
        features_df['total_position_error'] = features_df['position_error_x'] + features_df['position_error_y'] + features_df['position_error_z']
        
        # 4. Current feedback stability (key indicator)
        features_df['current_stability'] = features_df[['X1_CurrentFeedback', 'Y1_CurrentFeedback', 'S1_CurrentFeedback']].std(axis=1)
        
        # 5. Power efficiency (important from our analysis)
        features_df['power_efficiency'] = features_df[['X1_OutputPower', 'Y1_OutputPower', 'S1_OutputPower']].mean(axis=1) / \
                                        features_df[['X1_CurrentFeedback', 'Y1_CurrentFeedback', 'S1_CurrentFeedback']].mean(axis=1)
        
        # 6. Completion indicator (most important feature from previous model)
        features_df['completion_indicator'] = 100 * np.exp(-features_df['current_stability'] / 50)
        
        # Remove any infinite or NaN values
        features_df = features_df.replace([np.inf, -np.inf], np.nan)
        features_df = features_df.fillna(0)
        
        print(f"Extracted {features_df.shape[1]} most important features")
        return features_df
    
    def prepare_training_data(self, df):
        """Prepare data for training"""
        print("Preparing training data...")
        
        # Extract features
        features = self.extract_features(df)
        
        # Prepare target variable
        target = df['tool_condition'].values
        
        # Encode target variable
        target_encoded = self.label_encoder.fit_transform(target)
        
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
        """Train XGBoost model with optimized parameters"""
        print("Training XGBoost model...")
        
        # Initialize XGBoost classifier with optimized parameters
        self.model = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.1,
            subsample=0.9,
            colsample_bytree=0.9,
            objective='binary:logistic',
            random_state=42,
            eval_metric='logloss'
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
        """Evaluate the trained model"""
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
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'predictions': y_pred,
            'probabilities': y_pred_proba
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
        """Predict tool wear for new data"""
        if self.model is None:
            raise ValueError("Model not trained yet. Please train the model first.")
        
        # Extract features from new data
        features = self.extract_features(new_data)
        
        # Scale features
        features_scaled = self.scaler.transform(features)
        
        # Make predictions
        predictions = self.model.predict(features_scaled)
        probabilities = self.model.predict_proba(features_scaled)[:, 1]
        
        # Convert back to original labels
        predictions_labels = self.label_encoder.inverse_transform(predictions)
        
        return predictions_labels, probabilities
    
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