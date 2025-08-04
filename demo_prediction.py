import pandas as pd
import numpy as np
from tool_wear_predictor import ToolWearPredictor
import matplotlib.pyplot as plt
import seaborn as sns

def demo_prediction():
    """Demo script to show how to use the trained model for predictions"""
    print("=== Tool Wear Prediction Demo ===")
    
    # Initialize predictor
    predictor = ToolWearPredictor()
    
    # Train the model (or load if already trained)
    try:
        predictor.load_model('tool_wear_model.pkl')
        print("Loaded pre-trained model")
    except:
        print("Training new model...")
        # Load and prepare data
        data = predictor.load_and_prepare_data()
        
        # Prepare training data
        X_train, X_test, y_train, y_test, feature_names = predictor.prepare_training_data(data)
        
        # Train model
        model = predictor.train_xgboost_model(X_train, y_train, feature_names)
        
        # Evaluate model
        results = predictor.evaluate_model(X_test, y_test)
        
        # Save model
        predictor.save_model()
    
    # Demo: Predict on a sample of new data
    print("\n=== Making Predictions on Sample Data ===")
    
    # Load a sample of data for prediction demo
    try:
        # Load a small sample from experiment 3 (unworn) for demo
        sample_data = pd.read_csv("data/CNC data /experiment_03.csv").tail(100)
        sample_data['tool_condition'] = 'unworn'  # We know this is unworn for demo
        
        print(f"Sample data shape: {sample_data.shape}")
        
        # Make predictions
        predictions, probabilities = predictor.predict_new_data(sample_data)
        
        # Analyze results
        print(f"\nPrediction Results:")
        print(f"Total samples: {len(predictions)}")
        print(f"Predicted unworn: {sum(predictions == 'unworn')}")
        print(f"Predicted worn: {sum(predictions == 'worn')}")
        
        # Calculate accuracy for demo (since we know the true labels)
        true_labels = sample_data['tool_condition'].values
        accuracy = sum(predictions == true_labels) / len(predictions)
        print(f"Demo accuracy: {accuracy:.4f}")
        
        # Show probability distribution
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.hist(probabilities, bins=20, alpha=0.7, color='blue')
        plt.title('Probability Distribution')
        plt.xlabel('Probability of Tool Wear')
        plt.ylabel('Frequency')
        plt.axvline(x=0.5, color='red', linestyle='--', label='Decision Threshold')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.scatter(range(len(probabilities)), probabilities, alpha=0.6)
        plt.title('Probability vs Sample Index')
        plt.xlabel('Sample Index')
        plt.ylabel('Probability of Tool Wear')
        plt.axhline(y=0.5, color='red', linestyle='--', label='Decision Threshold')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig('prediction_demo.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Show feature importance
        predictor.plot_feature_importance(top_n=10)
        
        print("\n=== Demo Complete ===")
        print("The model can now predict tool wear based on sensor data patterns.")
        print("Key features used for prediction:")
        for idx, row in predictor.feature_importance.head(5).iterrows():
            print(f"- {row['feature']}: {row['importance']:.4f}")
            
    except Exception as e:
        print(f"Error in demo: {e}")
        print("Make sure the model is trained first by running tool_wear_predictor.py")

def predict_single_sample():
    """Demo function to predict on a single sample"""
    print("\n=== Single Sample Prediction Demo ===")
    
    try:
        # Load the trained model
        predictor = ToolWearPredictor()
        predictor.load_model('tool_wear_model.pkl')
        
        # Create a single sample (simulating real-time data)
        # This would be the current sensor readings from the CNC machine
        sample_data = pd.read_csv("data/CNC data /experiment_01.csv").iloc[-1:].copy()
        
        print("Current sensor readings:")
        print(f"X1 Current: {sample_data['X1_CurrentFeedback'].values[0]:.4f}")
        print(f"Y1 Current: {sample_data['Y1_CurrentFeedback'].values[0]:.4f}")
        print(f"S1 Current: {sample_data['S1_CurrentFeedback'].values[0]:.4f}")
        print(f"X1 Power: {sample_data['X1_OutputPower'].values[0]:.4f}")
        print(f"Y1 Power: {sample_data['Y1_OutputPower'].values[0]:.4f}")
        
        # Make prediction
        prediction, probability = predictor.predict_new_data(sample_data)
        
        print(f"\nPrediction: {prediction[0]}")
        print(f"Confidence: {probability[0]:.4f}")
        
        if prediction[0] == 'worn':
            print("⚠️  WARNING: Tool wear detected! Consider tool replacement.")
        else:
            print("✅ Tool condition appears normal.")
            
    except Exception as e:
        print(f"Error in single sample prediction: {e}")

if __name__ == "__main__":
    demo_prediction()
    predict_single_sample() 