#!/usr/bin/env python3
"""
Test script to verify the improved tool wear prediction model
This script tests whether the model correctly identifies unworn tools
"""

import pandas as pd
import numpy as np
from tool_wear_predictor import ToolWearPredictor
import warnings
warnings.filterwarnings('ignore')

def test_unworn_tool_detection():
    """Test if the model correctly identifies unworn tools"""
    print("=== Testing Unworn Tool Detection ===")
    
    # Initialize predictor
    predictor = ToolWearPredictor()
    
    # Load and prepare data
    print("Loading and preparing data...")
    data = predictor.load_and_prepare_data_new()
    
    # Prepare training data
    X_train, X_test, y_train, y_test, feature_names = predictor.prepare_training_data(data)
    
    # Train model
    print("Training improved model...")
    model = predictor.train_xgboost_model(X_train, y_train, feature_names)
    
    # Evaluate model
    print("Evaluating model...")
    results = predictor.evaluate_model(X_test, y_test)
    
    print(f"\nModel Performance:")
    print(f"Accuracy: {results['accuracy']:.4f}")
    print(f"Precision: {results['precision']:.4f}")
    print(f"Recall: {results['recall']:.4f}")
    print(f"F1-Score: {results['f1_score']:.4f}")
    print(f"False Positive Rate: {results['false_positive_rate']:.4f}")
    print(f"False Negative Rate: {results['false_negative_rate']:.4f}")
    
    # Test with unworn tool data
    print("\n=== Testing with Unworn Tool Data ===")
    
    # Get unworn samples from test set
    unworn_indices = np.where(y_test == 0)[0]  # Assuming 0 = unworn
    if len(unworn_indices) > 0:
        unworn_X_test = X_test[unworn_indices]
        unworn_y_test = y_test[unworn_indices]
        
        # Make predictions on unworn data
        unworn_predictions = model.predict(unworn_X_test)
        unworn_probabilities = model.predict_proba(unworn_X_test)[:, 1]
        
        # Calculate metrics for unworn data
        unworn_accuracy = np.mean(unworn_predictions == unworn_y_test)
        false_positives = np.sum(unworn_predictions != unworn_y_test)
        total_unworn = len(unworn_predictions)
        
        print(f"Unworn samples tested: {total_unworn}")
        print(f"Unworn accuracy: {unworn_accuracy:.4f}")
        print(f"False positives (unworn predicted as worn): {false_positives}/{total_unworn} ({false_positives/total_unworn*100:.1f}%)")
        
        if false_positives > 0:
            print("⚠️  WARNING: Model still has false positives for unworn tools!")
            print("Average confidence for false positives:", np.mean(unworn_probabilities[unworn_predictions != unworn_y_test]))
        else:
            print("✅ SUCCESS: Model correctly identifies all unworn tools!")
    
    # Test with worn tool data
    print("\n=== Testing with Worn Tool Data ===")
    
    # Get worn samples from test set
    worn_indices = np.where(y_test == 1)[0]  # Assuming 1 = worn
    if len(worn_indices) > 0:
        worn_X_test = X_test[worn_indices]
        worn_y_test = y_test[worn_indices]
        
        # Make predictions on worn data
        worn_predictions = model.predict(worn_X_test)
        worn_probabilities = model.predict_proba(worn_X_test)[:, 1]
        
        # Calculate metrics for worn data
        worn_accuracy = np.mean(worn_predictions == worn_y_test)
        false_negatives = np.sum(worn_predictions != worn_y_test)
        total_worn = len(worn_predictions)
        
        print(f"Worn samples tested: {total_worn}")
        print(f"Worn accuracy: {worn_accuracy:.4f}")
        print(f"False negatives (worn predicted as unworn): {false_negatives}/{total_worn} ({false_negatives/total_worn*100:.1f}%)")
        
        if false_negatives > 0:
            print("⚠️  WARNING: Model has false negatives for worn tools!")
        else:
            print("✅ SUCCESS: Model correctly identifies all worn tools!")
    
    return results

def test_risk_score_calculation():
    """Test the risk score calculation for unworn tools"""
    print("\n=== Testing Risk Score Calculation ===")
    
    # Initialize predictor
    predictor = ToolWearPredictor()
    
    # Load and prepare data
    data = predictor.load_and_prepare_data_new()
    X_train, X_test, y_train, y_test, feature_names = predictor.prepare_training_data(data)
    model = predictor.train_xgboost_model(X_train, y_train, feature_names)
    
    # Get unworn samples
    unworn_indices = np.where(y_test == 0)[0]
    if len(unworn_indices) > 0:
        unworn_X_test = X_test[unworn_indices]
        
        # Make predictions using the improved method
        # We need to convert back to original format for the predict_new_data method
        # This is a simplified test
        unworn_predictions = model.predict(unworn_X_test)
        unworn_probabilities = model.predict_proba(unworn_X_test)[:, 1]
        
        # Calculate risk score (simulating the app logic)
        total_samples = len(unworn_predictions)
        worn_count = np.sum(unworn_predictions == 1)  # Assuming 1 = worn
        wear_percentage = (worn_count / total_samples) * 100
        
        print(f"Unworn samples: {total_samples}")
        print(f"Predicted as worn: {worn_count}")
        print(f"Wear percentage: {wear_percentage:.1f}%")
        
        if wear_percentage < 10:
            print("✅ SUCCESS: Risk score calculation correctly shows low wear for unworn tools!")
        else:
            print(f"⚠️  WARNING: Risk score calculation shows {wear_percentage:.1f}% wear for unworn tools!")

if __name__ == "__main__":
    # Run tests
    results = test_unworn_tool_detection()
    test_risk_score_calculation()
    
    print("\n=== Test Summary ===")
    print("The improved model should:")
    print("1. Have lower false positive rate for unworn tools")
    print("2. Maintain good accuracy for worn tools")
    print("3. Provide more conservative risk scores for unworn tools")
    print("4. Use confidence thresholds to reduce false alarms") 