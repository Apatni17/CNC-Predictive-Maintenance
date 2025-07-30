import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import seaborn as sns
import numpy as np

def main():
    st.set_page_config(page_title="CNC Predictive Maintenance", layout="wide")
    st.title("🔧 CNC Predictive Maintenance Model Comparison")
    
    # File upload
    uploaded_file = st.file_uploader("📁 Upload your dataset (CSV)", type=['csv'])
    
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        st.success(f"✅ Dataset loaded: {len(data)} rows, {len(data.columns)} columns")
        
        # Show column names
        st.write("**Available columns:**", list(data.columns))
        
        # Detect target column
        possible_target_columns = ['Anomaly', 'anomaly', 'target', 'failure', 'label', 'class', 'Machine failure', 'machine failure', 'Machine Failure']
        target_column = None
        
        for col in possible_target_columns:
            if col in data.columns:
                target_column = col
                break
        
        if target_column:
            st.success(f"✅ Target column detected: '{target_column}'")
            
            # Show original class distribution
            original_counts = data[target_column].value_counts()
            st.write("**Original class distribution:**")
            st.write(original_counts)
            
            # Custom sampling section
            st.markdown("### 📊 Custom Data Sampling")
            
            # Show original data info
            total_normal = len(data[data[target_column] == 0])
            total_failures = len(data[data[target_column] == 1])
            st.info(f"📈 Total data available: {total_normal} normal cases, {total_failures} failure cases")
            
            # Training distribution controls
            st.markdown("#### 🎯 Training Distribution")
            col1, col2 = st.columns(2)
            with col1:
                training_normal = st.number_input('✅ Training Normal Cases', 
                                                value=700, min_value=1, max_value=total_normal,
                                                help='Number of normal cases for training')
            with col2:
                training_failures = st.number_input('❌ Training Failure Cases', 
                                                  value=300, min_value=1, max_value=total_failures,
                                                  help='Number of failure cases for training')
            
            # Testing distribution controls
            st.markdown("#### 🧪 Testing Distribution")
            col1, col2 = st.columns(2)
            with col1:
                test_normal = st.number_input('✅ Test Normal Cases', 
                                            value=min(1000, total_normal - training_normal), 
                                            min_value=0, max_value=total_normal - training_normal,
                                            help='Number of normal cases for testing')
            with col2:
                test_failures = st.number_input('❌ Test Failure Cases', 
                                              value=min(500, total_failures - training_failures), 
                                              min_value=0, max_value=total_failures - training_failures,
                                              help='Number of failure cases for testing')
            
            # Validation
            if training_normal + test_normal > total_normal:
                st.error(f"❌ Too many normal cases requested! Available: {total_normal}, Requested: {training_normal + test_normal}")
                return
            if training_failures + test_failures > total_failures:
                st.error(f"❌ Too many failure cases requested! Available: {total_failures}, Requested: {training_failures + test_failures}")
                return
            
            # Perform the sampling
            if st.button('🔄 Apply Sampling', key='apply_sampling'):
                # Sample training data
                training_normal_data = data[data[target_column] == 0].sample(n=training_normal, random_state=42)
                training_failure_data = data[data[target_column] == 1].sample(n=training_failures, random_state=42)
                training_data = pd.concat([training_normal_data, training_failure_data]).sample(frac=1, random_state=42).reset_index(drop=True)
                
                # Sample test data
                remaining_normal = data[data[target_column] == 0].drop(training_normal_data.index)
                remaining_failures = data[data[target_column] == 1].drop(training_failure_data.index)
                
                test_normal_data = remaining_normal.sample(n=test_normal, random_state=42)
                test_failure_data = remaining_failures.sample(n=test_failures, random_state=42)
                test_data = pd.concat([test_normal_data, test_failure_data]).sample(frac=1, random_state=42).reset_index(drop=True)
                
                st.success(f"✅ Training data: {len(training_data)} rows ({training_normal} normal + {training_failures} failures)")
                st.success(f"✅ Test data: {len(test_data)} rows ({test_normal} normal + {test_failures} failures)")
                
                # Show distributions
                training_counts = training_data[target_column].value_counts()
                test_counts = test_data[target_column].value_counts()
                
                col1, col2 = st.columns(2)
                with col1:
                    st.write("**Training distribution:**")
                    st.write(training_counts)
                with col2:
                    st.write("**Test distribution:**")
                    st.write(test_counts)
                
                # Store data for model comparison
                st.session_state.training_data = training_data
                st.session_state.test_data = test_data
                st.session_state.target_column = target_column
            
            # Model selection
            st.markdown("### 🤖 Model Selection")
            models_to_compare = st.multiselect(
                'Select models to compare:',
                ['Decision Tree', 'Random Forest', 'Logistic Regression', 'Gaussian Naive Bayes', 'K-Nearest Neighbors', 'XGBoost'],
                default=['Decision Tree', 'Random Forest', 'Logistic Regression', 'XGBoost']
            )
            
            if st.button('🚀 Compare Models', use_container_width=True):
                if len(models_to_compare) == 0:
                    st.error("Please select at least one model to compare.")
                elif 'training_data' not in st.session_state:
                    st.error("Please apply sampling first by clicking 'Apply Sampling'.")
                else:
                    # Run comparison
                    results = compare_models(st.session_state.training_data, st.session_state.test_data, st.session_state.target_column, models_to_compare)
                    display_results(results, models_to_compare)
        else:
            st.error(f"❌ No target column found! Please ensure your data has one of these columns: {possible_target_columns}")
            st.write("**Available columns:**", list(data.columns))

def compare_models(training_data, test_data, target_column, models_to_compare):
    """Compare multiple models and return results"""
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.naive_bayes import GaussianNB
    from sklearn.neighbors import KNeighborsClassifier
    from xgboost import XGBClassifier
    
    # Prepare training data
    X_train = training_data.drop(columns=[target_column])
    y_train = training_data[target_column]
    
    # Prepare test data
    X_test = test_data.drop(columns=[target_column])
    y_test = test_data[target_column]
    
    # Keep only numeric columns for both datasets
    numeric_columns = X_train.select_dtypes(include=[np.number]).columns
    X_train = X_train[numeric_columns]
    X_test = X_test[numeric_columns]
    
    # Clean feature names for XGBoost compatibility
    X_train.columns = [col.replace('[', '').replace(']', '').replace('<', '').replace('>', '').replace(' ', '_') for col in X_train.columns]
    X_test.columns = [col.replace('[', '').replace(']', '').replace('<', '').replace('>', '').replace(' ', '_') for col in X_test.columns]
    
    # Model mapping
    model_map = {
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'Random Forest': RandomForestClassifier(random_state=42),
        'Logistic Regression': LogisticRegression(random_state=42),
        'Gaussian Naive Bayes': GaussianNB(),
        'K-Nearest Neighbors': KNeighborsClassifier(),
        'XGBoost': XGBClassifier(random_state=42)
    }
    
    results = {}
    
    for model_name in models_to_compare:
        if model_name in model_map:
            # Train model
            model = model_map[model_name]
            model.fit(X_train, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test)
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, zero_division=0)
            recall = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)
            
            # Confusion matrix
            cm = confusion_matrix(y_test, y_pred)
            
            results[model_name] = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'confusion_matrix': cm,
                'y_test': y_test,
                'y_pred': y_pred
            }
    
    return results

def display_results(results, models_to_compare):
    """Display model comparison results"""
    st.markdown("### 📊 Model Comparison Results")
    
    # Metrics comparison table
    st.markdown("#### 📈 Performance Metrics")
    metrics_df = pd.DataFrame({
        'Model': list(results.keys()),
        'Accuracy': [results[model]['accuracy'] for model in results.keys()],
        'Precision': [results[model]['precision'] for model in results.keys()],
        'Recall': [results[model]['recall'] for model in results.keys()],
        'F1-Score': [results[model]['f1'] for model in results.keys()]
    })
    
    # Format metrics as percentages
    for col in ['Accuracy', 'Precision', 'Recall', 'F1-Score']:
        metrics_df[col] = metrics_df[col].apply(lambda x: f"{x:.3f} ({x*100:.1f}%)")
    
    st.dataframe(metrics_df, use_container_width=True)
    
    # Find best model
    best_model = max(results.keys(), key=lambda x: results[x]['f1'])
    st.success(f"🏆 **Best Model (by F1-Score):** {best_model}")
    
    # Confusion matrices
    st.markdown("#### 🎯 Confusion Matrices")
    
    # Create subplots for confusion matrices
    n_models = len(results)
    cols = min(3, n_models)
    rows = (n_models + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows))
    if n_models == 1:
        axes = [axes]
    elif rows == 1:
        axes = axes
    else:
        axes = axes.flatten()
    
    for i, (model_name, result) in enumerate(results.items()):
        ax = axes[i]
        cm = result['confusion_matrix']
        
        # Create confusion matrix heatmap
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                   xticklabels=['Normal', 'Failure'],
                   yticklabels=['Normal', 'Failure'])
        ax.set_title(f'{model_name}\nAccuracy: {result["accuracy"]:.3f}')
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
    
    # Hide empty subplots
    for i in range(n_models, len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    st.pyplot(fig)
    
    # Detailed analysis for each model
    st.markdown("#### 📋 Detailed Analysis")
    
    for model_name, result in results.items():
        with st.expander(f"📊 {model_name} - Detailed Analysis"):
            cm = result['confusion_matrix']
            
            # Extract values from confusion matrix
            tn, fp, fn, tp = cm.ravel()
            
            st.markdown(f"**Confusion Matrix Summary for {model_name}:**")
            st.markdown(f"- **True Negatives (TN):** {tn} - Correctly predicted normal cases")
            st.markdown(f"- **False Positives (FP):** {fp} - Normal cases incorrectly predicted as failures")
            st.markdown(f"- **False Negatives (FN):** {fn} - Failure cases incorrectly predicted as normal")
            st.markdown(f"- **True Positives (TP):** {tp} - Correctly predicted failure cases")
            
            # Analysis
            st.markdown("**📈 Analysis:**")
            
            # Overall accuracy
            total = tn + fp + fn + tp
            accuracy = (tn + tp) / total
            st.markdown(f"- **Overall Accuracy:** {accuracy:.1%} - Model correctly predicted {(tn + tp)} out of {total} cases")
            
            # Precision (for failure class)
            if tp + fp > 0:
                precision = tp / (tp + fp)
                st.markdown(f"- **Precision:** {precision:.1%} - When model predicts failure, it's correct {precision:.1%} of the time")
            else:
                st.markdown("- **Precision:** N/A - Model never predicted failure")
            
            # Recall (for failure class)
            if tp + fn > 0:
                recall = tp / (tp + fn)
                st.markdown(f"- **Recall:** {recall:.1%} - Model caught {recall:.1%} of all actual failures")
            else:
                st.markdown("- **Recall:** N/A - No actual failures in test set")
            
            # F1 Score
            if result['f1'] > 0:
                st.markdown(f"- **F1-Score:** {result['f1']:.3f} - Balanced measure of precision and recall")
            
            # Practical implications
            st.markdown("**🔧 Practical Implications:**")
            if fp > fn:
                st.warning(f"⚠️ **High False Positives ({fp}):** Model generates many false alarms. This may lead to unnecessary maintenance checks but ensures safety.")
            elif fn > fp:
                st.error(f"❌ **High False Negatives ({fn}):** Model misses many actual failures. This is dangerous for predictive maintenance.")
            else:
                st.success(f"✅ **Balanced Performance:** Model has reasonable balance between false alarms and missed failures.")
            
            # Recommendations
            st.markdown("**💡 Recommendations:**")
            if result['f1'] > 0.8:
                st.success("✅ **Excellent Model:** This model performs very well and can be used for production.")
            elif result['f1'] > 0.6:
                st.info("📊 **Good Model:** This model performs reasonably well but could be improved.")
            else:
                st.warning("⚠️ **Poor Model:** This model needs improvement before deployment.")

if __name__ == "__main__":
    main()
# Updated for deployment
