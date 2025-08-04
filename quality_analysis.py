import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

class QualityAnalyzer:
    def __init__(self, data_dir="data/CNC mill wear /"):
        self.data_dir = data_dir
        self.combined_data = None
        self.sampled_data = None
        
    def load_and_prepare_data(self):
        """Load data and create quality labels"""
        print("Loading CNC mill data...")
        
        # Load all experiments
        experiments = []
        for i in range(1, 19):
            try:
                file_path = f"{self.data_dir}experiment_{i:02d}.csv"
                df = pd.read_csv(file_path)
                df['experiment_id'] = i
                experiments.append(df)
                print(f"Loaded experiment_{i:02d}.csv")
            except FileNotFoundError:
                print(f"Warning: {file_path} not found")
                continue
        
        if not experiments:
            raise ValueError("No experiment files found!")
        
        # Combine all experiments
        self.combined_data = pd.concat(experiments, ignore_index=True)
        print(f"Combined data shape: {self.combined_data.shape}")
        
        # Sample data (300 points per experiment)
        self.sampled_data = self.combined_data.groupby('experiment_id').apply(
            lambda x: x.sample(n=min(300, len(x)), random_state=42)
        ).reset_index(drop=True)
        
        print(f"Sampled data shape: {self.sampled_data.shape}")
        
        # Create quality labels
        self.create_quality_labels()
        
    def create_quality_labels(self):
        """Create labels for tool condition and machine finalization"""
        print("Creating quality labels...")
        
        # 1. Tool Condition (based on experiment ID: 1-8 unworn, 9-18 worn)
        self.sampled_data['tool_condition'] = (
            self.sampled_data['experiment_id'].apply(lambda x: 0 if x <= 8 else 1)
        )
        
        # 2. Machine Finalization (completion success) - CORRECTED LOGIC
        # Check if the process reached completion by looking for "End" or "end"
        self.sampled_data['process_completed'] = (
            (self.sampled_data['Machining_Process'] == 'End') | 
            (self.sampled_data['Machining_Process'] == 'end')
        ).astype(int)
        
        # Check for successful position attainment
        position_tolerance = 0.1  # 0.1mm tolerance
        x_position_success = (abs(self.sampled_data['X1_ActualPosition'] - self.sampled_data['X1_CommandPosition']) < position_tolerance).astype(int)
        y_position_success = (abs(self.sampled_data['Y1_ActualPosition'] - self.sampled_data['Y1_CommandPosition']) < position_tolerance).astype(int)
        z_position_success = (abs(self.sampled_data['Z1_ActualPosition'] - self.sampled_data['Z1_CommandPosition']) < position_tolerance).astype(int)
        
        self.sampled_data['position_success'] = ((x_position_success + y_position_success + z_position_success) >= 2).astype(int)
        
        # Check for stable current feedback
        current_threshold = self.sampled_data['X1_CurrentFeedback'].quantile(0.95)
        self.sampled_data['current_stable'] = (
            self.sampled_data['X1_CurrentFeedback'] < current_threshold
        ).astype(int)
        
        # Check for successful velocity attainment
        velocity_tolerance = 0.05  # 5% tolerance
        self.sampled_data['velocity_success'] = (
            abs(self.sampled_data['X1_ActualVelocity'] - self.sampled_data['X1_CommandVelocity']) / 
            (self.sampled_data['X1_CommandVelocity'] + 1e-6) < velocity_tolerance
        ).astype(int)
        
        # Machine finalization success (at least 3 out of 4 criteria)
        success_sum = (self.sampled_data['process_completed'] + 
                      self.sampled_data['position_success'] + 
                      self.sampled_data['current_stable'] + 
                      self.sampled_data['velocity_success'])
        
        self.sampled_data['machine_finalization'] = (success_sum >= 3).astype(int)
        
        print(f"Tool condition distribution:")
        print(self.sampled_data['tool_condition'].value_counts())
        print(f"\nMachine finalization success rate: {self.sampled_data['machine_finalization'].mean():.2%}")
        
    def analyze_tool_condition_impact(self):
        """Analyze how tool condition affects machine finalization"""
        print("\n=== Analyzing Tool Condition Impact ===")
        
        # Create summary table
        summary_data = []
        
        for condition in [0, 1]:  # 0 = unworn, 1 = worn
            condition_data = self.sampled_data[self.sampled_data['tool_condition'] == condition]
            
            finalization_rate = condition_data['machine_finalization'].mean()
            
            summary_data.append({
                'Tool Condition': 'Unworn' if condition == 0 else 'Worn',
                'Machine Finalization Rate': f"{finalization_rate:.2%}",
                'Sample Size': len(condition_data)
            })
        
        summary_df = pd.DataFrame(summary_data)
        print("\nTool Condition Impact Summary:")
        print(summary_df)
        
        # Create visualization
        plt.figure(figsize=(10, 6))
        
        # Machine finalization by tool condition
        finalization_by_condition = self.sampled_data.groupby('tool_condition')['machine_finalization'].mean()
        bars = plt.bar(['Unworn', 'Worn'], finalization_by_condition.values, color=['#2ecc71', '#e74c3c'], alpha=0.7)
        plt.title('Machine Finalization Rate by Tool Condition', fontsize=14, fontweight='bold')
        plt.ylabel('Finalization Success Rate')
        plt.ylim(0, 1)
        
        # Add percentage labels
        for i, v in enumerate(finalization_by_condition.values):
            plt.text(i, v + 0.01, f'{v:.1%}', ha='center', va='bottom', fontweight='bold', fontsize=12)
        
        plt.tight_layout()
        plt.savefig('tool_condition_impact.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        return summary_df
        
    def analyze_quality_correlations(self):
        """Analyze correlations between quality metrics and key variables"""
        print("\n=== Analyzing Quality Correlations ===")
        
        # Select key variables for analysis
        key_variables = [
            'X1_CurrentFeedback', 'Y1_CurrentFeedback', 'S1_CurrentFeedback',
            'X1_DCBusVoltage', 'Y1_DCBusVoltage', 'S1_DCBusVoltage',
            'X1_OutputCurrent', 'Y1_OutputCurrent', 'S1_OutputCurrent',
            'X1_OutputPower', 'Y1_OutputPower', 'S1_OutputPower',
            'M1_CURRENT_FEEDRATE',
            'X1_ActualPosition', 'Y1_ActualPosition', 'Z1_ActualPosition',
            'X1_CommandPosition', 'Y1_CommandPosition', 'Z1_CommandPosition',
            'X1_ActualVelocity', 'Y1_ActualVelocity', 'S1_ActualVelocity',
            'X1_CommandVelocity', 'Y1_CommandVelocity', 'S1_CommandVelocity'
        ]
        
        # Filter to available variables
        available_vars = [v for v in key_variables if v in self.sampled_data.columns]
        
        # Create correlation matrix with quality metrics
        quality_vars = ['tool_condition', 'machine_finalization']
        corr_data = self.sampled_data[available_vars + quality_vars].copy()
        
        # Calculate correlations
        corr_matrix = corr_data.corr()
        
        # Get correlations with quality metrics
        quality_correlations = {}
        for quality_var in quality_vars:
            quality_correlations[quality_var] = corr_matrix[quality_var].abs().sort_values(ascending=False)
        
        # Create correlation heatmap for quality metrics
        quality_corr = corr_matrix[quality_vars].loc[available_vars]
        
        # Sort variables by their correlation with machine finalization for better readability
        finalization_corr = quality_corr['machine_finalization'].abs().sort_values(ascending=False)
        top_vars = finalization_corr.head(15).index.tolist()  # Top 15 variables
        
        # Create a more readable heatmap with top variables
        quality_corr_top = quality_corr.loc[top_vars]
        
        plt.figure(figsize=(12, 10))
        
        # Create custom color palette for better contrast
        colors = ['#d73027', '#f46d43', '#fdae61', '#fee08b', '#e6f598', '#abdda4', '#66c2a5', '#3288bd']
        
        sns.heatmap(quality_corr_top, 
                   annot=True, 
                   cmap='RdBu_r', 
                   center=0,
                   square=True,
                   fmt='.2f',
                   cbar_kws={"shrink": .8, "label": "Correlation Coefficient"},
                   annot_kws={'size': 10, 'weight': 'bold'},
                   linewidths=0.5,
                   linecolor='white')
        
        plt.title('Correlation Matrix: Tool Condition & Machine Finalization vs Top Variables', 
                 fontsize=18, pad=20, fontweight='bold')
        plt.xlabel('Quality Metrics', fontsize=14, fontweight='bold')
        plt.ylabel('Key Variables (Sorted by Machine Finalization Correlation)', fontsize=14, fontweight='bold')
        
        # Rotate x-axis labels for better readability
        plt.xticks(rotation=0, fontsize=12)
        plt.yticks(rotation=0, fontsize=10)
        
        plt.tight_layout()
        plt.savefig('quality_correlations.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Print top correlations for each quality metric
        print("\nTop 10 correlations with Tool Condition:")
        print(quality_correlations['tool_condition'].head(11))
        
        print("\nTop 10 correlations with Machine Finalization:")
        print(quality_correlations['machine_finalization'].head(11))
        
        return quality_correlations
        
    def create_quality_prediction_models(self):
        """Create prediction models for quality metrics"""
        print("\n=== Creating Quality Prediction Models ===")
        
        # Select features for prediction
        feature_vars = [
            'X1_CurrentFeedback', 'Y1_CurrentFeedback', 'S1_CurrentFeedback',
            'X1_DCBusVoltage', 'Y1_DCBusVoltage', 'S1_DCBusVoltage',
            'X1_OutputCurrent', 'Y1_OutputCurrent', 'S1_OutputCurrent',
            'X1_OutputPower', 'Y1_OutputPower', 'S1_OutputPower',
            'M1_CURRENT_FEEDRATE',
            'X1_ActualPosition', 'Y1_ActualPosition', 'Z1_ActualPosition',
            'X1_CommandPosition', 'Y1_CommandPosition', 'Z1_CommandPosition',
            'X1_ActualVelocity', 'Y1_ActualVelocity', 'S1_ActualVelocity',
            'X1_CommandVelocity', 'Y1_CommandVelocity', 'S1_CommandVelocity'
        ]
        
        available_features = [f for f in feature_vars if f in self.sampled_data.columns]
        
        # Create models for each quality metric
        quality_metrics = ['tool_condition', 'machine_finalization']
        model_results = {}
        
        for metric in quality_metrics:
            print(f"\n--- Predicting {metric.replace('_', ' ').title()} ---")
            
            X = self.sampled_data[available_features]
            y = self.sampled_data[metric]
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
            
            # Train Random Forest
            rf = RandomForestClassifier(n_estimators=100, random_state=42)
            rf.fit(X_train, y_train)
            
            # Get feature importance
            feature_importance = pd.DataFrame({
                'feature': available_features,
                'importance': rf.feature_importances_
            }).sort_values('importance', ascending=False)
            
            # Model performance
            y_pred = rf.predict(X_test)
            
            model_results[metric] = {
                'feature_importance': feature_importance,
                'classification_report': classification_report(y_test, y_pred, output_dict=True),
                'accuracy': (y_pred == y_test).mean()
            }
            
            print(f"Accuracy: {model_results[metric]['accuracy']:.3f}")
            print("Top 5 features:")
            print(feature_importance.head())
        
        # Create feature importance plots
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        for i, metric in enumerate(quality_metrics):
            top_features = model_results[metric]['feature_importance'].head(10)
            axes[i].barh(range(len(top_features)), top_features['importance'])
            axes[i].set_yticks(range(len(top_features)))
            axes[i].set_yticklabels(top_features['feature'])
            axes[i].set_xlabel('Feature Importance')
            axes[i].set_title(f'{metric.replace("_", " ").title()} Prediction')
            axes[i].invert_yaxis()
        
        plt.tight_layout()
        plt.savefig('quality_prediction_importance.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        return model_results
        
    def create_quality_summary_statistics(self):
        """Create summary statistics for quality analysis"""
        print("\n=== Creating Quality Summary Statistics ===")
        
        # Group by tool condition and calculate statistics
        quality_stats = self.sampled_data.groupby('tool_condition').agg({
            'machine_finalization': ['mean', 'std', 'count'],
            'X1_CurrentFeedback': ['mean', 'std'],
            'X1_DCBusVoltage': ['mean', 'std'],
            'M1_CURRENT_FEEDRATE': ['mean', 'std']
        }).round(4)
        
        # Flatten column names
        quality_stats.columns = ['_'.join(col).strip() for col in quality_stats.columns]
        
        # Save statistics
        quality_stats.to_csv('quality_statistics.csv')
        print("Saved quality statistics to 'quality_statistics.csv'")
        
        # Create summary table for display
        summary_table = []
        for condition in [0, 1]:
            condition_name = 'Unworn' if condition == 0 else 'Worn'
            condition_data = self.sampled_data[self.sampled_data['tool_condition'] == condition]
            
            summary_table.append({
                'Tool Condition': condition_name,
                'Machine Finalization Rate': f"{condition_data['machine_finalization'].mean():.2%}",
                'Avg X1 Current Feedback': f"{condition_data['X1_CurrentFeedback'].mean():.3f}",
                'Avg X1 DC Bus Voltage': f"{condition_data['X1_DCBusVoltage'].mean():.3f}",
                'Avg Feedrate': f"{condition_data['M1_CURRENT_FEEDRATE'].mean():.1f}",
                'Sample Size': len(condition_data)
            })
        
        summary_df = pd.DataFrame(summary_table)
        print("\nQuality Summary by Tool Condition:")
        print(summary_df)
        
        return quality_stats, summary_df
        
    def run_complete_analysis(self):
        """Run the complete quality analysis"""
        print("🚀 Starting Quality Analysis...")
        
        # Load and prepare data
        self.load_and_prepare_data()
        
        # Analyze tool condition impact
        tool_impact = self.analyze_tool_condition_impact()
        
        # Analyze quality correlations
        quality_correlations = self.analyze_quality_correlations()
        
        # Create prediction models
        model_results = self.create_quality_prediction_models()
        
        # Create summary statistics
        quality_stats, summary_df = self.create_quality_summary_statistics()
        
        print("\n✅ Quality Analysis Complete!")
        print("\nGenerated files:")
        print("- tool_condition_impact.png")
        print("- quality_correlations.png")
        print("- quality_prediction_importance.png")
        print("- quality_statistics.csv")
        
        return {
            'tool_impact': tool_impact,
            'quality_correlations': quality_correlations,
            'model_results': model_results,
            'quality_stats': quality_stats,
            'summary_df': summary_df
        }

if __name__ == "__main__":
    analyzer = QualityAnalyzer()
    results = analyzer.run_complete_analysis() 