import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class ToolWearAnalyzer:
    def __init__(self, data_dir="data/CNC mill wear /"):
        self.data_dir = data_dir
        self.experiments = []
        self.combined_data = None
        self.sampled_data = None
        
    def load_experiments(self, experiment_numbers=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]):
        """Load multiple experiment files"""
        print("Loading experiment data...")
        
        for exp_num in experiment_numbers:
            try:
                file_path = f"{self.data_dir}experiment_{exp_num:02d}.csv"
                df = pd.read_csv(file_path)
                df['experiment_id'] = exp_num
                self.experiments.append(df)
                print(f"Loaded experiment_{exp_num:02d}.csv - {len(df)} rows")
            except FileNotFoundError:
                print(f"Warning: {file_path} not found")
                
        # Combine all experiments
        if self.experiments:
            self.combined_data = pd.concat(self.experiments, ignore_index=True)
            print(f"\nCombined data shape: {self.combined_data.shape}")
            
    def sample_data(self, samples_per_experiment=300):
        """Sample data from each experiment"""
        print(f"\nSampling {samples_per_experiment} points from each experiment...")
        
        sampled_dfs = []
        for exp_num in self.combined_data['experiment_id'].unique():
            exp_data = self.combined_data[self.combined_data['experiment_id'] == exp_num]
            
            if len(exp_data) >= samples_per_experiment:
                # Sample randomly
                sampled = exp_data.sample(n=samples_per_experiment, random_state=42)
            else:
                # If not enough data, take all available
                sampled = exp_data
                print(f"Warning: Experiment {exp_num} only has {len(exp_data)} samples")
                
            sampled_dfs.append(sampled)
            
        self.sampled_data = pd.concat(sampled_dfs, ignore_index=True)
        print(f"Sampled data shape: {self.sampled_data.shape}")
        
    def create_tool_wear_labels(self):
        """Create tool wear labels based on experiment numbers"""
        # Based on README: experiments 1-8 unworn, 9-18 worn
        unworn_experiments = [1, 2, 3, 4, 5, 6, 7, 8]
        
        self.sampled_data['tool_wear'] = self.sampled_data['experiment_id'].apply(
            lambda x: 1 if x in unworn_experiments else 0  # 1 = unworn, 0 = worn
        )
        
        print(f"\nTool wear distribution:")
        print(self.sampled_data['tool_wear'].value_counts())
        
    def select_key_features(self):
        """Select key features for analysis based on the research questions"""
        # Features related to the research questions
        key_features = [
            # Current feedback and voltage (most important for prediction)
            'X1_CurrentFeedback', 'Y1_CurrentFeedback', 'Z1_CurrentFeedback', 'S1_CurrentFeedback',
            'X1_DCBusVoltage', 'Y1_DCBusVoltage', 'Z1_DCBusVoltage', 'S1_DCBusVoltage',
            'X1_OutputCurrent', 'Y1_OutputCurrent', 'Z1_OutputCurrent', 'S1_OutputCurrent',
            'X1_OutputVoltage', 'Y1_OutputVoltage', 'Z1_OutputVoltage', 'S1_OutputVoltage',
            'X1_OutputPower', 'Y1_OutputPower', 'S1_OutputPower',
            
            # Feedrate and velocities (important for wear mechanisms)
            'M1_CURRENT_FEEDRATE',
            'X1_ActualVelocity', 'Y1_ActualVelocity', 'Z1_ActualVelocity', 'S1_ActualVelocity',
            'X1_CommandVelocity', 'Y1_CommandVelocity', 'Z1_CommandVelocity', 'S1_CommandVelocity',
            
            # Accelerations (mechanical stress)
            'X1_ActualAcceleration', 'Y1_ActualAcceleration', 'Z1_ActualAcceleration', 'S1_ActualAcceleration',
            'X1_CommandAcceleration', 'Y1_CommandAcceleration', 'Z1_CommandAcceleration', 'S1_CommandAcceleration',
            
            # Position differences (system performance)
            'X1_ActualPosition', 'Y1_ActualPosition', 'Z1_ActualPosition', 'S1_ActualPosition',
            'X1_CommandPosition', 'Y1_CommandPosition', 'Z1_CommandPosition', 'S1_CommandPosition',
            
            # System inertia
            'S1_SystemInertia'
        ]
        
        # Filter features that exist in the data
        available_features = [f for f in key_features if f in self.sampled_data.columns]
        print(f"\nSelected {len(available_features)} key features for analysis")
        
        return available_features
        
    def create_correlation_matrix(self, features):
        """Create correlation matrix with tool wear"""
        print("\nCreating correlation matrix...")
        
        # Prepare data for correlation analysis
        corr_data = self.sampled_data[features + ['tool_wear']].copy()
        
        # Calculate correlations
        corr_matrix = corr_data.corr()
        
        # Create correlation heatmap
        plt.figure(figsize=(20, 16))
        
        # Create mask for upper triangle
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        
        # Plot heatmap
        sns.heatmap(corr_matrix, 
                   mask=mask,
                   annot=True, 
                   cmap='RdBu_r', 
                   center=0,
                   square=True,
                   fmt='.2f',
                   cbar_kws={"shrink": .8})
        
        plt.title('Correlation Matrix: Tool Wear vs Key Variables', fontsize=16, pad=20)
        plt.tight_layout()
        plt.savefig('correlation_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Get top correlations with tool wear
        tool_wear_corr = corr_matrix['tool_wear'].abs().sort_values(ascending=False)
        print("\nTop 10 correlations with tool wear:")
        print(tool_wear_corr.head(11))  # 11 to include tool_wear itself
        
        return corr_matrix
        
    def create_feature_importance_analysis(self, features):
        """Analyze feature importance using Random Forest"""
        print("\nAnalyzing feature importance...")
        
        # Prepare data
        X = self.sampled_data[features]
        y = self.sampled_data['tool_wear']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
        
        # Train Random Forest
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X_train, y_train)
        
        # Get feature importance
        feature_importance = pd.DataFrame({
            'feature': features,
            'importance': rf.feature_importances_
        }).sort_values('importance', ascending=False)
        
        # Plot feature importance
        plt.figure(figsize=(12, 8))
        top_features = feature_importance.head(15)
        sns.barplot(data=top_features, x='importance', y='feature')
        plt.title('Top 15 Feature Importance for Tool Wear Prediction', fontsize=14)
        plt.xlabel('Importance')
        plt.tight_layout()
        plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("\nTop 10 most important features:")
        print(feature_importance.head(10))
        
        return rf, feature_importance
        
    def create_roc_analysis(self, rf, features):
        """Create ROC curve analysis"""
        print("\nCreating ROC curve analysis...")
        
        # Prepare data
        X = self.sampled_data[features]
        y = self.sampled_data['tool_wear']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
        
        # Get predictions
        y_pred_proba = rf.predict_proba(X_test)[:, 1]
        y_pred = rf.predict(X_test)
        
        # Calculate ROC curve
        fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        
        # Plot ROC curve
        plt.figure(figsize=(10, 8))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve: Tool Wear Prediction', fontsize=14)
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('roc_curve.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Print classification report
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=['Worn', 'Unworn']))
        
        return roc_auc
        
    def create_distribution_plots(self, top_features):
        """Create distribution plots for top features"""
        print("\nCreating distribution plots...")
        
        # Select top 6 features for visualization
        top_6_features = top_features.head(6)['feature'].tolist()
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.ravel()
        
        for i, feature in enumerate(top_6_features):
            # Create violin plot
            sns.violinplot(data=self.sampled_data, x='tool_wear', y=feature, ax=axes[i])
            axes[i].set_title(f'{feature} vs Tool Wear')
            axes[i].set_xlabel('Tool Wear (0=Worn, 1=Unworn)')
            axes[i].set_ylabel(feature)
            
        plt.tight_layout()
        plt.savefig('feature_distributions.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    def create_time_series_analysis(self, features):
        """Create time series analysis for key features"""
        print("\nCreating time series analysis...")
        
        # Select a few key features for time series analysis
        key_ts_features = ['X1_CurrentFeedback', 'Y1_CurrentFeedback', 'X1_DCBusVoltage', 'M1_CURRENT_FEEDRATE']
        available_ts_features = [f for f in key_ts_features if f in features]
        
        if not available_ts_features:
            print("No time series features available")
            return
            
        # Sample one experiment from each condition
        unworn_exp = self.sampled_data[self.sampled_data['tool_wear'] == 1]['experiment_id'].iloc[0]
        worn_exp = self.sampled_data[self.sampled_data['tool_wear'] == 0]['experiment_id'].iloc[0]
        
        fig, axes = plt.subplots(len(available_ts_features), 1, figsize=(15, 4*len(available_ts_features)))
        if len(available_ts_features) == 1:
            axes = [axes]
            
        for i, feature in enumerate(available_ts_features):
            # Get data for both conditions
            unworn_data = self.sampled_data[
                (self.sampled_data['experiment_id'] == unworn_exp) & 
                (self.sampled_data['tool_wear'] == 1)
            ][feature].values
            
            worn_data = self.sampled_data[
                (self.sampled_data['experiment_id'] == worn_exp) & 
                (self.sampled_data['tool_wear'] == 0)
            ][feature].values
            
            # Plot time series
            axes[i].plot(unworn_data, label='Unworn Tool', alpha=0.7)
            axes[i].plot(worn_data, label='Worn Tool', alpha=0.7)
            axes[i].set_title(f'{feature} - Time Series Comparison')
            axes[i].set_xlabel('Time Steps')
            axes[i].set_ylabel(feature)
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)
            
        plt.tight_layout()
        plt.savefig('time_series_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    def create_summary_statistics(self, features):
        """Create summary statistics table"""
        print("\nCreating summary statistics...")
        
        # Group by tool wear condition
        stats = self.sampled_data.groupby('tool_wear')[features].agg(['mean', 'std', 'min', 'max'])
        
        # Flatten column names
        stats.columns = [f'{col[0]}_{col[1]}' for col in stats.columns]
        
        # Save to CSV
        stats.to_csv('tool_wear_statistics.csv')
        
        print("Summary statistics saved to 'tool_wear_statistics.csv'")
        
        # Display top features statistics
        top_features = ['X1_CurrentFeedback', 'Y1_CurrentFeedback', 'X1_DCBusVoltage', 'M1_CURRENT_FEEDRATE']
        available_top = [f for f in top_features if f in features]
        
        if available_top:
            print(f"\nSummary statistics for key features:")
            key_stats = stats[[f'{f}_mean' for f in available_top]]
            print(key_stats)
            
        return stats
        
    def run_complete_analysis(self):
        """Run the complete analysis pipeline"""
        print("=== CNC MILL TOOL WEAR ANALYSIS ===\n")
        
        # Load and sample data
        self.load_experiments()
        self.sample_data(samples_per_experiment=300)
        self.create_tool_wear_labels()
        
        # Select features
        features = self.select_key_features()
        
        # Run analyses
        corr_matrix = self.create_correlation_matrix(features)
        rf_model, feature_importance = self.create_feature_importance_analysis(features)
        roc_auc = self.create_roc_analysis(rf_model, features)
        self.create_distribution_plots(feature_importance)
        self.create_time_series_analysis(features)
        stats = self.create_summary_statistics(features)
        
        print("\n=== ANALYSIS COMPLETE ===")
        print(f"ROC AUC Score: {roc_auc:.3f}")
        print("All plots saved as PNG files")
        print("Summary statistics saved as CSV")

if __name__ == "__main__":
    # Create analyzer and run analysis
    analyzer = ToolWearAnalyzer()
    analyzer.run_complete_analysis() 