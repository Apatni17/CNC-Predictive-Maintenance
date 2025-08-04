import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

class MachineCompletionAnalyzer:
    def __init__(self, data_dir="data/CNC mill wear /"):
        self.data_dir = data_dir
        self.combined_data = None
        self.sampled_data = None
        
    def load_and_prepare_data(self):
        """Load data and create machine completion labels"""
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
        
        # Create machine completion labels based on process completion
        self.create_completion_labels()
        
    def create_completion_labels(self):
        """Create machine completion success labels"""
        # Define completion success based on process indicators
        # We'll use a combination of factors to determine if a process completed successfully
        
        # 1. Check if the process reached completion (based on machining process)
        self.sampled_data['process_completed'] = (
            self.sampled_data['Machining_Process'].notna() & 
            (self.sampled_data['Machining_Process'] != '')
        ).astype(int)
        
        # 2. Check for successful position attainment (actual vs command)
        position_tolerance = 0.1  # 0.1mm tolerance
        x_position_success = (abs(self.sampled_data['X1_ActualPosition'] - self.sampled_data['X1_CommandPosition']) < position_tolerance).astype(int)
        y_position_success = (abs(self.sampled_data['Y1_ActualPosition'] - self.sampled_data['Y1_CommandPosition']) < position_tolerance).astype(int)
        z_position_success = (abs(self.sampled_data['Z1_ActualPosition'] - self.sampled_data['Z1_CommandPosition']) < position_tolerance).astype(int)
        
        self.sampled_data['position_success'] = ((x_position_success + y_position_success + z_position_success) >= 2).astype(int)
        
        # 3. Check for stable current feedback (no excessive spikes)
        current_threshold = self.sampled_data['X1_CurrentFeedback'].quantile(0.95)
        self.sampled_data['current_stable'] = (
            self.sampled_data['X1_CurrentFeedback'] < current_threshold
        ).astype(int)
        
        # 4. Check for successful velocity attainment
        velocity_tolerance = 0.05  # 5% tolerance
        self.sampled_data['velocity_success'] = (
            abs(self.sampled_data['X1_ActualVelocity'] - self.sampled_data['X1_CommandVelocity']) / 
            (self.sampled_data['X1_CommandVelocity'] + 1e-6) < velocity_tolerance
        ).astype(int)
        
        # Combine all success criteria (at least 3 out of 4 criteria must be met)
        success_sum = (self.sampled_data['process_completed'] + 
                      self.sampled_data['position_success'] + 
                      self.sampled_data['current_stable'] + 
                      self.sampled_data['velocity_success'])
        
        self.sampled_data['completion_success'] = (success_sum >= 3).astype(int)
        
        print(f"Completion success rate: {self.sampled_data['completion_success'].mean():.2%}")
        
    def analyze_velocity_acceleration_impact(self):
        """Analyze how spindle and axis velocities/accelerations influence completion rates"""
        print("\n=== Analyzing Velocity/Acceleration Impact on Completion ===")
        
        # Select velocity and acceleration features
        velocity_features = [
            'X1_CommandVelocity', 'Y1_CommandVelocity', 'Z1_CommandVelocity', 'S1_CommandVelocity',
            'X1_ActualVelocity', 'Y1_ActualVelocity', 'Z1_ActualVelocity', 'S1_ActualVelocity'
        ]
        
        # Calculate accelerations (rate of change of velocity)
        for axis in ['X1', 'Y1', 'Z1', 'S1']:
            self.sampled_data[f'{axis}_Acceleration'] = self.sampled_data[f'{axis}_ActualVelocity'].diff()
        
        acceleration_features = ['X1_Acceleration', 'Y1_Acceleration', 'Z1_Acceleration', 'S1_Acceleration']
        
        # Remove NaN values from acceleration calculation
        self.sampled_data = self.sampled_data.dropna(subset=acceleration_features)
        
        # Analyze completion rates by velocity ranges
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.ravel()
        
        velocity_analysis = {}
        
        for i, feature in enumerate(velocity_features[:4]):  # Plot first 4 features
            if feature in self.sampled_data.columns:
                # Create velocity bins
                bins = pd.cut(self.sampled_data[feature], bins=5)
                completion_by_velocity = self.sampled_data.groupby(bins)['completion_success'].mean()
                
                # Plot
                completion_by_velocity.plot(kind='bar', ax=axes[i], color='skyblue', alpha=0.7)
                axes[i].set_title(f'Completion Rate by {feature}')
                axes[i].set_xlabel(feature)
                axes[i].set_ylabel('Completion Success Rate')
                axes[i].tick_params(axis='x', rotation=45)
                
                # Store analysis results
                velocity_analysis[feature] = completion_by_velocity
                
                # Calculate correlation
                correlation = self.sampled_data[feature].corr(self.sampled_data['completion_success'])
                print(f"{feature} correlation with completion: {correlation:.3f}")
        
        plt.tight_layout()
        plt.savefig('velocity_completion_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Analyze acceleration impact
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.ravel()
        
        acceleration_analysis = {}
        
        for i, feature in enumerate(acceleration_features):
            if feature in self.sampled_data.columns:
                # Create acceleration bins
                bins = pd.cut(self.sampled_data[feature], bins=5)
                completion_by_accel = self.sampled_data.groupby(bins)['completion_success'].mean()
                
                # Plot
                completion_by_accel.plot(kind='bar', ax=axes[i], color='lightcoral', alpha=0.7)
                axes[i].set_title(f'Completion Rate by {feature}')
                axes[i].set_xlabel(feature)
                axes[i].set_ylabel('Completion Success Rate')
                axes[i].tick_params(axis='x', rotation=45)
                
                # Store analysis results
                acceleration_analysis[feature] = completion_by_accel
                
                # Calculate correlation
                correlation = self.sampled_data[feature].corr(self.sampled_data['completion_success'])
                print(f"{feature} correlation with completion: {correlation:.3f}")
        
        plt.tight_layout()
        plt.savefig('acceleration_completion_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        return velocity_analysis, acceleration_analysis
    
    def analyze_cutting_forces_impact(self):
        """Analyze how cutting forces (current feedback, output power/current) explain completion differences"""
        print("\n=== Analyzing Cutting Forces Impact on Completion ===")
        
        # Select cutting force related features
        cutting_force_features = [
            'X1_CurrentFeedback', 'Y1_CurrentFeedback', 'Z1_CurrentFeedback', 'S1_CurrentFeedback',
            'X1_OutputCurrent', 'Y1_OutputCurrent', 'Z1_OutputCurrent', 'S1_OutputCurrent',
            'X1_OutputPower', 'Y1_OutputPower', 'Z1_OutputPower', 'S1_OutputPower'
        ]
        
        # Filter to available features
        available_features = [f for f in cutting_force_features if f in self.sampled_data.columns]
        
        # Create correlation analysis
        correlations = {}
        for feature in available_features:
            correlation = self.sampled_data[feature].corr(self.sampled_data['completion_success'])
            correlations[feature] = correlation
            print(f"{feature} correlation with completion: {correlation:.3f}")
        
        # Plot cutting force distributions by completion status
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.ravel()
        
        for i, feature in enumerate(available_features[:4]):  # Plot first 4 features
            # Create violin plot
            sns.violinplot(data=self.sampled_data, x='completion_success', y=feature, ax=axes[i])
            axes[i].set_title(f'{feature} Distribution by Completion Status')
            axes[i].set_xlabel('Completion Success (0=Failed, 1=Success)')
            axes[i].set_ylabel(feature)
        
        plt.tight_layout()
        plt.savefig('cutting_forces_completion_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Create scatter plot matrix for top cutting force features
        top_features = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)[:4]
        top_feature_names = [f[0] for f in top_features]
        
        if len(top_feature_names) >= 2:
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            axes = axes.ravel()
            
            for i, feature in enumerate(top_feature_names[:4]):
                # Scatter plot with completion status
                success_data = self.sampled_data[self.sampled_data['completion_success'] == 1]
                failed_data = self.sampled_data[self.sampled_data['completion_success'] == 0]
                
                axes[i].scatter(success_data.index, success_data[feature], 
                              alpha=0.6, color='green', label='Success', s=20)
                axes[i].scatter(failed_data.index, failed_data[feature], 
                              alpha=0.6, color='red', label='Failed', s=20)
                axes[i].set_title(f'{feature} vs Completion Status')
                axes[i].set_xlabel('Data Point Index')
                axes[i].set_ylabel(feature)
                axes[i].legend()
                axes[i].grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig('cutting_forces_scatter_analysis.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        # Machine learning analysis for cutting forces
        print("\n--- Machine Learning Analysis for Cutting Forces ---")
        
        # Prepare features for ML
        X = self.sampled_data[available_features]
        y = self.sampled_data['completion_success']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        
        # Train Random Forest
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X_train, y_train)
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': available_features,
            'importance': rf.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\nTop cutting force features for completion prediction:")
        print(feature_importance.head(10))
        
        # Plot feature importance
        plt.figure(figsize=(12, 8))
        top_10_features = feature_importance.head(10)
        plt.barh(range(len(top_10_features)), top_10_features['importance'])
        plt.yticks(range(len(top_10_features)), top_10_features['feature'])
        plt.xlabel('Feature Importance')
        plt.title('Cutting Force Features Importance for Completion Prediction')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig('cutting_forces_importance.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Model performance
        y_pred = rf.predict(X_test)
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        return correlations, feature_importance
    
    def create_summary_statistics(self):
        """Create summary statistics for completion analysis"""
        print("\n=== Creating Summary Statistics ===")
        
        # Group by completion status
        completion_stats = self.sampled_data.groupby('completion_success').agg({
            'X1_CommandVelocity': ['mean', 'std'],
            'Y1_CommandVelocity': ['mean', 'std'],
            'S1_CommandVelocity': ['mean', 'std'],
            'X1_CurrentFeedback': ['mean', 'std'],
            'Y1_CurrentFeedback': ['mean', 'std'],
            'X1_OutputCurrent': ['mean', 'std'],
            'Y1_OutputCurrent': ['mean', 'std']
        }).round(3)
        
        # Save statistics
        completion_stats.to_csv('machine_completion_statistics.csv')
        print("Saved machine completion statistics to 'machine_completion_statistics.csv'")
        
        return completion_stats
    
    def run_complete_analysis(self):
        """Run the complete machine completion analysis"""
        print("🚀 Starting Machine Completion Analysis...")
        
        # Load and prepare data
        self.load_and_prepare_data()
        
        # Analyze velocity/acceleration impact
        velocity_analysis, acceleration_analysis = self.analyze_velocity_acceleration_impact()
        
        # Analyze cutting forces impact
        correlations, feature_importance = self.analyze_cutting_forces_impact()
        
        # Create summary statistics
        completion_stats = self.create_summary_statistics()
        
        print("\n✅ Machine Completion Analysis Complete!")
        print("\nGenerated files:")
        print("- velocity_completion_analysis.png")
        print("- acceleration_completion_analysis.png") 
        print("- cutting_forces_completion_analysis.png")
        print("- cutting_forces_scatter_analysis.png")
        print("- cutting_forces_importance.png")
        print("- machine_completion_statistics.csv")
        
        return {
            'velocity_analysis': velocity_analysis,
            'acceleration_analysis': acceleration_analysis,
            'correlations': correlations,
            'feature_importance': feature_importance,
            'completion_stats': completion_stats
        }

if __name__ == "__main__":
    analyzer = MachineCompletionAnalyzer()
    results = analyzer.run_complete_analysis() 