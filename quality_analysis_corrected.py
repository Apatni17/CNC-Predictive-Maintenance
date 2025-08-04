import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import warnings
warnings.filterwarnings('ignore')

class QualityAnalyzerCorrected:
    def __init__(self, data_dir="data/CNC mill wear /"):
        self.data_dir = data_dir
        self.sampled_data = None
        
    def load_and_prepare_data(self):
        """Load and prepare data from multiple experiments"""
        print("Loading CNC mill data...")
        
        all_data = []
        experiments_to_sample = [1, 2, 3, 9, 10, 11]  # 3 unworn (1-3), 3 worn (9-11)
        
        for exp_id in experiments_to_sample:
            filename = f"experiment_{exp_id:02d}.csv"
            filepath = f"{self.data_dir}{filename}"
            
            try:
                data = pd.read_csv(filepath)
                data['experiment_id'] = exp_id
                print(f"Loaded {filename}")
                all_data.append(data)
            except FileNotFoundError:
                print(f"Warning: {filepath} not found")
                continue
        
        if not all_data:
            raise ValueError("No data files found!")
        
        # Combine all data
        combined_data = pd.concat(all_data, ignore_index=True)
        print(f"Combined data shape: {combined_data.shape}")
        
        # Sample 300 data points from each experiment
        sampled_data = []
        for exp_id in experiments_to_sample:
            exp_data = combined_data[combined_data['experiment_id'] == exp_id]
            if len(exp_data) >= 300:
                sampled = exp_data.sample(n=300, random_state=42)
            else:
                sampled = exp_data  # Use all data if less than 300
            sampled_data.append(sampled)
        
        self.sampled_data = pd.concat(sampled_data, ignore_index=True)
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
        print(f"Process completion rate: {self.sampled_data['process_completed'].mean():.2%}")
        
    def analyze_tool_condition_impact(self):
        """Analyze how tool condition affects machine finalization"""
        print("\n=== Analyzing Tool Condition Impact ===")
        
        # Create summary table
        summary_data = []
        
        for condition in [0, 1]:  # 0 = unworn, 1 = worn
            condition_data = self.sampled_data[self.sampled_data['tool_condition'] == condition]
            
            finalization_rate = condition_data['machine_finalization'].mean()
            process_completion_rate = condition_data['process_completed'].mean()
            
            summary_data.append({
                'Tool Condition': 'Unworn' if condition == 0 else 'Worn',
                'Machine Finalization Rate': f"{finalization_rate:.2%}",
                'Process Completion Rate': f"{process_completion_rate:.2%}",
                'Sample Size': len(condition_data)
            })
        
        summary_df = pd.DataFrame(summary_data)
        print("\nTool Condition Impact Summary:")
        print(summary_df)
        
        # Create visualization
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Machine finalization by tool condition
        finalization_by_condition = self.sampled_data.groupby('tool_condition')['machine_finalization'].mean()
        axes[0].bar(['Unworn', 'Worn'], finalization_by_condition.values, color=['#2ecc71', '#e74c3c'], alpha=0.7)
        axes[0].set_title('Machine Finalization Rate by Tool Condition')
        axes[0].set_ylabel('Finalization Success Rate')
        axes[0].set_ylim(0, 1)
        
        # Add percentage labels
        for i, v in enumerate(finalization_by_condition.values):
            axes[0].text(i, v + 0.01, f'{v:.1%}', ha='center', va='bottom', fontweight='bold')
        
        # Process completion by tool condition
        completion_by_condition = self.sampled_data.groupby('tool_condition')['process_completed'].mean()
        axes[1].bar(['Unworn', 'Worn'], completion_by_condition.values, color=['#3498db', '#f39c12'], alpha=0.7)
        axes[1].set_title('Process Completion Rate by Tool Condition')
        axes[1].set_ylabel('Process Completion Rate')
        axes[1].set_ylim(0, 1)
        
        # Add percentage labels
        for i, v in enumerate(completion_by_condition.values):
            axes[1].text(i, v + 0.01, f'{v:.1%}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('tool_condition_impact_corrected.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        return summary_df
    
    def run_complete_analysis(self):
        """Run the complete corrected quality analysis"""
        print("🚀 Starting Corrected Quality Analysis...")
        
        # Load and prepare data
        self.load_and_prepare_data()
        
        # Analyze tool condition impact
        self.analyze_tool_condition_impact()
        
        print("\n✅ Corrected Quality Analysis Complete!")
        print("\nKey Findings:")
        print("- Machine finalization is now based on actual 'End'/'end' completion indicators")
        print("- Process completion rates show the true completion status")
        print("- Tool condition impact analysis provides accurate insights")

if __name__ == "__main__":
    analyzer = QualityAnalyzerCorrected()
    analyzer.run_complete_analysis() 