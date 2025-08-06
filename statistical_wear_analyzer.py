import pandas as pd
import numpy as np
from scipy.spatial.distance import mahalanobis
from scipy.stats import chi2
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class StatisticalWearAnalyzer:
    def __init__(self):
        self.fresh_experiments = [1, 2, 3, 4, 5, 11, 12, 17]
        self.worn_experiments = [6, 7, 8, 9, 10, 13, 14, 15, 16, 18]
        
        # Key features for wear analysis
        self.wear_features = [
            'X1_CurrentFeedback', 'X1_DCBusVoltage', 'M1_CURRENT_FEEDRATE', 
            'X1_OutputPower', 'X1_ActualVelocity', 'X1_OutputCurrent',
            'Y1_CurrentFeedback', 'Z1_CurrentFeedback', 'X1_ActualPosition', 
            'X1_CommandVelocity', 'S1_CurrentFeedback', 'X1_OutputVoltage'
        ]
        
    def load_and_analyze_experiments(self, data_dir="data/CNC mill wear /"):
        """Load experiments and perform statistical analysis"""
        print("🔍 Statistical Wear Analysis")
        print("=" * 50)
        
        fresh_data = []
        worn_data = []
        
        # Load fresh tool experiments
        for exp_id in self.fresh_experiments:
            filepath = Path(data_dir) / f"experiment_{exp_id:02d}.csv"
            if filepath.exists():
                data = pd.read_csv(filepath)
                data = data[self.wear_features].fillna(0)
                data['experiment_id'] = exp_id
                data['tool_condition'] = 'fresh'
                fresh_data.append(data)
                print(f"✅ Loaded fresh experiment {exp_id}: {len(data)} samples")
        
        # Load worn tool experiments  
        for exp_id in self.worn_experiments:
            filepath = Path(data_dir) / f"experiment_{exp_id:02d}.csv"
            if filepath.exists():
                data = pd.read_csv(filepath)
                data = data[self.wear_features].fillna(0)
                data['experiment_id'] = exp_id
                data['tool_condition'] = 'worn'
                worn_data.append(data)
                print(f"✅ Loaded worn experiment {exp_id}: {len(data)} samples")
        
        self.fresh_df = pd.concat(fresh_data, ignore_index=True)
        self.worn_df = pd.concat(worn_data, ignore_index=True)
        
        print(f"\n📊 Data Summary:")
        print(f"   Fresh tool samples: {len(self.fresh_df):,}")
        print(f"   Worn tool samples: {len(self.worn_df):,}")
        
        return self.fresh_df, self.worn_df
    
    def calculate_statistical_distributions(self):
        """Calculate statistical distributions for fresh vs worn tools"""
        print(f"\n📈 Statistical Distribution Analysis:")
        print("=" * 40)
        
        # Calculate means and covariances
        fresh_features = self.fresh_df[self.wear_features]
        worn_features = self.worn_df[self.wear_features]
        
        self.fresh_mean = fresh_features.mean()
        self.fresh_cov = fresh_features.cov()
        self.fresh_std = fresh_features.std()
        
        self.worn_mean = worn_features.mean()
        self.worn_cov = worn_features.cov()
        self.worn_std = worn_features.std()
        
        print("Fresh Tool Statistics:")
        for feature in self.wear_features[:5]:  # Show top 5
            print(f"   {feature}: μ={self.fresh_mean[feature]:.3f}, σ={self.fresh_std[feature]:.3f}")
        
        print("\nWorn Tool Statistics:")
        for feature in self.wear_features[:5]:  # Show top 5
            print(f"   {feature}: μ={self.worn_mean[feature]:.3f}, σ={self.worn_std[feature]:.3f}")
        
        return {
            'fresh_mean': self.fresh_mean,
            'fresh_cov': self.fresh_cov,
            'worn_mean': self.worn_mean, 
            'worn_cov': self.worn_cov
        }
    
    def calculate_mahalanobis_thresholds(self, confidence_level=0.95):
        """Calculate Mahalanobis distance thresholds for anomaly detection"""
        print(f"\n🎯 Mahalanobis Distance Analysis:")
        print("=" * 35)
        
        # Calculate Mahalanobis distances for fresh tools
        fresh_features = self.fresh_df[self.wear_features].values
        fresh_mahal_distances = []
        
        try:
            fresh_cov_inv = np.linalg.pinv(self.fresh_cov)
            for sample in fresh_features:
                distance = mahalanobis(sample, self.fresh_mean, fresh_cov_inv)
                fresh_mahal_distances.append(distance)
        except np.linalg.LinAlgError:
            print("⚠️ Using pseudo-inverse for covariance matrix")
            fresh_cov_inv = np.linalg.pinv(self.fresh_cov)
            for sample in fresh_features:
                distance = mahalanobis(sample, self.fresh_mean, fresh_cov_inv)
                fresh_mahal_distances.append(distance)
        
        # Calculate threshold based on chi-squared distribution
        df = len(self.wear_features)  # degrees of freedom
        threshold = chi2.ppf(confidence_level, df)
        
        self.fresh_mahal_distances = np.array(fresh_mahal_distances)
        self.mahal_threshold = threshold
        
        # Calculate percentage of fresh samples exceeding threshold
        anomaly_rate = np.mean(self.fresh_mahal_distances > threshold) * 100
        
        print(f"   Mahalanobis threshold ({confidence_level*100}% confidence): {threshold:.3f}")
        print(f"   Fresh tool anomaly rate: {anomaly_rate:.1f}%")
        print(f"   Mean fresh Mahalanobis distance: {np.mean(self.fresh_mahal_distances):.3f}")
        print(f"   Std fresh Mahalanobis distance: {np.std(self.fresh_mahal_distances):.3f}")
        
        return threshold, self.fresh_mahal_distances
    
    def analyze_wear_progression_patterns(self):
        """Analyze how wear progresses within experiments"""
        print(f"\n🕒 Wear Progression Pattern Analysis:")
        print("=" * 40)
        
        progression_stats = {}
        
        # Analyze each experiment for temporal patterns
        for exp_id in self.fresh_experiments + self.worn_experiments:
            filepath = Path("data/CNC mill wear /") / f"experiment_{exp_id:02d}.csv"
            if filepath.exists():
                data = pd.read_csv(filepath)
                data = data[self.wear_features].fillna(0)
                
                # Calculate Mahalanobis distance progression through time
                distances = []
                fresh_cov_inv = np.linalg.pinv(self.fresh_cov)
                
                for _, row in data.iterrows():
                    distance = mahalanobis(row.values, self.fresh_mean, fresh_cov_inv)
                    distances.append(distance)
                
                # Analyze progression pattern
                distances = np.array(distances)
                start_distance = np.mean(distances[:50])  # First 50 samples
                end_distance = np.mean(distances[-50:])   # Last 50 samples
                progression_rate = (end_distance - start_distance) / len(distances)
                
                tool_type = 'fresh' if exp_id in self.fresh_experiments else 'worn'
                progression_stats[exp_id] = {
                    'tool_type': tool_type,
                    'start_distance': start_distance,
                    'end_distance': end_distance,
                    'progression_rate': progression_rate,
                    'total_change': end_distance - start_distance
                }
                
                print(f"   Exp {exp_id} ({tool_type}): {start_distance:.2f} → {end_distance:.2f} (Δ{end_distance-start_distance:+.2f})")
        
        self.progression_stats = progression_stats
        return progression_stats
    
    def determine_statistical_wear_ranges(self):
        """Determine wear ranges based on statistical analysis"""
        print(f"\n📊 Statistical Wear Range Determination:")
        print("=" * 45)
        
        # Analyze progression patterns
        fresh_stats = {k: v for k, v in self.progression_stats.items() if v['tool_type'] == 'fresh'}
        worn_stats = {k: v for k, v in self.progression_stats.items() if v['tool_type'] == 'worn'}
        
        # Calculate statistical ranges
        fresh_start_distances = [v['start_distance'] for v in fresh_stats.values()]
        fresh_end_distances = [v['end_distance'] for v in fresh_stats.values()]
        
        worn_start_distances = [v['start_distance'] for v in worn_stats.values()]
        worn_end_distances = [v['end_distance'] for v in worn_stats.values()]
        
        # Convert to 0-1 scale based on maximum observed distance
        max_distance = max(
            max(fresh_end_distances) if fresh_end_distances else 0,
            max(worn_end_distances) if worn_end_distances else 0
        )
        
        # Statistical wear ranges
        fresh_start_range = (np.mean(fresh_start_distances) / max_distance, 
                           np.std(fresh_start_distances) / max_distance)
        fresh_end_range = (np.mean(fresh_end_distances) / max_distance,
                         np.std(fresh_end_distances) / max_distance)
        
        worn_start_range = (np.mean(worn_start_distances) / max_distance,
                          np.std(worn_start_distances) / max_distance)
        worn_end_range = (np.mean(worn_end_distances) / max_distance,
                        np.std(worn_end_distances) / max_distance)
        
        print(f"📈 Fresh Tool Statistical Ranges:")
        print(f"   Start: {fresh_start_range[0]:.1%} ± {fresh_start_range[1]:.1%}")
        print(f"   End: {fresh_end_range[0]:.1%} ± {fresh_end_range[1]:.1%}")
        
        print(f"\n📈 Worn Tool Statistical Ranges:")
        print(f"   Start: {worn_start_range[0]:.1%} ± {worn_start_range[1]:.1%}")
        print(f"   End: {worn_end_range[0]:.1%} ± {worn_end_range[1]:.1%}")
        
        # Recommend ranges
        recommended_ranges = {
            'fresh_start': max(0.0, fresh_start_range[0] - fresh_start_range[1]),
            'fresh_end': min(1.0, fresh_end_range[0] + fresh_end_range[1]),
            'worn_start': max(0.0, worn_start_range[0] - worn_start_range[1]),
            'worn_end': min(1.0, worn_end_range[0] + worn_end_range[1])
        }
        
        print(f"\n🎯 Recommended Statistical Ranges:")
        print(f"   Fresh Tools: {recommended_ranges['fresh_start']:.1%} → {recommended_ranges['fresh_end']:.1%}")
        print(f"   Worn Tools: {recommended_ranges['worn_start']:.1%} → {recommended_ranges['worn_end']:.1%}")
        
        return recommended_ranges
    
    def create_statistical_visualizations(self):
        """Create visualizations of statistical analysis"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Feature distributions comparison
        feature_to_plot = 'M1_CURRENT_FEEDRATE'
        axes[0,0].hist(self.fresh_df[feature_to_plot], bins=50, alpha=0.7, label='Fresh Tools', color='green')
        axes[0,0].hist(self.worn_df[feature_to_plot], bins=50, alpha=0.7, label='Worn Tools', color='red')
        axes[0,0].set_xlabel(feature_to_plot)
        axes[0,0].set_ylabel('Frequency')
        axes[0,0].set_title('Feature Distribution: Fresh vs Worn Tools')
        axes[0,0].legend()
        axes[0,0].grid(True, alpha=0.3)
        
        # 2. Mahalanobis distance distribution
        axes[0,1].hist(self.fresh_mahal_distances, bins=50, alpha=0.7, color='blue')
        axes[0,1].axvline(self.mahal_threshold, color='red', linestyle='--', label=f'Threshold: {self.mahal_threshold:.2f}')
        axes[0,1].set_xlabel('Mahalanobis Distance')
        axes[0,1].set_ylabel('Frequency') 
        axes[0,1].set_title('Mahalanobis Distance Distribution (Fresh Tools)')
        axes[0,1].legend()
        axes[0,1].grid(True, alpha=0.3)
        
        # 3. Progression patterns
        for exp_id, stats in list(self.progression_stats.items())[:6]:  # Plot first 6
            color = 'green' if stats['tool_type'] == 'fresh' else 'red'
            axes[1,0].scatter(exp_id, stats['total_change'], color=color, s=60, alpha=0.7)
        
        axes[1,0].set_xlabel('Experiment ID')
        axes[1,0].set_ylabel('Mahalanobis Distance Change')
        axes[1,0].set_title('Wear Progression Patterns by Experiment')
        axes[1,0].grid(True, alpha=0.3)
        
        # 4. Statistical summary
        summary_text = f"""
        🔍 Statistical Wear Analysis Summary
        
        📊 Data:
        • Fresh tool samples: {len(self.fresh_df):,}
        • Worn tool samples: {len(self.worn_df):,}
        • Features analyzed: {len(self.wear_features)}
        
        🎯 Mahalanobis Analysis:
        • Threshold (95% confidence): {self.mahal_threshold:.3f}
        • Mean fresh distance: {np.mean(self.fresh_mahal_distances):.3f}
        • Anomaly detection ready!
        
        🕒 Approach:
        • Data-driven thresholds
        • Multivariate analysis
        • Statistical foundation
        """
        
        axes[1,1].text(0.1, 0.5, summary_text, transform=axes[1,1].transAxes,
                      fontsize=10, verticalalignment='center',
                      bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
        axes[1,1].set_title('Statistical Analysis Summary')
        axes[1,1].axis('off')
        
        plt.tight_layout()
        plt.savefig('statistical_wear_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("📊 Statistical analysis visualization saved as 'statistical_wear_analysis.png'")

def main():
    print("🧠 Statistical Wear Analysis with Mahalanobis Distance")
    print("=" * 60)
    
    analyzer = StatisticalWearAnalyzer()
    
    # Load and analyze data
    fresh_df, worn_df = analyzer.load_and_analyze_experiments()
    
    # Calculate statistical distributions
    stats = analyzer.calculate_statistical_distributions()
    
    # Calculate Mahalanobis thresholds
    threshold, distances = analyzer.calculate_mahalanobis_thresholds()
    
    # Analyze wear progression patterns
    progression_stats = analyzer.analyze_wear_progression_patterns()
    
    # Determine statistical wear ranges
    recommended_ranges = analyzer.determine_statistical_wear_ranges()
    
    # Create visualizations
    analyzer.create_statistical_visualizations()
    
    print(f"\n✅ Statistical analysis complete!")
    print(f"🎯 Use these data-driven ranges instead of assumptions!")
    
    return analyzer, recommended_ranges

if __name__ == "__main__":
    analyzer, ranges = main()