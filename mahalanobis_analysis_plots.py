import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from statistical_wear_analyzer import StatisticalWearAnalyzer
import warnings
warnings.filterwarnings('ignore')

class MahalanobisAnalysisPlots:
    def __init__(self):
        self.analyzer = StatisticalWearAnalyzer()
        self.threshold = 21.026  # 95% confidence threshold
        self.fresh_experiments = [1, 2, 3, 4, 5, 11, 12, 17]
        self.worn_experiments = [6, 7, 8, 9, 10, 13, 14, 15, 16, 18]
        
    def analyze_all_experiments(self):
        """Analyze Mahalanobis distances for all experiments"""
        print("🔍 Analyzing Mahalanobis Distances for All Experiments")
        print("=" * 60)
        
        # Load baseline statistics
        fresh_df, worn_df = self.analyzer.load_and_analyze_experiments()
        self.analyzer.calculate_statistical_distributions()
        self.analyzer.calculate_mahalanobis_thresholds()
        
        experiment_results = {}
        
        # Analyze each experiment
        for exp_id in range(1, 19):
            filepath = Path("data/CNC mill wear /") / f"experiment_{exp_id:02d}.csv"
            if filepath.exists():
                print(f"📊 Processing experiment {exp_id}...")
                
                data = pd.read_csv(filepath)
                data = data[self.analyzer.wear_features].fillna(0)
                
                # Calculate Mahalanobis distances
                distances = []
                for _, row in data.iterrows():
                    try:
                        distance = self.analyzer.calculate_mahalanobis_distance(row.values, 
                                                                               self.analyzer.fresh_mean.values, 
                                                                               self.analyzer.fresh_cov_inv)
                        distances.append(distance)
                    except:
                        distances.append(0)
                
                distances = np.array(distances)
                
                # Calculate metrics
                max_distance = np.max(distances)
                threshold_crossings = np.where(distances > self.threshold)[0]
                
                if len(threshold_crossings) > 0:
                    first_crossing_pct = (threshold_crossings[0] / len(distances)) * 100
                else:
                    first_crossing_pct = None
                
                tool_type = 'fresh' if exp_id in self.fresh_experiments else 'worn'
                
                experiment_results[exp_id] = {
                    'tool_type': tool_type,
                    'max_distance': max_distance,
                    'distances': distances,
                    'first_crossing_pct': first_crossing_pct,
                    'total_crossings': len(threshold_crossings),
                    'progress_pct': np.linspace(0, 100, len(distances))
                }
                
                print(f"   Max distance: {max_distance:.2f}")
                print(f"   First crossing: {first_crossing_pct:.1f}%" if first_crossing_pct else "   No crossings")
        
        return experiment_results
    
    def create_analysis_plots(self, experiment_results):
        """Create the three requested analysis plots"""
        print(f"\n📈 Creating Mahalanobis Analysis Plots...")
        
        # Set up the plotting style
        plt.style.use('default')
        sns.set_palette("husl")
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Plot 1: % into experiment when threshold was reached vs Experiment Number
        exp_numbers = []
        crossing_percentages = []
        tool_types = []
        
        for exp_id, results in experiment_results.items():
            if results['first_crossing_pct'] is not None:
                exp_numbers.append(exp_id)
                crossing_percentages.append(results['first_crossing_pct'])
                tool_types.append(results['tool_type'])
        
        # Create scatter plot with different colors for fresh vs worn
        fresh_mask = np.array(tool_types) == 'fresh'
        worn_mask = np.array(tool_types) == 'worn'
        
        axes[0, 0].scatter(np.array(exp_numbers)[fresh_mask], np.array(crossing_percentages)[fresh_mask], 
                          c='green', s=80, alpha=0.7, label='Fresh Tools', marker='o')
        axes[0, 0].scatter(np.array(exp_numbers)[worn_mask], np.array(crossing_percentages)[worn_mask], 
                          c='red', s=80, alpha=0.7, label='Worn Tools', marker='s')
        
        axes[0, 0].set_xlabel('Experiment Number')
        axes[0, 0].set_ylabel('% Into Experiment When Threshold Reached')
        axes[0, 0].set_title('When Do Tools First Exceed Mahalanobis Threshold?')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].legend()
        axes[0, 0].set_ylim(0, 100)
        
        # Plot 2: Maximum Mahalanobis distance vs Experiment Number
        all_exp_numbers = list(experiment_results.keys())
        max_distances = [experiment_results[exp]['max_distance'] for exp in all_exp_numbers]
        all_tool_types = [experiment_results[exp]['tool_type'] for exp in all_exp_numbers]
        
        fresh_mask_all = np.array(all_tool_types) == 'fresh'
        worn_mask_all = np.array(all_tool_types) == 'worn'
        
        axes[0, 1].scatter(np.array(all_exp_numbers)[fresh_mask_all], np.array(max_distances)[fresh_mask_all], 
                          c='green', s=80, alpha=0.7, label='Fresh Tools', marker='o')
        axes[0, 1].scatter(np.array(all_exp_numbers)[worn_mask_all], np.array(max_distances)[worn_mask_all], 
                          c='red', s=80, alpha=0.7, label='Worn Tools', marker='s')
        
        # Add threshold line
        axes[0, 1].axhline(y=self.threshold, color='orange', linestyle='--', linewidth=2, 
                          label=f'Threshold ({self.threshold:.1f})')
        
        axes[0, 1].set_xlabel('Experiment Number')
        axes[0, 1].set_ylabel('Maximum Mahalanobis Distance')
        axes[0, 1].set_title('Peak Mahalanobis Distance by Experiment')
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].legend()
        
        # Plot 3: Mahalanobis distance vs % Progress - Fresh Tools
        axes[1, 0].set_title('Mahalanobis Distance vs Progress - Fresh Tools')
        fresh_colors = plt.cm.Greens(np.linspace(0.4, 0.9, len(self.fresh_experiments)))
        
        for i, exp_id in enumerate(self.fresh_experiments):
            if exp_id in experiment_results:
                results = experiment_results[exp_id]
                # Sample data to reduce plot density
                sample_indices = np.linspace(0, len(results['distances'])-1, 200, dtype=int)
                sampled_progress = results['progress_pct'][sample_indices]
                sampled_distances = results['distances'][sample_indices]
                
                axes[1, 0].plot(sampled_progress, sampled_distances, 
                               color=fresh_colors[i], alpha=0.7, linewidth=1.5, 
                               label=f'Exp {exp_id}')
        
        axes[1, 0].axhline(y=self.threshold, color='red', linestyle='--', linewidth=2, 
                          label=f'Threshold ({self.threshold:.1f})')
        axes[1, 0].set_xlabel('% Progress Through Experiment')
        axes[1, 0].set_ylabel('Mahalanobis Distance')
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Plot 4: Mahalanobis distance vs % Progress - Worn Tools
        axes[1, 1].set_title('Mahalanobis Distance vs Progress - Worn Tools')
        worn_colors = plt.cm.Reds(np.linspace(0.4, 0.9, len(self.worn_experiments)))
        
        for i, exp_id in enumerate(self.worn_experiments):
            if exp_id in experiment_results:
                results = experiment_results[exp_id]
                # Sample data to reduce plot density
                sample_indices = np.linspace(0, len(results['distances'])-1, 200, dtype=int)
                sampled_progress = results['progress_pct'][sample_indices]
                sampled_distances = results['distances'][sample_indices]
                
                axes[1, 1].plot(sampled_progress, sampled_distances, 
                               color=worn_colors[i], alpha=0.7, linewidth=1.5, 
                               label=f'Exp {exp_id}')
        
        axes[1, 1].axhline(y=self.threshold, color='red', linestyle='--', linewidth=2, 
                          label=f'Threshold ({self.threshold:.1f})')
        axes[1, 1].set_xlabel('% Progress Through Experiment')
        axes[1, 1].set_ylabel('Mahalanobis Distance')
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        plt.savefig('mahalanobis_comprehensive_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("📊 Comprehensive analysis plot saved as 'mahalanobis_comprehensive_analysis.png'")
        
    def create_summary_statistics(self, experiment_results):
        """Create summary statistics table"""
        print(f"\n📋 MAHALANOBIS DISTANCE ANALYSIS SUMMARY")
        print("=" * 50)
        
        fresh_results = {k: v for k, v in experiment_results.items() if v['tool_type'] == 'fresh'}
        worn_results = {k: v for k, v in experiment_results.items() if v['tool_type'] == 'worn'}
        
        # Fresh tools statistics
        fresh_max_distances = [v['max_distance'] for v in fresh_results.values()]
        fresh_crossings = [v['first_crossing_pct'] for v in fresh_results.values() if v['first_crossing_pct'] is not None]
        
        print(f"\n🟢 FRESH TOOLS:")
        print(f"   Average max distance: {np.mean(fresh_max_distances):.2f}")
        print(f"   Max distance range: {np.min(fresh_max_distances):.2f} - {np.max(fresh_max_distances):.2f}")
        print(f"   Tools exceeding threshold: {len(fresh_crossings)}/{len(fresh_results)}")
        if fresh_crossings:
            print(f"   Average crossing point: {np.mean(fresh_crossings):.1f}% into experiment")
        
        # Worn tools statistics
        worn_max_distances = [v['max_distance'] for v in worn_results.values()]
        worn_crossings = [v['first_crossing_pct'] for v in worn_results.values() if v['first_crossing_pct'] is not None]
        
        print(f"\n🔴 WORN TOOLS:")
        print(f"   Average max distance: {np.mean(worn_max_distances):.2f}")
        print(f"   Max distance range: {np.min(worn_max_distances):.2f} - {np.max(worn_max_distances):.2f}")
        print(f"   Tools exceeding threshold: {len(worn_crossings)}/{len(worn_results)}")
        if worn_crossings:
            print(f"   Average crossing point: {np.mean(worn_crossings):.1f}% into experiment")
        
        # Key insights
        print(f"\n💡 KEY INSIGHTS:")
        
        # Which experiments never cross threshold
        no_crossing_experiments = [exp for exp, res in experiment_results.items() 
                                 if res['first_crossing_pct'] is None]
        if no_crossing_experiments:
            print(f"   Experiments never exceeding threshold: {no_crossing_experiments}")
        
        # Early vs late crossings
        all_crossings = [v['first_crossing_pct'] for v in experiment_results.values() 
                        if v['first_crossing_pct'] is not None]
        if all_crossings:
            early_crossings = [x for x in all_crossings if x < 25]
            late_crossings = [x for x in all_crossings if x > 75]
            print(f"   Early crossings (<25%): {len(early_crossings)} experiments")
            print(f"   Late crossings (>75%): {len(late_crossings)} experiments")
        
        return experiment_results

def main():
    print("🎯 Mahalanobis Distance Comprehensive Analysis")
    print("=" * 55)
    
    analyzer = MahalanobisAnalysisPlots()
    
    # Analyze all experiments
    results = analyzer.analyze_all_experiments()
    
    # Create plots
    analyzer.create_analysis_plots(results)
    
    # Generate summary statistics
    analyzer.create_summary_statistics(results)
    
    print(f"\n✅ Analysis Complete!")
    print(f"📊 Check 'mahalanobis_comprehensive_analysis.png' for visual insights!")

if __name__ == "__main__":
    main()