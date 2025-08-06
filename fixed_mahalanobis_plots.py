import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy.spatial.distance import mahalanobis
import warnings
warnings.filterwarnings('ignore')

def create_mahalanobis_plots():
    """Create comprehensive Mahalanobis distance analysis plots"""
    print("🎯 Fixed Mahalanobis Distance Analysis")
    print("=" * 45)
    
    # Define experiment categories
    fresh_experiments = [1, 2, 3, 4, 5, 11, 12, 17]
    worn_experiments = [6, 7, 8, 9, 10, 13, 14, 15, 16, 18]
    threshold = 21.026
    
    # Key features for analysis
    base_features = [
        'X1_CurrentFeedback', 'X1_DCBusVoltage', 'M1_CURRENT_FEEDRATE', 'X1_OutputPower',
        'X1_ActualVelocity', 'X1_OutputCurrent', 'Y1_CurrentFeedback', 'Z1_CurrentFeedback',
        'X1_ActualPosition', 'X1_CommandVelocity', 'S1_CurrentFeedback', 'X1_OutputVoltage'
    ]
    
    # Build baseline from fresh tools (first 4 experiments)
    print("📊 Building baseline from fresh experiments 1-4...")
    baseline_data = []
    
    for exp_id in [1, 2, 3, 4]:
        filepath = Path("data/CNC mill wear /") / f"experiment_{exp_id:02d}.csv"
        if filepath.exists():
            data = pd.read_csv(filepath)
            # Sample to make computation manageable
            if len(data) > 300:
                data = data.sample(n=300, random_state=42)
            data = data[base_features].fillna(0)
            baseline_data.append(data)
            print(f"   ✅ Loaded experiment {exp_id}: {len(data)} samples")
    
    # Combine baseline data
    baseline_df = pd.concat(baseline_data, ignore_index=True)
    baseline_mean = baseline_df.mean()
    baseline_cov = baseline_df.cov()
    
    # Ensure covariance matrix is invertible
    try:
        baseline_cov_inv = np.linalg.inv(baseline_cov.values)
        print(f"✅ Baseline established: {len(baseline_df)} samples")
    except np.linalg.LinAlgError:
        # Use pseudo-inverse if singular
        baseline_cov_inv = np.linalg.pinv(baseline_cov.values)
        print(f"⚠️  Using pseudo-inverse for covariance matrix")
    
    # Analyze all experiments
    experiment_results = {}
    
    print(f"\n🔍 Analyzing Mahalanobis distances for all experiments...")
    
    for exp_id in range(1, 19):
        filepath = Path("data/CNC mill wear /") / f"experiment_{exp_id:02d}.csv"
        if not filepath.exists():
            continue
            
        print(f"📊 Processing experiment {exp_id}...")
        data = pd.read_csv(filepath)
        data = data[base_features].fillna(0)
        
        # Calculate Mahalanobis distances for each row
        distances = []
        for idx, row in data.iterrows():
            try:
                # Calculate Mahalanobis distance using scipy
                distance = mahalanobis(row.values, baseline_mean.values, baseline_cov_inv)
                distances.append(distance)
            except Exception as e:
                distances.append(0.0)
        
        distances = np.array(distances)
        
        # Find threshold crossings
        threshold_crossings = np.where(distances > threshold)[0]
        max_distance = np.max(distances) if len(distances) > 0 else 0.0
        
        # Calculate percentage into experiment when first crossing occurs
        if len(threshold_crossings) > 0:
            first_crossing_pct = (threshold_crossings[0] / len(distances)) * 100
        else:
            first_crossing_pct = None
        
        # Determine tool type
        tool_type = 'fresh' if exp_id in fresh_experiments else 'worn'
        
        # Store results
        experiment_results[exp_id] = {
            'tool_type': tool_type,
            'max_distance': max_distance,
            'distances': distances,
            'first_crossing_pct': first_crossing_pct,
            'total_crossings': len(threshold_crossings),
            'progress_pct': np.linspace(0, 100, len(distances))
        }
        
        print(f"   Max distance: {max_distance:.2f}")
        if first_crossing_pct is not None:
            print(f"   First crossing: {first_crossing_pct:.1f}%")
        else:
            print(f"   No threshold crossings")
    
    # Create the plots
    print(f"\n📈 Creating analysis plots...")
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Mahalanobis Distance Comprehensive Analysis', fontsize=16, fontweight='bold')
    
    # Plot 1: % into experiment when threshold was reached vs Experiment Number
    exp_numbers = []
    crossing_percentages = []
    tool_types = []
    
    for exp_id, results in experiment_results.items():
        if results['first_crossing_pct'] is not None:
            exp_numbers.append(exp_id)
            crossing_percentages.append(results['first_crossing_pct'])
            tool_types.append(results['tool_type'])
    
    if len(exp_numbers) > 0:
        fresh_mask = np.array(tool_types) == 'fresh'
        worn_mask = np.array(tool_types) == 'worn'
        
        if np.any(fresh_mask):
            axes[0, 0].scatter(np.array(exp_numbers)[fresh_mask], np.array(crossing_percentages)[fresh_mask], 
                              c='green', s=100, alpha=0.8, label='Fresh Tools', marker='o', edgecolors='darkgreen')
        if np.any(worn_mask):
            axes[0, 0].scatter(np.array(exp_numbers)[worn_mask], np.array(crossing_percentages)[worn_mask], 
                              c='red', s=100, alpha=0.8, label='Worn Tools', marker='s', edgecolors='darkred')
    
    axes[0, 0].set_xlabel('Experiment Number', fontweight='bold')
    axes[0, 0].set_ylabel('% Into Experiment When\nThreshold First Reached', fontweight='bold')
    axes[0, 0].set_title('When Do Tools First Exceed Threshold?', fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()
    axes[0, 0].set_ylim(0, 100)
    axes[0, 0].set_xlim(0, 19)
    
    # Plot 2: Maximum Mahalanobis distance vs Experiment Number  
    all_exp_numbers = list(experiment_results.keys())
    max_distances = [experiment_results[exp]['max_distance'] for exp in all_exp_numbers]
    all_tool_types = [experiment_results[exp]['tool_type'] for exp in all_exp_numbers]
    
    fresh_mask_all = np.array(all_tool_types) == 'fresh'
    worn_mask_all = np.array(all_tool_types) == 'worn'
    
    axes[0, 1].scatter(np.array(all_exp_numbers)[fresh_mask_all], np.array(max_distances)[fresh_mask_all], 
                      c='green', s=100, alpha=0.8, label='Fresh Tools', marker='o', edgecolors='darkgreen')
    axes[0, 1].scatter(np.array(all_exp_numbers)[worn_mask_all], np.array(max_distances)[worn_mask_all], 
                      c='red', s=100, alpha=0.8, label='Worn Tools', marker='s', edgecolors='darkred')
    
    # Add threshold line
    axes[0, 1].axhline(y=threshold, color='orange', linestyle='--', linewidth=3, 
                      label=f'Anomaly Threshold ({threshold:.1f})', alpha=0.8)
    
    axes[0, 1].set_xlabel('Experiment Number', fontweight='bold')
    axes[0, 1].set_ylabel('Maximum Mahalanobis Distance', fontweight='bold')
    axes[0, 1].set_title('Peak Mahalanobis Distance by Experiment', fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend()
    axes[0, 1].set_xlim(0, 19)
    
    # Plot 3: Mahalanobis distance vs % Progress - Fresh Tools
    axes[1, 0].set_title('Mahalanobis Distance vs Progress - Fresh Tools', fontweight='bold')
    fresh_colors = plt.cm.Greens(np.linspace(0.4, 0.9, len(fresh_experiments)))
    
    for i, exp_id in enumerate(fresh_experiments):
        if exp_id in experiment_results:
            results = experiment_results[exp_id]
            # Sample data points for cleaner visualization
            if len(results['distances']) > 500:
                sample_indices = np.linspace(0, len(results['distances'])-1, 500, dtype=int)
                sampled_progress = results['progress_pct'][sample_indices]
                sampled_distances = results['distances'][sample_indices]
            else:
                sampled_progress = results['progress_pct']
                sampled_distances = results['distances']
            
            axes[1, 0].plot(sampled_progress, sampled_distances, 
                           color=fresh_colors[i], alpha=0.7, linewidth=2, 
                           label=f'Exp {exp_id}')
    
    axes[1, 0].axhline(y=threshold, color='red', linestyle='--', linewidth=3, 
                      label=f'Threshold ({threshold:.1f})', alpha=0.8)
    axes[1, 0].set_xlabel('% Progress Through Experiment', fontweight='bold')
    axes[1, 0].set_ylabel('Mahalanobis Distance', fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
    
    # Plot 4: Mahalanobis distance vs % Progress - Worn Tools
    axes[1, 1].set_title('Mahalanobis Distance vs Progress - Worn Tools', fontweight='bold')
    worn_colors = plt.cm.Reds(np.linspace(0.4, 0.9, len(worn_experiments)))
    
    for i, exp_id in enumerate(worn_experiments):
        if exp_id in experiment_results:
            results = experiment_results[exp_id]
            # Sample data points for cleaner visualization
            if len(results['distances']) > 500:
                sample_indices = np.linspace(0, len(results['distances'])-1, 500, dtype=int)
                sampled_progress = results['progress_pct'][sample_indices]
                sampled_distances = results['distances'][sample_indices]
            else:
                sampled_progress = results['progress_pct']
                sampled_distances = results['distances']
            
            axes[1, 1].plot(sampled_progress, sampled_distances, 
                           color=worn_colors[i], alpha=0.7, linewidth=2, 
                           label=f'Exp {exp_id}')
    
    axes[1, 1].axhline(y=threshold, color='red', linestyle='--', linewidth=3, 
                      label=f'Threshold ({threshold:.1f})', alpha=0.8)
    axes[1, 1].set_xlabel('% Progress Through Experiment', fontweight='bold')
    axes[1, 1].set_ylabel('Mahalanobis Distance', fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('mahalanobis_comprehensive_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Print summary statistics
    print(f"\n📋 ANALYSIS SUMMARY")
    print("=" * 30)
    
    fresh_results = {k: v for k, v in experiment_results.items() if v['tool_type'] == 'fresh'}
    worn_results = {k: v for k, v in experiment_results.items() if v['tool_type'] == 'worn'}
    
    # Fresh tools statistics
    fresh_max_distances = [v['max_distance'] for v in fresh_results.values()]
    fresh_crossings = [v['first_crossing_pct'] for v in fresh_results.values() if v['first_crossing_pct'] is not None]
    
    print(f"\n🟢 FRESH TOOLS:")
    print(f"   Experiments analyzed: {len(fresh_results)}")
    print(f"   Average max distance: {np.mean(fresh_max_distances):.2f}")
    print(f"   Max distance range: {np.min(fresh_max_distances):.2f} - {np.max(fresh_max_distances):.2f}")
    print(f"   Tools exceeding threshold: {len(fresh_crossings)}/{len(fresh_results)}")
    if fresh_crossings:
        print(f"   Average crossing point: {np.mean(fresh_crossings):.1f}% into experiment")
    
    # Worn tools statistics
    worn_max_distances = [v['max_distance'] for v in worn_results.values()]
    worn_crossings = [v['first_crossing_pct'] for v in worn_results.values() if v['first_crossing_pct'] is not None]
    
    print(f"\n🔴 WORN TOOLS:")
    print(f"   Experiments analyzed: {len(worn_results)}")
    print(f"   Average max distance: {np.mean(worn_max_distances):.2f}")
    print(f"   Max distance range: {np.min(worn_max_distances):.2f} - {np.max(worn_max_distances):.2f}")
    print(f"   Tools exceeding threshold: {len(worn_crossings)}/{len(worn_results)}")
    if worn_crossings:
        print(f"   Average crossing point: {np.mean(worn_crossings):.1f}% into experiment")
    
    # Key insights
    print(f"\n💡 KEY INSIGHTS:")
    
    # Overall threshold crossing rate
    total_crossings = len([v for v in experiment_results.values() if v['first_crossing_pct'] is not None])
    print(f"   Total experiments exceeding threshold: {total_crossings}/{len(experiment_results)}")
    
    # Early vs late crossings
    all_crossings = [v['first_crossing_pct'] for v in experiment_results.values() 
                    if v['first_crossing_pct'] is not None]
    if all_crossings:
        early_crossings = [x for x in all_crossings if x < 25]
        late_crossings = [x for x in all_crossings if x > 75]
        print(f"   Early crossings (<25%): {len(early_crossings)} experiments")
        print(f"   Late crossings (>75%): {len(late_crossings)} experiments")
        print(f"   Average crossing point: {np.mean(all_crossings):.1f}%")
    
    print(f"\n✅ Analysis complete!")
    print(f"📊 Plots saved as 'mahalanobis_comprehensive_analysis.png'")
    
    return experiment_results

if __name__ == "__main__":
    results = create_mahalanobis_plots()