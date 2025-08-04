import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def load_experiment_data(exp_num, n_samples=300):
    """Load experiment data and take n_samples"""
    file_path = f"data/CNC mill wear /experiment_{exp_num:02d}.csv"
    df = pd.read_csv(file_path)
    
    # Take n_samples from the middle of the dataset to avoid startup/shutdown effects
    start_idx = len(df) // 2 - n_samples // 2
    end_idx = start_idx + n_samples
    
    return df.iloc[start_idx:end_idx].copy()

def extract_current_power_features(df):
    """Extract current and power related features"""
    features = {}
    
    # Current feedback for all axes
    current_cols = ['X1_CurrentFeedback', 'Y1_CurrentFeedback', 'Z1_CurrentFeedback', 'S1_CurrentFeedback']
    for col in current_cols:
        features[f'{col}_mean'] = df[col].mean()
        features[f'{col}_std'] = df[col].std()
        features[f'{col}_max'] = df[col].max()
        features[f'{col}_min'] = df[col].min()
        features[f'{col}_range'] = df[col].max() - df[col].min()
    
    # Output power for available axes
    power_cols = ['X1_OutputPower', 'Y1_OutputPower', 'S1_OutputPower']
    for col in power_cols:
        if col in df.columns:
            features[f'{col}_mean'] = df[col].mean()
            features[f'{col}_std'] = df[col].std()
            features[f'{col}_max'] = df[col].max()
            features[f'{col}_min'] = df[col].min()
    
    # Output current and voltage
    current_output_cols = ['X1_OutputCurrent', 'Y1_OutputCurrent', 'S1_OutputCurrent']
    voltage_cols = ['X1_OutputVoltage', 'Y1_OutputVoltage', 'S1_OutputVoltage']
    
    for col in current_output_cols:
        if col in df.columns:
            features[f'{col}_mean'] = df[col].mean()
            features[f'{col}_std'] = df[col].std()
    
    for col in voltage_cols:
        if col in df.columns:
            features[f'{col}_mean'] = df[col].mean()
            features[f'{col}_std'] = df[col].std()
    
    return features

def create_comparison_dataset():
    """Create dataset comparing unworn vs worn tools"""
    
    # Load unworn tool experiments (1 and 2)
    print("Loading unworn tool experiments...")
    exp1_unworn = load_experiment_data(1, 300)
    exp2_unworn = load_experiment_data(2, 300)
    
    # Load worn tool experiments (7 and 8)
    print("Loading worn tool experiments...")
    exp7_worn = load_experiment_data(7, 300)
    exp8_worn = load_experiment_data(8, 300)
    
    # Extract features
    print("Extracting features...")
    features_unworn_1 = extract_current_power_features(exp1_unworn)
    features_unworn_2 = extract_current_power_features(exp2_unworn)
    features_worn_7 = extract_current_power_features(exp7_worn)
    features_worn_8 = extract_current_power_features(exp8_worn)
    
    # Create comparison dataframe
    comparison_data = {
        'Experiment': ['Exp1_Unworn', 'Exp2_Unworn', 'Exp7_Worn', 'Exp8_Worn'],
        'Tool_Condition': ['Unworn', 'Unworn', 'Worn', 'Worn'],
        **{k: [features_unworn_1[k], features_unworn_2[k], features_worn_7[k], features_worn_8[k]] 
           for k in features_unworn_1.keys()}
    }
    
    comparison_df = pd.DataFrame(comparison_data)
    
    # Create detailed datasets for visualization
    unworn_data = pd.concat([exp1_unworn, exp2_unworn], ignore_index=True)
    worn_data = pd.concat([exp7_worn, exp8_worn], ignore_index=True)
    
    # Add tool condition labels
    unworn_data['Tool_Condition'] = 'Unworn'
    worn_data['Tool_Condition'] = 'Worn'
    
    # Combine for analysis
    combined_data = pd.concat([unworn_data, worn_data], ignore_index=True)
    
    return comparison_df, combined_data, unworn_data, worn_data

def create_time_series_plot(unworn_data, worn_data):
    """Create time series comparison plot"""
    plt.figure(figsize=(12, 8))
    
    current_cols = ['X1_CurrentFeedback', 'Y1_CurrentFeedback', 'Z1_CurrentFeedback', 'S1_CurrentFeedback']
    
    for i, col in enumerate(current_cols):
        plt.plot(unworn_data[col].iloc[:100], label=f'{col} (Unworn)', alpha=0.7, linewidth=2)
        plt.plot(worn_data[col].iloc[:100], label=f'{col} (Worn)', alpha=0.7, linewidth=2)
    
    plt.title('Current Feedback Time Series Comparison', fontsize=16, fontweight='bold')
    plt.xlabel('Sample Index', fontsize=12)
    plt.ylabel('Current Feedback', fontsize=12)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('tab1_time_series.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_distribution_plot(unworn_data, worn_data):
    """Create distribution comparison plot"""
    plt.figure(figsize=(12, 8))
    
    current_cols = ['X1_CurrentFeedback', 'Y1_CurrentFeedback', 'Z1_CurrentFeedback', 'S1_CurrentFeedback']
    
    for col in current_cols:
        plt.hist(unworn_data[col], alpha=0.6, label=f'{col} (Unworn)', bins=30, density=True)
        plt.hist(worn_data[col], alpha=0.6, label=f'{col} (Worn)', bins=30, density=True)
    
    plt.title('Current Feedback Distribution Comparison', fontsize=16, fontweight='bold')
    plt.xlabel('Current Feedback', fontsize=12)
    plt.ylabel('Density', fontsize=12)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    plt.tight_layout()
    plt.savefig('tab2_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_power_comparison_plot(unworn_data, worn_data):
    """Create power comparison plot"""
    plt.figure(figsize=(12, 8))
    
    power_cols = ['X1_OutputPower', 'Y1_OutputPower', 'S1_OutputPower']
    
    for col in power_cols:
        if col in unworn_data.columns:
            plt.plot(unworn_data[col].iloc[:100], label=f'{col} (Unworn)', alpha=0.7, linewidth=2)
            plt.plot(worn_data[col].iloc[:100], label=f'{col} (Worn)', alpha=0.7, linewidth=2)
    
    plt.title('Output Power Time Series Comparison', fontsize=16, fontweight='bold')
    plt.xlabel('Sample Index', fontsize=12)
    plt.ylabel('Output Power', fontsize=12)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('tab3_power_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_box_plot(unworn_data, worn_data):
    """Create box plot comparison"""
    plt.figure(figsize=(14, 8))
    
    current_cols = ['X1_CurrentFeedback', 'Y1_CurrentFeedback', 'Z1_CurrentFeedback', 'S1_CurrentFeedback']
    
    # Prepare data for box plot
    data_to_plot = []
    labels = []
    
    for col in current_cols:
        data_to_plot.extend([unworn_data[col], worn_data[col]])
        labels.extend([f'{col}_Unworn', f'{col}_Worn'])
    
    plt.boxplot(data_to_plot, labels=labels)
    plt.title('Current Feedback Distribution by Tool Condition', fontsize=16, fontweight='bold')
    plt.ylabel('Current Feedback', fontsize=12)
    plt.xticks(rotation=45, fontsize=10)
    plt.tight_layout()
    plt.savefig('tab4_box_plot.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_statistical_heatmap(unworn_data, worn_data):
    """Create statistical summary heatmap"""
    plt.figure(figsize=(10, 6))
    
    current_cols = ['X1_CurrentFeedback', 'Y1_CurrentFeedback', 'Z1_CurrentFeedback', 'S1_CurrentFeedback']
    power_cols = ['X1_OutputPower', 'Y1_OutputPower', 'S1_OutputPower']
    
    # Calculate mean differences
    mean_diff_data = []
    feature_names = []
    
    for col in current_cols + power_cols:
        if col in unworn_data.columns:
            unworn_mean = unworn_data[col].mean()
            worn_mean = worn_data[col].mean()
            diff = worn_mean - unworn_mean
            mean_diff_data.append(diff)
            feature_names.append(col)
    
    # Create heatmap data
    heatmap_data = np.array(mean_diff_data).reshape(1, -1)
    sns.heatmap(heatmap_data, 
                xticklabels=feature_names, 
                yticklabels=['Worn - Unworn'],
                annot=True, 
                fmt='.2e',
                cmap='RdBu_r',
                center=0,
                cbar_kws={'label': 'Difference'})
    plt.title('Mean Difference: Worn - Unworn Tools', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('tab5_statistical_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_correlation_heatmap(combined_data):
    """Create correlation heatmap"""
    plt.figure(figsize=(10, 8))
    
    current_cols = ['X1_CurrentFeedback', 'Y1_CurrentFeedback', 'Z1_CurrentFeedback', 'S1_CurrentFeedback']
    power_cols = ['X1_OutputPower', 'Y1_OutputPower', 'S1_OutputPower']
    correlation_cols = current_cols + power_cols
    
    correlation_data = combined_data[correlation_cols].corr()
    sns.heatmap(correlation_data, annot=True, cmap='coolwarm', center=0, fmt='.2f')
    plt.title('Correlation Matrix: Current vs Power', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('tab6_correlation_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_statistical_tests_plot(unworn_data, worn_data):
    """Create statistical significance plot"""
    plt.figure(figsize=(12, 8))
    
    current_cols = ['X1_CurrentFeedback', 'Y1_CurrentFeedback', 'Z1_CurrentFeedback', 'S1_CurrentFeedback']
    power_cols = ['X1_OutputPower', 'Y1_OutputPower', 'S1_OutputPower']
    
    # Perform t-tests
    test_results = []
    test_features = []
    
    for col in current_cols + power_cols:
        if col in unworn_data.columns:
            t_stat, p_value = stats.ttest_ind(unworn_data[col], worn_data[col])
            test_results.append(p_value)
            test_features.append(col)
    
    # Plot p-values
    bars = plt.bar(range(len(test_results)), test_results, color=['red' if p < 0.05 else 'blue' for p in test_results])
    plt.axhline(y=0.05, color='r', linestyle='--', label='p=0.05 (Significance Threshold)', linewidth=2)
    plt.title('Statistical Significance Test Results', fontsize=16, fontweight='bold')
    plt.xlabel('Features', fontsize=12)
    plt.ylabel('p-value', fontsize=12)
    plt.xticks(range(len(test_features)), test_features, rotation=45, fontsize=10)
    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig('tab7_statistical_tests.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_effect_size_plot(unworn_data, worn_data):
    """Create effect size plot"""
    plt.figure(figsize=(12, 8))
    
    current_cols = ['X1_CurrentFeedback', 'Y1_CurrentFeedback', 'Z1_CurrentFeedback', 'S1_CurrentFeedback']
    power_cols = ['X1_OutputPower', 'Y1_OutputPower', 'S1_OutputPower']
    
    effect_sizes = []
    feature_names = []
    
    for col in current_cols + power_cols:
        if col in unworn_data.columns:
            # Calculate Cohen's d
            pooled_std = np.sqrt(((unworn_data[col].std()**2 + worn_data[col].std()**2)) / 2)
            cohens_d = (worn_data[col].mean() - unworn_data[col].mean()) / pooled_std
            effect_sizes.append(abs(cohens_d))
            feature_names.append(col)
    
    bars = plt.bar(range(len(effect_sizes)), effect_sizes, 
                   color=['red' if d > 0.8 else 'orange' if d > 0.5 else 'blue' for d in effect_sizes])
    plt.title('Effect Size Analysis (|Cohen\'s d|)', fontsize=16, fontweight='bold')
    plt.xlabel('Features', fontsize=12)
    plt.ylabel('|Cohen\'s d|', fontsize=12)
    plt.xticks(range(len(feature_names)), feature_names, rotation=45, fontsize=10)
    plt.axhline(y=0.8, color='red', linestyle='--', label='Large Effect (>0.8)', alpha=0.7)
    plt.axhline(y=0.5, color='orange', linestyle='--', label='Medium Effect (>0.5)', alpha=0.7)
    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig('tab8_effect_size.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_moving_average_plot(unworn_data, worn_data):
    """Create moving average plot"""
    plt.figure(figsize=(12, 8))
    
    # Calculate moving average for current feedback
    window_size = 20
    unworn_ma = unworn_data['X1_CurrentFeedback'].rolling(window=window_size).mean()
    worn_ma = worn_data['X1_CurrentFeedback'].rolling(window=window_size).mean()
    
    plt.plot(unworn_ma.iloc[:200], label='Unworn Tool', alpha=0.8, linewidth=2, color='blue')
    plt.plot(worn_ma.iloc[:200], label='Worn Tool', alpha=0.8, linewidth=2, color='red')
    plt.title(f'Moving Average Current Feedback (Window={window_size})', fontsize=16, fontweight='bold')
    plt.xlabel('Sample Index', fontsize=12)
    plt.ylabel('Current Feedback (MA)', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('tab9_moving_average.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_variance_comparison_plot(unworn_data, worn_data):
    """Create variance comparison plot"""
    plt.figure(figsize=(10, 8))
    
    current_cols = ['X1_CurrentFeedback', 'Y1_CurrentFeedback', 'Z1_CurrentFeedback', 'S1_CurrentFeedback']
    
    variances_unworn = [unworn_data[col].var() for col in current_cols]
    variances_worn = [worn_data[col].var() for col in current_cols]
    
    x = np.arange(len(current_cols))
    width = 0.35
    
    plt.bar(x - width/2, variances_unworn, width, label='Unworn', alpha=0.8, color='blue')
    plt.bar(x + width/2, variances_worn, width, label='Worn', alpha=0.8, color='red')
    
    plt.title('Variance Comparison by Axis', fontsize=16, fontweight='bold')
    plt.xlabel('Axes', fontsize=12)
    plt.ylabel('Variance', fontsize=12)
    plt.xticks(x, ['X1', 'Y1', 'Z1', 'S1'], fontsize=12)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig('tab10_variance_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_summary_statistics_table(unworn_data, worn_data):
    """Create summary statistics table"""
    plt.figure(figsize=(12, 8))
    plt.axis('off')
    
    current_cols = ['X1_CurrentFeedback', 'Y1_CurrentFeedback', 'Z1_CurrentFeedback', 'S1_CurrentFeedback']
    
    # Create summary table
    summary_stats = []
    for col in current_cols:
        unworn_stats = unworn_data[col].describe()
        worn_stats = worn_data[col].describe()
        
        summary_stats.append([
            col,
            f"{unworn_stats['mean']:.3f}",
            f"{worn_stats['mean']:.3f}",
            f"{worn_stats['mean'] - unworn_stats['mean']:.3f}",
            f"{worn_stats['std'] / unworn_stats['std']:.2f}"
        ])
    
    table_data = [['Feature', 'Unworn Mean', 'Worn Mean', 'Difference', 'Std Ratio']] + summary_stats
    table = plt.table(cellText=table_data, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    plt.title('Summary Statistics Comparison', fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig('tab11_summary_statistics.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_all_visualizations():
    """Create all visualization tabs"""
    print("Creating individual visualization tabs...")
    
    # Load data
    comparison_df, combined_data, unworn_data, worn_data = create_comparison_dataset()
    
    # Create individual plots
    create_time_series_plot(unworn_data, worn_data)
    create_distribution_plot(unworn_data, worn_data)
    create_power_comparison_plot(unworn_data, worn_data)
    create_box_plot(unworn_data, worn_data)
    create_statistical_heatmap(unworn_data, worn_data)
    create_correlation_heatmap(combined_data)
    create_statistical_tests_plot(unworn_data, worn_data)
    create_effect_size_plot(unworn_data, worn_data)
    create_moving_average_plot(unworn_data, worn_data)
    create_variance_comparison_plot(unworn_data, worn_data)
    create_summary_statistics_table(unworn_data, worn_data)
    
    print("All visualization tabs created successfully!")
    print("\nGenerated files:")
    print("- tab1_time_series.png")
    print("- tab2_distribution.png")
    print("- tab3_power_comparison.png")
    print("- tab4_box_plot.png")
    print("- tab5_statistical_heatmap.png")
    print("- tab6_correlation_heatmap.png")
    print("- tab7_statistical_tests.png")
    print("- tab8_effect_size.png")
    print("- tab9_moving_average.png")
    print("- tab10_variance_comparison.png")
    print("- tab11_summary_statistics.png")
    
    return comparison_df

if __name__ == "__main__":
    print("Starting Tool Wear Analysis with Tabbed Visualizations...")
    
    # Create all visualizations
    comparison_df = create_all_visualizations()
    
    print("\nAnalysis complete! All visualization tabs have been created.") 