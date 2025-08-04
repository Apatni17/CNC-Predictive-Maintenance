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

def create_visualizations(comparison_df, combined_data, unworn_data, worn_data):
    """Create comprehensive visualizations"""
    
    # Set up the plotting
    fig = plt.figure(figsize=(20, 24))
    
    # 1. Current Feedback Comparison - Time Series
    plt.subplot(4, 3, 1)
    current_cols = ['X1_CurrentFeedback', 'Y1_CurrentFeedback', 'Z1_CurrentFeedback', 'S1_CurrentFeedback']
    
    for i, col in enumerate(current_cols):
        plt.plot(unworn_data[col].iloc[:100], label=f'{col} (Unworn)', alpha=0.7)
        plt.plot(worn_data[col].iloc[:100], label=f'{col} (Worn)', alpha=0.7)
    
    plt.title('Current Feedback Comparison (First 100 samples)')
    plt.xlabel('Sample Index')
    plt.ylabel('Current Feedback')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    
    # 2. Current Feedback Distribution
    plt.subplot(4, 3, 2)
    for col in current_cols:
        plt.hist(unworn_data[col], alpha=0.5, label=f'{col} (Unworn)', bins=30)
        plt.hist(worn_data[col], alpha=0.5, label=f'{col} (Worn)', bins=30)
    
    plt.title('Current Feedback Distribution')
    plt.xlabel('Current Feedback')
    plt.ylabel('Frequency')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # 3. Power Output Comparison
    plt.subplot(4, 3, 3)
    power_cols = ['X1_OutputPower', 'Y1_OutputPower', 'S1_OutputPower']
    
    for col in power_cols:
        if col in unworn_data.columns:
            plt.plot(unworn_data[col].iloc[:100], label=f'{col} (Unworn)', alpha=0.7)
            plt.plot(worn_data[col].iloc[:100], label=f'{col} (Worn)', alpha=0.7)
    
    plt.title('Output Power Comparison (First 100 samples)')
    plt.xlabel('Sample Index')
    plt.ylabel('Output Power')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    
    # 4. Box Plot - Current Feedback by Tool Condition
    plt.subplot(4, 3, 4)
    current_data = []
    labels = []
    
    for col in current_cols:
        current_data.extend([unworn_data[col], worn_data[col]])
        labels.extend([f'{col}_Unworn', f'{col}_Worn'])
    
    plt.boxplot(current_data, labels=labels)
    plt.title('Current Feedback Distribution by Tool Condition')
    plt.ylabel('Current Feedback')
    plt.xticks(rotation=45)
    
    # 5. Statistical Summary Heatmap
    plt.subplot(4, 3, 5)
    
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
                center=0)
    plt.title('Mean Difference: Worn - Unworn')
    
    # 6. Correlation Heatmap - Current vs Power
    plt.subplot(4, 3, 6)
    correlation_cols = current_cols + power_cols
    correlation_data = combined_data[correlation_cols].corr()
    sns.heatmap(correlation_data, annot=True, cmap='coolwarm', center=0)
    plt.title('Correlation Matrix: Current vs Power')
    
    # 7. Statistical Test Results
    plt.subplot(4, 3, 7)
    
    # Perform t-tests
    test_results = []
    test_features = []
    
    for col in current_cols + power_cols:
        if col in unworn_data.columns:
            t_stat, p_value = stats.ttest_ind(unworn_data[col], worn_data[col])
            test_results.append(p_value)
            test_features.append(col)
    
    # Plot p-values
    plt.bar(range(len(test_results)), test_results)
    plt.axhline(y=0.05, color='r', linestyle='--', label='p=0.05')
    plt.title('Statistical Significance (p-values)')
    plt.xlabel('Features')
    plt.ylabel('p-value')
    plt.xticks(range(len(test_features)), test_features, rotation=45)
    plt.legend()
    
    # 8. Feature Importance (based on effect size)
    plt.subplot(4, 3, 8)
    
    effect_sizes = []
    for col in current_cols + power_cols:
        if col in unworn_data.columns:
            # Calculate Cohen's d
            pooled_std = np.sqrt(((unworn_data[col].std()**2 + worn_data[col].std()**2)) / 2)
            cohens_d = (worn_data[col].mean() - unworn_data[col].mean()) / pooled_std
            effect_sizes.append(abs(cohens_d))
    
    plt.bar(range(len(effect_sizes)), effect_sizes)
    plt.title('Effect Size (|Cohen\'s d|)')
    plt.xlabel('Features')
    plt.ylabel('|Cohen\'s d|')
    plt.xticks(range(len(test_features)), test_features, rotation=45)
    
    # 9. Time Series Analysis - Moving Average
    plt.subplot(4, 3, 9)
    
    # Calculate moving average for current feedback
    window_size = 20
    unworn_ma = unworn_data['X1_CurrentFeedback'].rolling(window=window_size).mean()
    worn_ma = worn_data['X1_CurrentFeedback'].rolling(window=window_size).mean()
    
    plt.plot(unworn_ma.iloc[:200], label='Unworn Tool', alpha=0.8)
    plt.plot(worn_ma.iloc[:200], label='Worn Tool', alpha=0.8)
    plt.title(f'Moving Average Current Feedback (Window={window_size})')
    plt.xlabel('Sample Index')
    plt.ylabel('Current Feedback (MA)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 10. Variance Comparison
    plt.subplot(4, 3, 10)
    
    variances_unworn = [unworn_data[col].var() for col in current_cols]
    variances_worn = [worn_data[col].var() for col in current_cols]
    
    x = np.arange(len(current_cols))
    width = 0.35
    
    plt.bar(x - width/2, variances_unworn, width, label='Unworn', alpha=0.8)
    plt.bar(x + width/2, variances_worn, width, label='Worn', alpha=0.8)
    
    plt.title('Variance Comparison by Axis')
    plt.xlabel('Axes')
    plt.ylabel('Variance')
    plt.xticks(x, ['X1', 'Y1', 'Z1', 'S1'])
    plt.legend()
    
    # 11. Peak Analysis
    plt.subplot(4, 3, 11)
    
    # Find peaks in current feedback
    from scipy.signal import find_peaks
    
    unworn_peaks, _ = find_peaks(unworn_data['X1_CurrentFeedback'].iloc[:200], height=0)
    worn_peaks, _ = find_peaks(worn_data['X1_CurrentFeedback'].iloc[:200], height=0)
    
    plt.plot(unworn_data['X1_CurrentFeedback'].iloc[:200], label='Unworn', alpha=0.7)
    plt.plot(worn_data['X1_CurrentFeedback'].iloc[:200], label='Worn', alpha=0.7)
    plt.plot(unworn_peaks, unworn_data['X1_CurrentFeedback'].iloc[:200].iloc[unworn_peaks], 'ro', label='Unworn Peaks')
    plt.plot(worn_peaks, worn_data['X1_CurrentFeedback'].iloc[:200].iloc[worn_peaks], 'bo', label='Worn Peaks')
    
    plt.title('Peak Analysis - Current Feedback')
    plt.xlabel('Sample Index')
    plt.ylabel('Current Feedback')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 12. Summary Statistics Table
    plt.subplot(4, 3, 12)
    plt.axis('off')
    
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
    table.set_fontsize(8)
    table.scale(1, 2)
    plt.title('Summary Statistics', pad=20)
    
    plt.tight_layout()
    plt.savefig('tool_wear_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return comparison_df

def print_analysis_summary(comparison_df):
    """Print key findings from the analysis"""
    
    print("\n" + "="*80)
    print("TOOL WEAR ANALYSIS SUMMARY")
    print("="*80)
    
    # Key findings
    print("\nKEY FINDINGS:")
    print("-" * 40)
    
    # Current feedback analysis
    current_features = [col for col in comparison_df.columns if 'CurrentFeedback' in col and 'mean' in col]
    
    print("\n1. CURRENT FEEDBACK ANALYSIS:")
    for feature in current_features:
        unworn_mean = comparison_df[comparison_df['Tool_Condition'] == 'Unworn'][feature].mean()
        worn_mean = comparison_df[comparison_df['Tool_Condition'] == 'Worn'][feature].mean()
        diff = worn_mean - unworn_mean
        change_pct = (diff / abs(unworn_mean)) * 100 if unworn_mean != 0 else 0
        
        print(f"   {feature}:")
        print(f"     Unworn: {unworn_mean:.4f}")
        print(f"     Worn: {worn_mean:.4f}")
        print(f"     Difference: {diff:.4f} ({change_pct:+.1f}%)")
    
    # Power analysis
    power_features = [col for col in comparison_df.columns if 'OutputPower' in col and 'mean' in col]
    
    print("\n2. POWER CONSUMPTION ANALYSIS:")
    for feature in power_features:
        unworn_mean = comparison_df[comparison_df['Tool_Condition'] == 'Unworn'][feature].mean()
        worn_mean = comparison_df[comparison_df['Tool_Condition'] == 'Worn'][feature].mean()
        diff = worn_mean - unworn_mean
        change_pct = (diff / abs(unworn_mean)) * 100 if unworn_mean != 0 else 0
        
        print(f"   {feature}:")
        print(f"     Unworn: {unworn_mean:.6f}")
        print(f"     Worn: {worn_mean:.6f}")
        print(f"     Difference: {diff:.6f} ({change_pct:+.1f}%)")
    
    print("\n3. RECOMMENDATIONS FOR PREDICTIVE MAINTENANCE:")
    print("-" * 50)
    print("• Monitor current feedback values for significant increases")
    print("• Track power consumption patterns for early wear detection")
    print("• Set thresholds based on statistical analysis results")
    print("• Implement real-time monitoring of key sensor readings")
    print("• Use moving averages to smooth out noise in sensor data")

if __name__ == "__main__":
    print("Starting Tool Wear Analysis...")
    
    # Create comparison dataset
    comparison_df, combined_data, unworn_data, worn_data = create_comparison_dataset()
    
    # Create visualizations
    print("Creating visualizations...")
    comparison_df = create_visualizations(comparison_df, combined_data, unworn_data, worn_data)
    
    # Print analysis summary
    print_analysis_summary(comparison_df)
    
    print("\nAnalysis complete! Check 'tool_wear_analysis.png' for visualizations.") 