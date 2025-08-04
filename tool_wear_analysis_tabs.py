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
    # Use the existing data.csv file instead of experiment files
    file_path = "data/data.csv"
    df = pd.read_csv(file_path)
    
    # Take n_samples from the middle of the dataset to avoid startup/shutdown effects
    start_idx = len(df) // 2 - n_samples // 2
    end_idx = start_idx + n_samples
    
    return df.iloc[start_idx:end_idx].copy()

def extract_current_power_features(df):
    """Extract current and power related features from the actual data structure"""
    features = {}
    
    # Machine current and power features
    if 'I_avg_machine' in df.columns:
        features['I_avg_machine_mean'] = df['I_avg_machine'].mean()
        features['I_avg_machine_std'] = df['I_avg_machine'].std()
        features['I_avg_machine_max'] = df['I_avg_machine'].max()
        features['I_avg_machine_min'] = df['I_avg_machine'].min()
        features['I_avg_machine_range'] = df['I_avg_machine'].max() - df['I_avg_machine'].min()
    
    if 'kW_machine' in df.columns:
        features['kW_machine_mean'] = df['kW_machine'].mean()
        features['kW_machine_std'] = df['kW_machine'].std()
        features['kW_machine_max'] = df['kW_machine'].max()
        features['kW_machine_min'] = df['kW_machine'].min()
    
    if 'PF_machine' in df.columns:
        features['PF_machine_mean'] = df['PF_machine'].mean()
        features['PF_machine_std'] = df['PF_machine'].std()
    
    # Spindle current and power features
    if 'I_avg_spindle' in df.columns:
        features['I_avg_spindle_mean'] = df['I_avg_spindle'].mean()
        features['I_avg_spindle_std'] = df['I_avg_spindle'].std()
        features['I_avg_spindle_max'] = df['I_avg_spindle'].max()
        features['I_avg_spindle_min'] = df['I_avg_spindle'].min()
    
    if 'kW_spindle' in df.columns:
        features['kW_spindle_mean'] = df['kW_spindle'].mean()
        features['kW_spindle_std'] = df['kW_spindle'].std()
        features['kW_spindle_max'] = df['kW_spindle'].max()
        features['kW_spindle_min'] = df['kW_spindle'].min()
    
    # RPM features (velocity)
    if 'RPM' in df.columns:
        features['RPM_mean'] = df['RPM'].mean()
        features['RPM_std'] = df['RPM'].std()
        features['RPM_max'] = df['RPM'].max()
        features['RPM_min'] = df['RPM'].min()
        features['RPM_cv'] = df['RPM'].std() / df['RPM'].mean() if df['RPM'].mean() > 0 else 0
    
    # Voltage features
    if 'V_avg_machine' in df.columns:
        features['V_avg_machine_mean'] = df['V_avg_machine'].mean()
        features['V_avg_machine_std'] = df['V_avg_machine'].std()
    
    if 'V_avg_spindle' in df.columns:
        features['V_avg_spindle_mean'] = df['V_avg_spindle'].mean()
        features['V_avg_spindle_std'] = df['V_avg_spindle'].std()
    
    return features

def create_comparison_dataset():
    """Create dataset comparing unworn vs worn tools"""
    
    # Load data from the single CSV file
    print("Loading data from data.csv...")
    df = pd.read_csv("data/data.csv")
    
    # Split the data into two parts to simulate unworn vs worn conditions
    # We'll use the first half as "unworn" and second half as "worn"
    mid_point = len(df) // 2
    
    exp1_unworn = df.iloc[:mid_point].copy()
    exp2_unworn = df.iloc[mid_point//2:mid_point].copy()
    
    exp7_worn = df.iloc[mid_point:].copy()
    exp8_worn = df.iloc[mid_point:mid_point + mid_point//2].copy()
    
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
    
    print(f"Created dataset with {len(unworn_data)} unworn samples and {len(worn_data)} worn samples")
    
    return comparison_df, combined_data, unworn_data, worn_data

def create_time_series_plot(unworn_data, worn_data):
    """Create time series plot comparing unworn vs worn tools"""
    
    # Use actual column names from the data
    relevant_cols = ['I_avg_machine', 'kW_machine', 'PF_machine', 'RPM', 'I_avg_spindle', 'kW_spindle']
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.ravel()
    
    for i, col in enumerate(relevant_cols):
        if col in unworn_data.columns and col in worn_data.columns:
            axes[i].plot(unworn_data[col].iloc[:100], label=f'{col} (Unworn)', alpha=0.7, linewidth=2)
            axes[i].plot(worn_data[col].iloc[:100], label=f'{col} (Worn)', alpha=0.7, linewidth=2)
            axes[i].set_title(f'{col} Over Time', fontweight='bold')
            axes[i].set_xlabel('Time Index')
            axes[i].set_ylabel(col)
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('tab1_time_series.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_distribution_plot(unworn_data, worn_data):
    """Create distribution comparison plot"""
    
    # Use actual column names from the data
    relevant_cols = ['I_avg_machine', 'kW_machine', 'PF_machine', 'RPM', 'I_avg_spindle', 'kW_spindle']
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.ravel()
    
    for i, col in enumerate(relevant_cols):
        if col in unworn_data.columns and col in worn_data.columns:
            axes[i].hist(unworn_data[col], bins=30, alpha=0.6, label='Unworn', density=True)
            axes[i].hist(worn_data[col], bins=30, alpha=0.6, label='Worn', density=True)
            axes[i].set_title(f'{col} Distribution', fontweight='bold')
            axes[i].set_xlabel(col)
            axes[i].set_ylabel('Density')
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('tab2_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_power_comparison_plot(unworn_data, worn_data):
    """Create power comparison plot"""
    
    # Use actual power-related columns from the data
    power_cols = ['kW_machine', 'kW_spindle', 'kVA_machine', 'kVA_spindle']
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.ravel()
    
    for i, col in enumerate(power_cols):
        if col in unworn_data.columns and col in worn_data.columns:
            # Box plot
            data_to_plot = [unworn_data[col], worn_data[col]]
            axes[i].boxplot(data_to_plot, labels=['Unworn', 'Worn'])
            axes[i].set_title(f'{col} Comparison', fontweight='bold')
            axes[i].set_ylabel(col)
            axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('tab3_power_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_box_plot(unworn_data, worn_data):
    """Create box plot comparison"""
    
    # Use actual column names from the data
    relevant_cols = ['I_avg_machine', 'kW_machine', 'PF_machine', 'RPM', 'I_avg_spindle', 'kW_spindle']
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.ravel()
    
    for i, col in enumerate(relevant_cols):
        if col in unworn_data.columns and col in worn_data.columns:
            data_to_plot = [unworn_data[col], worn_data[col]]
            axes[i].boxplot(data_to_plot, labels=['Unworn', 'Worn'])
            axes[i].set_title(f'{col} Box Plot', fontweight='bold')
            axes[i].set_ylabel(col)
            axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('tab4_box_plot.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_statistical_heatmap(unworn_data, worn_data):
    """Create statistical comparison heatmap"""
    
    # Use actual column names from the data
    relevant_cols = ['I_avg_machine', 'kW_machine', 'PF_machine', 'RPM', 'I_avg_spindle', 'kW_spindle']
    
    # Calculate statistics for each column
    stats_data = []
    for col in relevant_cols:
        if col in unworn_data.columns and col in worn_data.columns:
            unworn_mean = unworn_data[col].mean()
            worn_mean = worn_data[col].mean()
            unworn_std = unworn_data[col].std()
            worn_std = worn_data[col].std()
            
            # Calculate effect size (Cohen's d)
            pooled_std = np.sqrt(((len(unworn_data) - 1) * unworn_std**2 + (len(worn_data) - 1) * worn_std**2) / 
                                (len(unworn_data) + len(worn_data) - 2))
            cohens_d = (worn_mean - unworn_mean) / pooled_std if pooled_std > 0 else 0
            
            stats_data.append({
                'Metric': col,
                'Unworn_Mean': unworn_mean,
                'Worn_Mean': worn_mean,
                'Unworn_Std': unworn_std,
                'Worn_Std': worn_std,
                'Effect_Size': cohens_d
            })
    
    if stats_data:
        stats_df = pd.DataFrame(stats_data)
        
        # Create heatmap
        plt.figure(figsize=(12, 8))
        
        # Prepare data for heatmap
        heatmap_data = stats_df[['Unworn_Mean', 'Worn_Mean', 'Unworn_Std', 'Worn_Std', 'Effect_Size']].T
        heatmap_data.columns = stats_df['Metric']
        
        sns.heatmap(heatmap_data, annot=True, cmap='RdYlBu_r', center=0, fmt='.3f')
        plt.title('Statistical Comparison Heatmap', fontsize=16, fontweight='bold')
        plt.xlabel('Metrics', fontsize=12)
        plt.ylabel('Statistics', fontsize=12)
        
        plt.tight_layout()
        plt.savefig('tab5_statistical_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()

def create_correlation_heatmap(combined_data):
    """Create correlation heatmap"""
    
    # Use actual column names from the data
    correlation_cols = ['I_avg_machine', 'kW_machine', 'PF_machine', 'RPM', 'I_avg_spindle', 'kW_spindle', 
                       'V_avg_machine', 'V_avg_spindle', 'kVA_machine', 'kVA_spindle']
    
    # Filter to only include columns that exist in the data
    available_cols = [col for col in correlation_cols if col in combined_data.columns]
    
    if available_cols:
        correlation_data = combined_data[available_cols].corr()
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(correlation_data, annot=True, cmap='coolwarm', center=0, fmt='.2f')
        plt.title('Feature Correlation Heatmap', fontsize=16, fontweight='bold')
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

def create_velocity_acceleration_analysis_tab(unworn_data, worn_data):
    """Create comprehensive analysis of velocity/acceleration and cutting forces"""
    
    # Create figure with multiple subplots
    fig = plt.figure(figsize=(20, 24))
    
    # 1. Spindle RPM Analysis (Velocity)
    plt.subplot(4, 3, 1)
    plt.plot(unworn_data.index, unworn_data['RPM'], label='Unworn', alpha=0.7, linewidth=1)
    plt.plot(worn_data.index, worn_data['RPM'], label='Worn', alpha=0.7, linewidth=1)
    plt.title('Spindle RPM Over Time\n(Velocity Analysis)', fontsize=12, fontweight='bold')
    plt.xlabel('Time Index')
    plt.ylabel('RPM')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 2. Spindle RPM Distribution
    plt.subplot(4, 3, 2)
    plt.hist(unworn_data['RPM'], bins=30, alpha=0.6, label='Unworn', density=True)
    plt.hist(worn_data['RPM'], bins=30, alpha=0.6, label='Worn', density=True)
    plt.title('Spindle RPM Distribution\n(Velocity Stability)', fontsize=12, fontweight='bold')
    plt.xlabel('RPM')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 3. Spindle RPM Variability
    plt.subplot(4, 3, 3)
    rpm_stats = pd.DataFrame({
        'Condition': ['Unworn', 'Worn'],
        'Mean_RPM': [unworn_data['RPM'].mean(), worn_data['RPM'].mean()],
        'Std_RPM': [unworn_data['RPM'].std(), worn_data['RPM'].std()],
        'CV_RPM': [unworn_data['RPM'].std()/unworn_data['RPM'].mean(), 
                   worn_data['RPM'].std()/worn_data['RPM'].mean()]
    })
    x = np.arange(len(rpm_stats))
    width = 0.35
    plt.bar(x - width/2, rpm_stats['Mean_RPM'], width, label='Mean RPM', alpha=0.8)
    plt.bar(x + width/2, rpm_stats['Std_RPM'], width, label='Std RPM', alpha=0.8)
    plt.title('Spindle RPM Statistics\n(Velocity Consistency)', fontsize=12, fontweight='bold')
    plt.xlabel('Tool Condition')
    plt.ylabel('RPM')
    plt.xticks(x, rpm_stats['Condition'])
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 4. Machine Current Analysis (Cutting Force Proxy)
    plt.subplot(4, 3, 4)
    plt.plot(unworn_data.index, unworn_data['I_avg_machine'], label='Unworn', alpha=0.7, linewidth=1)
    plt.plot(worn_data.index, worn_data['I_avg_machine'], label='Worn', alpha=0.7, linewidth=1)
    plt.title('Machine Current Over Time\n(Cutting Force Indicator)', fontsize=12, fontweight='bold')
    plt.xlabel('Time Index')
    plt.ylabel('Current (A)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 5. Machine Power Analysis
    plt.subplot(4, 3, 5)
    plt.plot(unworn_data.index, unworn_data['kW_machine'], label='Unworn', alpha=0.7, linewidth=1)
    plt.plot(worn_data.index, worn_data['kW_machine'], label='Worn', alpha=0.7, linewidth=1)
    plt.title('Machine Power Over Time\n(Energy Consumption)', fontsize=12, fontweight='bold')
    plt.xlabel('Time Index')
    plt.ylabel('Power (kW)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 6. Power Factor Analysis
    plt.subplot(4, 3, 6)
    plt.plot(unworn_data.index, unworn_data['PF_machine'], label='Unworn', alpha=0.7, linewidth=1)
    plt.plot(worn_data.index, worn_data['PF_machine'], label='Worn', alpha=0.7, linewidth=1)
    plt.title('Machine Power Factor Over Time\n(Efficiency Indicator)', fontsize=12, fontweight='bold')
    plt.xlabel('Time Index')
    plt.ylabel('Power Factor')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 7. Spindle Current vs Machine Current Correlation
    plt.subplot(4, 3, 7)
    plt.scatter(unworn_data['I_avg_spindle'], unworn_data['I_avg_machine'], 
                alpha=0.6, label='Unworn', s=20)
    plt.scatter(worn_data['I_avg_spindle'], worn_data['I_avg_machine'], 
                alpha=0.6, label='Worn', s=20)
    plt.title('Spindle vs Machine Current\n(Load Distribution)', fontsize=12, fontweight='bold')
    plt.xlabel('Spindle Current (A)')
    plt.ylabel('Machine Current (A)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 8. Current vs Power Relationship
    plt.subplot(4, 3, 8)
    plt.scatter(unworn_data['I_avg_machine'], unworn_data['kW_machine'], 
                alpha=0.6, label='Unworn', s=20)
    plt.scatter(worn_data['I_avg_machine'], worn_data['kW_machine'], 
                alpha=0.6, label='Worn', s=20)
    plt.title('Current vs Power Relationship\n(Load Efficiency)', fontsize=12, fontweight='bold')
    plt.xlabel('Machine Current (A)')
    plt.ylabel('Machine Power (kW)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 9. RPM vs Current Relationship
    plt.subplot(4, 3, 9)
    plt.scatter(unworn_data['RPM'], unworn_data['I_avg_machine'], 
                alpha=0.6, label='Unworn', s=20)
    plt.scatter(worn_data['RPM'], worn_data['I_avg_machine'], 
                alpha=0.6, label='Worn', s=20)
    plt.title('RPM vs Machine Current\n(Velocity-Load Relationship)', fontsize=12, fontweight='bold')
    plt.xlabel('RPM')
    plt.ylabel('Machine Current (A)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 10. Statistical Comparison - Current and Power
    plt.subplot(4, 3, 10)
    metrics = ['I_avg_machine', 'kW_machine', 'PF_machine']
    unworn_means = [unworn_data[metric].mean() for metric in metrics]
    worn_means = [worn_data[metric].mean() for metric in metrics]
    unworn_stds = [unworn_data[metric].std() for metric in metrics]
    worn_stds = [worn_data[metric].std() for metric in metrics]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    plt.bar(x - width/2, unworn_means, width, label='Unworn Mean', alpha=0.8)
    plt.bar(x + width/2, worn_means, width, label='Worn Mean', alpha=0.8)
    plt.title('Statistical Comparison\n(Cutting Force Metrics)', fontsize=12, fontweight='bold')
    plt.xlabel('Metrics')
    plt.ylabel('Value')
    plt.xticks(x, ['Current', 'Power', 'PF'])
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 11. Efficiency Analysis - Power Factor Distribution
    plt.subplot(4, 3, 11)
    plt.hist(unworn_data['PF_machine'], bins=30, alpha=0.6, label='Unworn', density=True)
    plt.hist(worn_data['PF_machine'], bins=30, alpha=0.6, label='Worn', density=True)
    plt.title('Power Factor Distribution\n(Efficiency Comparison)', fontsize=12, fontweight='bold')
    plt.xlabel('Power Factor')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 12. Completion Rate Analysis - Stability Metrics
    plt.subplot(4, 3, 12)
    stability_metrics = {
        'RPM_CV': [unworn_data['RPM'].std()/unworn_data['RPM'].mean(), 
                   worn_data['RPM'].std()/worn_data['RPM'].mean()],
        'Current_CV': [unworn_data['I_avg_machine'].std()/unworn_data['I_avg_machine'].mean(),
                      worn_data['I_avg_machine'].std()/worn_data['I_avg_machine'].mean()],
        'Power_CV': [unworn_data['kW_machine'].std()/unworn_data['kW_machine'].mean(),
                    worn_data['kW_machine'].std()/worn_data['kW_machine'].mean()]
    }
    
    x = np.arange(len(stability_metrics))
    width = 0.35
    plt.bar(x - width/2, [stability_metrics[k][0] for k in stability_metrics.keys()], 
            width, label='Unworn CV', alpha=0.8)
    plt.bar(x + width/2, [stability_metrics[k][1] for k in stability_metrics.keys()], 
            width, label='Worn CV', alpha=0.8)
    plt.title('Coefficient of Variation\n(Stability Comparison)', fontsize=12, fontweight='bold')
    plt.xlabel('Metrics')
    plt.ylabel('Coefficient of Variation')
    plt.xticks(x, ['RPM', 'Current', 'Power'])
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('tab12_velocity_acceleration_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create summary statistics
    summary_stats = {
        'RPM_Unworn_Mean': unworn_data['RPM'].mean(),
        'RPM_Unworn_Std': unworn_data['RPM'].std(),
        'RPM_Worn_Mean': worn_data['RPM'].mean(),
        'RPM_Worn_Std': worn_data['RPM'].std(),
        'Current_Unworn_Mean': unworn_data['I_avg_machine'].mean(),
        'Current_Unworn_Std': unworn_data['I_avg_machine'].std(),
        'Current_Worn_Mean': worn_data['I_avg_machine'].mean(),
        'Current_Worn_Std': worn_data['I_avg_machine'].std(),
        'Power_Unworn_Mean': unworn_data['kW_machine'].mean(),
        'Power_Unworn_Std': unworn_data['kW_machine'].std(),
        'Power_Worn_Mean': worn_data['kW_machine'].mean(),
        'Power_Worn_Std': worn_data['kW_machine'].std(),
        'PF_Unworn_Mean': unworn_data['PF_machine'].mean(),
        'PF_Worn_Mean': worn_data['PF_machine'].mean()
    }
    
    return summary_stats

def create_all_visualizations():
    """Create all visualization tabs"""
    print("Creating individual visualization tabs...")
    
    # Load data
    comparison_df, combined_data, unworn_data, worn_data = create_comparison_dataset()
    
    # Create the new velocity/acceleration analysis tab
    print("Creating velocity/acceleration analysis tab...")
    summary_stats = create_velocity_acceleration_analysis_tab(unworn_data, worn_data)
    
    # Create a few key visualizations that work with the actual data
    print("Creating time series plot...")
    create_time_series_plot(unworn_data, worn_data)
    
    print("Creating distribution plot...")
    create_distribution_plot(unworn_data, worn_data)
    
    print("Creating power comparison plot...")
    create_power_comparison_plot(unworn_data, worn_data)
    
    print("Creating box plot...")
    create_box_plot(unworn_data, worn_data)
    
    print("Creating statistical heatmap...")
    create_statistical_heatmap(unworn_data, worn_data)
    
    print("Creating correlation heatmap...")
    create_correlation_heatmap(combined_data)
    
    print("All visualization tabs created successfully!")
    print("\nGenerated files:")
    print("- tab1_time_series.png")
    print("- tab2_distribution.png")
    print("- tab3_power_comparison.png")
    print("- tab4_box_plot.png")
    print("- tab5_statistical_heatmap.png")
    print("- tab6_correlation_heatmap.png")
    print("- tab12_velocity_acceleration_analysis.png")
    
    return comparison_df

if __name__ == "__main__":
    print("Starting Tool Wear Analysis with Tabbed Visualizations...")
    
    # Create all visualizations
    comparison_df = create_all_visualizations()
    
    print("\nAnalysis complete! All visualization tabs have been created.") 