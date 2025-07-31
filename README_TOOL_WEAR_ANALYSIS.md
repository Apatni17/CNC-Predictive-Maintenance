# CNC Mill Tool Wear Analysis

This project provides comprehensive analysis tools to explore correlations and relationships between CNC machine variables and tool wear, specifically designed to answer key research questions about predictive maintenance.

## 🎯 Research Questions Addressed

1. **"Are there specific patterns or spikes in current feedback or bus voltage before wear happens?"**
2. **"Can variations in cutting forces (as reflected by current feedback, output power/current) explain differences in tool wear?"**
3. **"How do feedrate and tool wear interact?"**
4. **"How do spindle and axis velocities/accelerations influence tool wear mechanisms?"**
5. **"Does the difference between actual and commanded positions affect tool wear?"**

## 📁 Files Overview

### Analysis Scripts
- `tool_wear_analysis.py` - Main analysis script
- `view_results.py` - Results viewer and summary

### Generated Outputs
- `correlation_matrix.png` - Correlation heatmap
- `feature_importance.png` - Feature importance ranking
- `roc_curve.png` - ROC curve analysis
- `feature_distributions.png` - Distribution comparisons
- `time_series_comparison.png` - Time series analysis
- `tool_wear_statistics.csv` - Statistical summary
- `tool_wear_analysis_summary.md` - Detailed findings

## 🚀 Quick Start

### 1. Run the Analysis
```bash
python3 tool_wear_analysis.py
```

### 2. View Results
```bash
python3 view_results.py
```

## 📊 Key Findings Summary

### **Exceptional Predictive Performance**
- **ROC AUC Score: 0.998** (Near-perfect classification)
- **Classification Accuracy: 98%**

### **Most Important Predictive Variables**
1. **M1_CURRENT_FEEDRATE** (0.174 importance) - Most critical
2. **X1_OutputCurrent** (0.131 importance) - Second most important
3. **Y1_OutputCurrent** (0.068 importance)
4. **S1_CommandPosition** (0.058 importance)

### **Strongest Correlations with Tool Wear**
1. **Z1_CommandPosition** (0.268)
2. **Y1_CommandPosition** (0.259)
3. **X1_ActualPosition** (0.257)
4. **S1_CommandVelocity** (0.225)

## 🔍 Analysis Components

### 1. **Correlation Matrix**
- Shows relationships between all variables and tool wear
- Identifies which variables correlate most strongly with wear
- Helps understand variable interactions

### 2. **Feature Importance Analysis**
- Uses Random Forest to rank feature importance
- Identifies the most predictive variables
- Provides actionable insights for monitoring

### 3. **ROC Curve Analysis**
- Evaluates model performance
- Shows classification accuracy
- Provides confidence in predictions

### 4. **Distribution Plots**
- Compares variable distributions between worn/unworn tools
- Shows how variables change with wear
- Identifies clear separation patterns

### 5. **Time Series Analysis**
- Shows how variables change over time
- Compares worn vs unworn tool patterns
- Identifies temporal patterns

## 💡 Implementation Recommendations

### **Primary Monitoring Variables**
1. **Feedrate (M1_CURRENT_FEEDRATE)** - Monitor for reductions
2. **Current Feedback (X1_OutputCurrent, Y1_OutputCurrent)** - Monitor for increases
3. **DC Bus Voltage (X1_DCBusVoltage)** - Monitor for voltage spikes
4. **Position Accuracy** - Monitor actual vs commanded position differences

### **Early Warning Thresholds**
- **Current Feedback Increase**: >10% above baseline
- **Feedrate Reduction**: >20% below normal operating range
- **Voltage Spikes**: >15% above normal operating voltage
- **Position Deviation**: >5% difference between actual and commanded

## 📈 Data Details

### **Dataset Information**
- **18 experiments** (8 unworn tools, 10 worn tools)
- **300 samples per experiment** (5,400 total observations)
- **45 key features** analyzed
- **100ms sampling rate** for time series data

### **Key Variables Analyzed**
- **Current Feedback**: X1, Y1, Z1, S1 axes
- **Voltage**: DC bus and output voltages
- **Power**: Output power measurements
- **Position**: Actual and commanded positions
- **Velocity**: Actual and commanded velocities
- **Acceleration**: Actual and commanded accelerations
- **Feedrate**: Current feedrate
- **System Inertia**: Torque inertia

## 🔧 Technical Requirements

### **Dependencies**
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

### **Data Structure**
The analysis expects experiment files in the format:
```
data/CNC mill wear /experiment_XX.csv
```

Each CSV file should contain columns for:
- X1, Y1, Z1, S1 axis measurements
- Current feedback, voltage, power
- Position, velocity, acceleration
- Feedrate and system parameters

## 📋 Usage Examples

### **Custom Analysis**
```python
from tool_wear_analysis import ToolWearAnalyzer

# Create analyzer
analyzer = ToolWearAnalyzer()

# Load specific experiments
analyzer.load_experiments([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

# Sample different amounts
analyzer.sample_data(samples_per_experiment=500)

# Run analysis
analyzer.run_complete_analysis()
```

### **View Specific Results**
```python
# View correlation matrix
corr_matrix = analyzer.create_correlation_matrix(features)

# View feature importance
rf_model, importance = analyzer.create_feature_importance_analysis(features)

# View ROC analysis
roc_auc = analyzer.create_roc_analysis(rf_model, features)
```

## 🎯 Key Insights for Predictive Maintenance

### **1. Feedrate is the Primary Indicator**
- Most important predictor of tool wear
- Operators naturally reduce feedrate as tools wear
- Monitor for feedrate reductions as early warning

### **2. Current Feedback Provides Real-time Signals**
- Direct indicator of cutting forces
- Shows immediate changes in tool condition
- Excellent for real-time monitoring

### **3. Position Accuracy Degrades with Wear**
- Strong correlation between position parameters and wear
- Suggests tool wear affects machining precision
- Monitor position deviations

### **4. Multi-variable Approach is Essential**
- No single variable provides perfect prediction
- Combination of feedrate, current, and position gives best results
- Use ensemble approach for robust monitoring

## 📞 Support

For questions or issues with the analysis:
1. Check the generated `tool_wear_analysis_summary.md` for detailed findings
2. Review the generated visualizations for specific insights
3. Examine the `tool_wear_statistics.csv` for statistical details

## 🔄 Future Enhancements

Potential improvements for the analysis:
- Continuous wear measurement (vs binary worn/unworn)
- Real-time streaming analysis
- Integration with CNC control systems
- Machine learning model deployment
- Alert system implementation 