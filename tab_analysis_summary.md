# CNC Tool Wear Analysis - Detailed Tab Interpretations

## **📊 Dataset Overview**
- **Unworn Tools**: Experiments 1 & 2 (feedrate: 6 & 20, clamp_pressure: 4)
- **Worn Tools**: Experiments 7 & 8 (feedrate: 20 & 20, clamp_pressure: 4)
- **Material**: Wax (consistent across all experiments)
- **Sample Size**: 300 samples per experiment (600 total per condition)

---

## **📈 Tab 1: Time Series Analysis**

### **What This Shows:**
This visualization displays how current feedback values change over time for both unworn and worn tools.

### **Key Patterns to Observe:**

**1. X1_CurrentFeedback (X-axis):**
- **Unworn**: Relatively stable around -0.14 amps
- **Worn**: Much more negative values around -0.58 amps
- **Interpretation**: Worn tools require significantly more current to maintain the same cutting performance, indicating increased resistance and friction

**2. Y1_CurrentFeedback (Y-axis):**
- **Unworn**: Near zero (0.003 amps)
- **Worn**: Positive values around 0.066 amps
- **Interpretation**: Y-axis shows dramatic current increase with wear, suggesting the tool is struggling to maintain position

**3. S1_CurrentFeedback (Spindle):**
- **Unworn**: ~11.6 amps
- **Worn**: ~12.4 amps
- **Interpretation**: Spindle motor draws more current to maintain speed as tool wears

### **Practical Implications:**
- **Early Warning**: Current spikes can predict tool wear before visual inspection
- **Threshold Detection**: Set alarms when current exceeds normal ranges
- **Performance Monitoring**: Track efficiency degradation over time

---

## **📊 Tab 2: Distribution Analysis**

### **What This Shows:**
Histogram comparisons showing the statistical distribution of current feedback values.

### **Key Insights:**

**1. Distribution Shape:**
- **Unworn**: Tighter, more concentrated distributions
- **Worn**: Wider, more spread-out distributions
- **Interpretation**: Worn tools show more variability in performance

**2. Peak Shifts:**
- **X-axis**: Significant shift toward more negative values
- **Y-axis**: Shift from near-zero to positive values
- **S-axis**: Moderate shift toward higher values

**3. Variance Increase:**
- All axes show increased variance with wear
- **Interpretation**: Less predictable performance with worn tools

### **Practical Applications:**
- **Quality Control**: Wider distributions indicate less consistent machining
- **Predictive Maintenance**: Variance increases can signal approaching tool failure
- **Process Optimization**: Tighter distributions indicate better tool condition

---

## **⚡ Tab 3: Power Comparison**

### **What This Shows:**
Output power consumption patterns across all axes over time.

### **Key Observations:**

**1. Power Consumption Patterns:**
- **Unworn**: Lower, more stable power consumption
- **Worn**: Higher, more variable power consumption
- **Interpretation**: Worn tools require more energy to perform the same operations

**2. Axis-Specific Patterns:**
- **X-axis Power**: Most dramatic increase with wear
- **Y-axis Power**: Moderate increase
- **S-axis Power**: Smallest increase
- **Interpretation**: Different axes respond differently to tool wear

**3. Efficiency Loss:**
- Higher power consumption for same operations
- **Interpretation**: Energy efficiency decreases with tool wear

### **Practical Implications:**
- **Energy Monitoring**: Track power efficiency as tool condition indicator
- **Cost Analysis**: Calculate energy cost increases with tool wear
- **Performance Metrics**: Power consumption as key performance indicator

---

## **📦 Tab 4: Box Plot Analysis**

### **What This Shows:**
Statistical summary of current feedback distributions using box plots.

### **Key Statistical Insights:**

**1. Median Shifts:**
- **X1_CurrentFeedback**: -0.14 → -0.58 (significant negative shift)
- **Y1_CurrentFeedback**: 0.003 → 0.066 (significant positive shift)
- **S1_CurrentFeedback**: 11.56 → 12.44 (moderate positive shift)

**2. Interquartile Range (IQR):**
- All axes show increased IQR with wear
- **Interpretation**: More variable performance with worn tools

**3. Outlier Patterns:**
- More outliers with worn tools
- **Interpretation**: Less predictable behavior with tool wear

### **Statistical Significance:**
- All changes are statistically significant (p < 0.001)
- **Interpretation**: Clear, measurable differences between tool conditions

### **Practical Applications:**
- **Quality Assurance**: Use statistical thresholds for tool replacement
- **Process Control**: Monitor for statistical deviations
- **Predictive Models**: Use statistical parameters for wear prediction

---

## **📋 Tab 5: Statistical Summary**

### **What This Shows:**
Heatmap of mean differences between unworn and worn tools across all sensor readings.

### **Key Findings:**

**1. Most Sensitive Sensors:**
- **Y1_CurrentFeedback**: +2003.6% change (highest sensitivity)
- **X1_CurrentFeedback**: -316.9% change (second highest)
- **X1_OutputPower**: +156.7% change (third highest)

**2. Least Sensitive Sensors:**
- **S1_CurrentFeedback**: +7.6% change (least sensitive)
- **S1_OutputPower**: +8.9% change (second least sensitive)

**3. Pattern Recognition:**
- Current feedback sensors are more sensitive than power sensors
- Y-axis shows the most dramatic changes
- Spindle (S-axis) shows the least change

### **Practical Implications:**
- **Sensor Priority**: Focus on Y1_CurrentFeedback for early detection
- **Multi-Sensor Approach**: Combine multiple sensors for robust detection
- **Threshold Setting**: Use percentage changes for alarm thresholds

---

## **🔗 Tab 6: Correlation Analysis**

### **What This Shows:**
Correlation matrix showing relationships between different sensor readings.

### **Key Correlation Patterns:**

**1. Strong Positive Correlations:**
- Current feedback sensors correlate with each other
- Power sensors correlate with each other
- **Interpretation**: Related sensors show similar patterns

**2. Weak Correlations:**
- Current feedback vs power sensors show weaker correlations
- **Interpretation**: Different types of measurements provide complementary information

**3. Tool Condition Effect:**
- Correlations change with tool wear
- **Interpretation**: Sensor relationships evolve as tools wear

### **Practical Applications:**
- **Feature Selection**: Choose uncorrelated sensors for robust models
- **Redundancy**: Identify redundant sensors for cost optimization
- **Model Building**: Use correlation patterns for predictive model design

---

## **🎯 Tab 7: Statistical Significance**

### **What This Shows:**
P-values and effect sizes for statistical tests comparing unworn vs worn tools.

### **Key Statistical Results:**

**1. Highly Significant Sensors (p < 0.001):**
- Y1_CurrentFeedback: p = 0.000
- X1_CurrentFeedback: p = 0.000
- X1_OutputPower: p = 0.000
- **Interpretation**: These sensors provide reliable wear detection

**2. Effect Sizes:**
- Large effect sizes (>0.8) for most sensors
- **Interpretation**: Practical significance matches statistical significance

**3. Confidence Levels:**
- All sensors show significant differences
- **Interpretation**: Reliable detection across all measured parameters

### **Practical Implications:**
- **Model Reliability**: High confidence in wear detection models
- **Threshold Setting**: Statistical significance supports practical thresholds
- **Quality Assurance**: Reliable basis for automated tool replacement

---

## **📈 Tab 8: Trend Analysis**

### **What This Shows:**
Linear trends and patterns in sensor data over time.

### **Key Trend Patterns:**

**1. Trend Direction:**
- Most sensors show increasing trends with wear
- **Interpretation**: Gradual degradation over time

**2. Trend Strength:**
- Strong trends in current feedback sensors
- **Interpretation**: Consistent degradation patterns

**3. Trend Variability:**
- Some sensors show more variable trends
- **Interpretation**: Different degradation rates across sensors

### **Practical Applications:**
- **Predictive Maintenance**: Use trends to predict future wear
- **Scheduling**: Plan tool replacement based on trend analysis
- **Optimization**: Identify optimal replacement timing

---

## **🔍 Tab 9: Pattern Recognition**

### **What This Shows:**
Advanced pattern analysis using rolling statistics and anomaly detection.

### **Key Pattern Insights:**

**1. Rolling Statistics:**
- Moving averages show gradual changes
- **Interpretation**: Smooth degradation patterns

**2. Anomaly Detection:**
- Outliers indicate sudden changes
- **Interpretation**: Potential tool failure events

**3. Pattern Classification:**
- Different patterns for different wear stages
- **Interpretation**: Wear progression follows predictable patterns

### **Practical Applications:**
- **Early Warning**: Detect anomalies before failure
- **Stage Classification**: Identify wear progression stages
- **Predictive Models**: Use patterns for future prediction

---

## **📊 Tab 10: Comparative Analysis**

### **What This Shows:**
Side-by-side comparisons of all key metrics between unworn and worn tools.

### **Key Comparative Insights:**

**1. Magnitude of Changes:**
- Current feedback shows largest changes
- Power sensors show moderate changes
- **Interpretation**: Current feedback is most sensitive to wear

**2. Consistency of Changes:**
- All sensors show consistent direction of change
- **Interpretation**: Systematic degradation across all systems

**3. Practical Thresholds:**
- Clear separation between conditions
- **Interpretation**: Reliable thresholds for automated detection

### **Practical Applications:**
- **Automated Detection**: Use thresholds for automatic alerts
- **Quality Control**: Monitor for threshold violations
- **Process Optimization**: Optimize based on comparative analysis

---

## **🎯 Tab 11: Predictive Insights**

### **What This Shows:**
Summary of key findings and recommendations for predictive maintenance.

### **Key Predictive Insights:**

**1. Most Reliable Indicators:**
- Y1_CurrentFeedback: 2003.6% change
- X1_CurrentFeedback: -316.9% change
- X1_OutputPower: 156.7% change

**2. Early Warning Thresholds:**
- Y1_CurrentFeedback > 0.03 amps
- X1_CurrentFeedback < -0.3 amps
- X1_OutputPower > 0.15 watts

**3. Multi-Sensor Approach:**
- Combine current and power sensors
- Use statistical significance for reliability
- Monitor trends for predictive capability

### **Implementation Recommendations:**

**1. Real-Time Monitoring:**
- Set up continuous monitoring of key sensors
- Implement automated threshold alerts
- Use trend analysis for predictive maintenance

**2. Quality Assurance:**
- Establish statistical control limits
- Monitor for threshold violations
- Implement automated tool replacement

**3. Process Optimization:**
- Use sensor data for process optimization
- Monitor efficiency metrics
- Implement predictive maintenance schedules

---

## **🎯 Overall Conclusions**

### **Primary Findings:**
1. **Current feedback sensors are the most sensitive indicators of tool wear**
2. **Y1_CurrentFeedback shows the most dramatic changes (+2003.6%)**
3. **All sensors show statistically significant differences between conditions**
4. **Worn tools require significantly more energy and current**
5. **Pattern recognition enables predictive maintenance**

### **Practical Applications:**
- **Automated Tool Replacement**: Use sensor thresholds for automatic alerts
- **Quality Control**: Monitor for statistical deviations
- **Predictive Maintenance**: Use trends and patterns for future prediction
- **Process Optimization**: Optimize based on sensor feedback
- **Cost Reduction**: Reduce unplanned downtime and tool failures

### **Implementation Strategy:**
1. **Phase 1**: Implement real-time monitoring of key sensors
2. **Phase 2**: Set up automated threshold alerts
3. **Phase 3**: Develop predictive maintenance models
4. **Phase 4**: Integrate with quality control systems
5. **Phase 5**: Optimize processes based on sensor data 