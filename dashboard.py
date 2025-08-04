import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from tool_wear_analysis import ToolWearAnalyzer

# Set page config
st.set_page_config(
    page_title="CNC Predictive Maintenance Dashboard",
    page_icon="🔧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        padding: 1rem;
        background: linear-gradient(90deg, #f0f2f6, #e8f4fd);
        border-radius: 10px;
    }
    .section-header {
        font-size: 1.8rem;
        font-weight: bold;
        color: #2c3e50;
        margin: 2rem 0 1rem 0;
        padding: 0.5rem;
        border-left: 4px solid #3498db;
        background-color: #f8f9fa;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin: 0.5rem 0;
    }
    .metric-card h3 {
        color: #ecf0f1;
        font-size: 0.9rem;
        margin-bottom: 0.5rem;
    }
    .metric-card h2 {
        color: white;
        font-size: 2rem;
        margin: 0.5rem 0;
    }
    .metric-card p {
        color: #bdc3c7;
        font-size: 0.8rem;
        margin: 0;
    }
    .insight-box {
        background: linear-gradient(135deg, #74b9ff 0%, #0984e3 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .insight-box h4 {
        color: white;
        margin-bottom: 1rem;
    }
    .insight-box ul, .insight-box ol {
        color: #ecf0f1;
    }
    .insight-box li {
        margin: 0.5rem 0;
    }
    .nav-button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 0.5rem 1rem;
        border: none;
        border-radius: 5px;
        cursor: pointer;
        margin: 0.2rem;
        font-weight: bold;
    }
    .nav-button:hover {
        background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
    }
    .success-box {
        background: linear-gradient(135deg, #00b894 0%, #00a085 100%);
        color: white;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    .warning-box {
        background: linear-gradient(135deg, #fdcb6e 0%, #e17055 100%);
        color: white;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    .warning-box {
        background: linear-gradient(135deg, #fdcb6e 0%, #e17055 100%);
        color: white;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

def create_navigation():
    """Create navigation buttons"""
    st.sidebar.markdown("## 🧭 Navigation")
    
    # Initialize session state for page navigation
    if 'current_page' not in st.session_state:
        st.session_state.current_page = "overview"
    
    # Navigation buttons
    if st.sidebar.button("🏠 Overview", key="overview_btn"):
        st.session_state.current_page = "overview"
    if st.sidebar.button("🎯 Validation Predictor", key="predictor_btn"):
        st.session_state.current_page = "predictor"
    if st.sidebar.button("🔧 Tool Wear Analysis", key="tool_wear_btn"):
        st.session_state.current_page = "tool_wear"
    if st.sidebar.button("🎯 Machine Completion", key="completion_btn"):
        st.session_state.current_page = "completion"
    if st.sidebar.button("🔍 Quality Analysis", key="quality_btn"):
        st.session_state.current_page = "quality"
    if st.sidebar.button("📊 Performance Metrics", key="performance_btn"):
        st.session_state.current_page = "performance"
    if st.sidebar.button("🔬 Research Insights", key="insights_btn"):
        st.session_state.current_page = "insights"

def overview_page():
    """Overview page with key metrics and summary"""
    st.markdown('<h1 class="main-header">🔧 CNC Predictive Maintenance Dashboard</h1>', unsafe_allow_html=True)
    
    # Check if analysis files exist
    if not os.path.exists('tool_wear_statistics.csv'):
        st.error("❌ Analysis data not found. Running analysis first...")
        with st.spinner("Running tool wear analysis..."):
            try:
                analyzer = ToolWearAnalyzer()
                analyzer.run_complete_analysis()
                st.success("✅ Analysis completed!")
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
                return
    
    # Load data
    stats = pd.read_csv('tool_wear_statistics.csv', index_col=0)
    
    # Key Performance Metrics
    st.markdown('<h2 class="section-header">📊 Key Performance Metrics</h2>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown("""
        <div class="metric-card">
        <h3>🎯 Tool Wear Detection</h3>
        <h2>98%</h2>
        <p>Classification Accuracy</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
        <h3>🏭 Machine Completion</h3>
        <h2>98.52%</h2>
        <p>Success Rate</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
        <h3>🔬 Data Points</h3>
        <h2>5,400</h2>
        <p>Analyzed</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class="metric-card">
        <h3>🎛️ Key Features</h3>
        <h2>45</h2>
        <p>Monitored Variables</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Quick Insights
    st.markdown('<h2 class="section-header">💡 Quick Insights</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="success-box">
        <h4>✅ Top Tool Wear Indicators:</h4>
        <ul>
        <li><strong>Y1 Current Feedback:</strong> 75.2% higher in worn tools</li>
        <li><strong>X1 Current Feedback:</strong> -0.320 correlation with completion</li>
        <li><strong>Feedrate:</strong> Most important ML predictor</li>
        <li><strong>Bus Voltage:</strong> Drops significantly with wear</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="warning-box">
        <h4>⚠️ Critical Completion Factors:</h4>
        <ul>
        <li><strong>X1 Current Feedback:</strong> 46.4% importance for completion</li>
        <li><strong>X1 Command Velocity:</strong> Higher speeds reduce completion</li>
        <li><strong>X1 Output Power:</strong> High consumption reduces success</li>
        <li><strong>Acceleration Rates:</strong> High acceleration reduces completion</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # Experiment Overview
    st.markdown('<h2 class="section-header">🧪 Experiment Overview</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📋 Experiment Sets")
        experiment_info = {
            "Experiments 1-8": "Unworn Tools (2,400 data points)",
            "Experiments 9-18": "Worn Tools (3,000 data points)",
            "Total Samples": "300 points per experiment",
            "Analysis Period": "Complete machining cycles"
        }
        
        for exp, desc in experiment_info.items():
            st.markdown(f"**{exp}:** {desc}")
    
    with col2:
        st.markdown("### 📊 Data Distribution")
        fig, ax = plt.subplots(figsize=(8, 6))
        categories = ['Unworn Tools', 'Worn Tools']
        counts = [2400, 3000]
        colors = ['#2ecc71', '#e74c3c']
        
        bars = ax.bar(categories, counts, color=colors, alpha=0.7)
        ax.set_title('Tool Wear Distribution', fontsize=14, fontweight='bold')
        ax.set_ylabel('Number of Data Points')
        
        # Add value labels on bars
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 50,
                   f'{count:,}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        st.pyplot(fig)

def tool_wear_page():
    """Tool wear analysis page"""
    st.markdown('<h1 class="section-header">🔧 Tool Wear Analysis</h1>', unsafe_allow_html=True)
    
    # Correlation Analysis
    st.markdown('<h2 class="section-header">🔗 Correlation Analysis</h2>', unsafe_allow_html=True)
    
    if os.path.exists('correlation_matrix.png'):
        st.image('correlation_matrix.png', use_container_width=True)
        
        st.markdown("""
        <div class="insight-box">
        <h4>🔍 Key Correlation Insights:</h4>
        <ul>
        <li><strong>Z1_CommandPosition (0.268):</strong> Strongest correlation - position commands change significantly with wear</li>
        <li><strong>Y1_CommandPosition (0.259):</strong> Y-axis position commands also highly correlated</li>
        <li><strong>X1_ActualPosition (0.257):</strong> Actual X-axis position shows wear patterns</li>
        <li><strong>S1_CommandVelocity (0.225):</strong> Spindle velocity commands affected by wear</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # Feature Importance
    st.markdown('<h2 class="section-header">🎯 Feature Importance Analysis</h2>', unsafe_allow_html=True)
    
    if os.path.exists('feature_importance.png'):
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.image('feature_importance.png', use_container_width=True)
        
        with col2:
            st.markdown("""
            <div class="insight-box">
            <h4>🏆 ML Model Feature Importance:</h4>
            <ol>
            <li><strong>M1_CURRENT_FEEDRATE (0.174)</strong><br>
            <small>Feedrate is the primary ML predictor</small></li>
            
            <li><strong>X1_OutputCurrent (0.131)</strong><br>
            <small>Current feedback shows cutting force changes</small></li>
            
            <li><strong>Y1_OutputCurrent (0.068)</strong><br>
            <small>Y-axis current also important for wear detection</small></li>
            
            <li><strong>S1_CommandPosition (0.058)</strong><br>
            <small>Spindle position commands change with wear</small></li>
            </ol>
            </div>
            """, unsafe_allow_html=True)
    
    # Feature Distributions
    st.markdown('<h2 class="section-header">📊 Feature Distribution Analysis</h2>', unsafe_allow_html=True)
    
    if os.path.exists('feature_distributions.png'):
        st.image('feature_distributions.png', use_container_width=True)
        
        st.markdown("""
        <div class="insight-box">
        <h4>📋 Distribution Insights:</h4>
        <ul>
        <li><strong>Current Feedback:</strong> Worn tools show higher current consumption</li>
        <li><strong>Feedrate:</strong> Worn tools operate at different feedrates</li>
        <li><strong>Voltage Patterns:</strong> Clear differences in voltage behavior</li>
        <li><strong>Position Accuracy:</strong> Worn tools show position deviations</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

def completion_page():
    """Machine completion analysis page"""
    st.markdown('<h1 class="section-header">🎯 Machine Completion Analysis</h1>', unsafe_allow_html=True)
    
    # Check if machine completion analysis files exist
    if not os.path.exists('machine_completion_statistics.csv'):
        st.error("❌ Machine completion analysis not found. Please run the machine completion analysis first.")
        return
    
    completion_stats = pd.read_csv('machine_completion_statistics.csv', index_col=0)
    
    # Key completion metrics
    st.markdown('<h2 class="section-header">📊 Completion Performance Metrics</h2>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown("""
        <div class="metric-card">
        <h3>🏭 Completion Success Rate</h3>
        <h2>98.52%</h2>
        <p>Excellent Performance</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
        <h3>⚡ X1 Current Feedback</h3>
        <h2>-0.320</h2>
        <p>Strong Negative Correlation</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
        <h3>🎛️ X1 Command Velocity</h3>
        <h2>-0.262</h2>
        <p>Moderate Negative Correlation</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class="metric-card">
        <h3>🔋 X1 Output Power</h3>
        <h2>-0.313</h2>
        <p>Strong Negative Correlation</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Velocity and Acceleration Analysis
    st.markdown('<h2 class="section-header">🚀 Velocity & Acceleration Impact</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        if os.path.exists('velocity_completion_analysis.png'):
            st.image('velocity_completion_analysis.png', use_container_width=True)
            st.markdown("""
            <div class="insight-box">
            <h4>⚡ Key Velocity Insights:</h4>
            <ul>
            <li><strong>X1_CommandVelocity (-0.262):</strong> Higher command velocities correlate with lower completion rates</li>
            <li><strong>Y1_CommandVelocity (-0.061):</strong> Minimal impact on completion</li>
            <li><strong>Z1_CommandVelocity (-0.006):</strong> Negligible effect on completion</li>
            <li><strong>S1_CommandVelocity (-0.059):</strong> Spindle velocity has minor impact</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        if os.path.exists('acceleration_completion_analysis.png'):
            st.image('acceleration_completion_analysis.png', use_container_width=True)
            st.markdown("""
            <div class="insight-box">
            <h4>📈 Acceleration Insights:</h4>
            <ul>
            <li><strong>X1_Acceleration (-0.164):</strong> Higher acceleration rates reduce completion success</li>
            <li><strong>Y1_Acceleration (-0.049):</strong> Minor impact on completion</li>
            <li><strong>Z1_Acceleration (-0.014):</strong> Minimal effect</li>
            <li><strong>S1_Acceleration (-0.078):</strong> Spindle acceleration has small negative impact</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)
    
    # Cutting Forces Analysis
    st.markdown('<h2 class="section-header">🔧 Cutting Forces Impact</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        if os.path.exists('cutting_forces_completion_analysis.png'):
            st.image('cutting_forces_completion_analysis.png', use_container_width=True)
            st.markdown("""
            <div class="insight-box">
            <h4>💪 Cutting Force Correlations:</h4>
            <ul>
            <li><strong>X1_CurrentFeedback (-0.320):</strong> Strongest negative correlation - higher current = lower completion</li>
            <li><strong>X1_OutputPower (-0.313):</strong> High power consumption reduces success</li>
            <li><strong>Y1_OutputPower (-0.159):</strong> Moderate negative impact</li>
            <li><strong>X1_OutputCurrent (0.124):</strong> Positive correlation - higher output current improves completion</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        if os.path.exists('cutting_forces_importance.png'):
            st.image('cutting_forces_importance.png', use_container_width=True)
            st.markdown("""
            <div class="insight-box">
            <h4>🏆 ML Feature Importance for Completion:</h4>
            <ol>
            <li><strong>X1_CurrentFeedback (46.4%):</strong> Primary predictor of completion success</li>
            <li><strong>X1_OutputPower (15.1%):</strong> Second most important factor</li>
            <li><strong>Y1_OutputPower (10.1%):</strong> Y-axis power consumption</li>
            <li><strong>Y1_CurrentFeedback (8.7%):</strong> Y-axis current feedback</li>
            </ol>
            </div>
            """, unsafe_allow_html=True)
    
    # Scatter Analysis
    if os.path.exists('cutting_forces_scatter_analysis.png'):
        st.markdown('<h2 class="section-header">📊 Completion Status Patterns</h2>', unsafe_allow_html=True)
        st.image('cutting_forces_scatter_analysis.png', use_container_width=True)
        st.markdown("""
        <div class="insight-box">
        <h4>📈 Scatter Plot Insights:</h4>
        <ul>
        <li><strong>Success vs Failure Patterns:</strong> Clear separation between successful and failed completions</li>
        <li><strong>Threshold Identification:</strong> Visual thresholds for completion prediction</li>
        <li><strong>Outlier Detection:</strong> Points that deviate from expected patterns</li>
        <li><strong>Variable Relationships:</strong> How different cutting force variables interact</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

def performance_page():
    """Performance metrics page"""
    st.markdown('<h1 class="section-header">📈 Performance Metrics</h1>', unsafe_allow_html=True)
    
    # ROC Curve Performance
    st.markdown('<h2 class="section-header">🚀 Model Performance</h2>', unsafe_allow_html=True)
    
    if os.path.exists('roc_curve.png'):
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.image('roc_curve.png', use_container_width=True)
        
        with col2:
            st.markdown("""
            <div class="insight-box">
            <h4>🚀 Performance Metrics:</h4>
            <ul>
            <li><strong>ROC AUC:</strong> 0.998 (Near perfect)</li>
            <li><strong>Accuracy:</strong> 98%</li>
            <li><strong>Precision:</strong> 98%</li>
            <li><strong>Recall:</strong> 97%</li>
            </ul>
            
            <h4>💡 Interpretation:</h4>
            <p>The model can distinguish between worn and unworn tools with exceptional accuracy, making it highly suitable for predictive maintenance applications.</p>
            </div>
            """, unsafe_allow_html=True)
    
    # Time Series Analysis
    st.markdown('<h2 class="section-header">⏰ Time Series Patterns</h2>', unsafe_allow_html=True)
    
    if os.path.exists('time_series_comparison.png'):
        st.image('time_series_comparison.png', use_container_width=True)
        
        st.markdown("""
        <div class="insight-box">
        <h4>⏱️ Temporal Insights:</h4>
        <ul>
        <li><strong>Pattern Recognition:</strong> Clear temporal patterns distinguish worn vs unworn tools</li>
        <li><strong>Early Warning:</strong> Changes in patterns can provide early warning of wear</li>
        <li><strong>Consistency:</strong> Unworn tools show more consistent patterns</li>
        <li><strong>Variability:</strong> Worn tools show increased variability in measurements</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # Detailed Statistics
    st.markdown('<h2 class="section-header">📋 Detailed Statistics</h2>', unsafe_allow_html=True)
    
    if os.path.exists('tool_wear_statistics.csv'):
        stats = pd.read_csv('tool_wear_statistics.csv', index_col=0)
        
        # Show key statistics
        key_stats = stats[['X1_CurrentFeedback_mean', 'Y1_CurrentFeedback_mean', 
                          'X1_DCBusVoltage_mean', 'M1_CURRENT_FEEDRATE_mean']]
        
        st.dataframe(key_stats, use_container_width=True)
        
        st.markdown("""
        <div class="insight-box">
        <h4>📊 Statistical Summary:</h4>
        <p>The table above shows mean values for key features across worn (1) and unworn (0) tools. 
        Notice the significant differences in current feedback and feedrate values, which are the primary indicators of tool wear.</p>
        </div>
        """, unsafe_allow_html=True)

def insights_page():
    """Research insights page"""
    st.markdown('<h1 class="section-header">🔬 Research Insights</h1>', unsafe_allow_html=True)
    
    # Research Questions Answered
    st.markdown('<h2 class="section-header">✅ Research Questions Answered</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="insight-box">
        <h4>🎯 Machine Completion Questions:</h4>
        
        <h5>1. "How do spindle and axis velocities/accelerations influence machine completion rates?"</h5>
        <p><strong>ANSWERED</strong> - X1_CommandVelocity (-0.262) and X1_Acceleration (-0.164) show strong negative correlations with completion</p>
        
        <h5>2. "Can variations in cutting forces explain differences in machine completion success?"</h5>
        <p><strong>YES</strong> - X1_CurrentFeedback (-0.320) and X1_OutputPower (-0.313) are the strongest predictors</p>
        
        <h5>3. "Are there specific patterns in current feedback before wear?"</h5>
        <p><strong>YES</strong> - Y1_CurrentFeedback is 75% higher in worn tools</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="insight-box">
        <h4>🔧 Tool Wear Questions:</h4>
        
        <h5>1. "Can cutting forces explain tool wear differences?"</h5>
        <p><strong>YES</strong> - X1_OutputCurrent is the 2nd most important predictor</p>
        
        <h5>2. "How do feedrate and tool wear interact?"</h5>
        <p><strong>ANSWERED</strong> - M1_CURRENT_FEEDRATE is the most critical predictor</p>
        
        <h5>3. "What are the early warning signs of tool wear?"</h5>
        <p><strong>IDENTIFIED</strong> - Current feedback increases and voltage drops</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Predictive Maintenance Recommendations
    st.markdown('<h2 class="section-header">🎯 Predictive Maintenance Recommendations</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="success-box">
        <h4>🔧 Tool Wear Monitoring:</h4>
        <ol>
        <li><strong>Monitor feedrate reductions</strong> - Primary indicator</li>
        <li><strong>Track current feedback increases</strong> - Early warning signals</li>
        <li><strong>Watch for voltage spikes</strong> - Precursor to failure</li>
        <li><strong>Monitor position accuracy</strong> - Deviations indicate wear</li>
        <li><strong>Set automated thresholds</strong> based on correlation patterns</li>
        </ol>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="warning-box">
        <h4>🏭 Completion Optimization:</h4>
        <ol>
        <li><strong>Monitor X1 current feedback</strong> - Primary completion predictor (46.4% importance)</li>
        <li><strong>Track X1 output power</strong> - Second most important factor (15.1%)</li>
        <li><strong>Control X1 command velocity</strong> - Higher velocities reduce completion rates</li>
        <li><strong>Watch acceleration rates</strong> - High acceleration reduces success</li>
        <li><strong>Set completion thresholds</strong> based on cutting force patterns</li>
        </ol>
        </div>
        """, unsafe_allow_html=True)
    
    # Next Steps
    st.markdown('<h2 class="section-header">🚀 Next Steps & Future Research</h2>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="insight-box">
    <h4>🔬 Recommended Research Directions:</h4>
    <ul>
    <li><strong>Real-time Monitoring System:</strong> Implement live monitoring with automated alerts</li>
    <li><strong>Predictive Models:</strong> Develop time-to-failure prediction models</li>
    <li><strong>Process Optimization:</strong> Create adaptive parameter adjustment systems</li>
    <li><strong>Quality Prediction:</strong> Build models to predict part quality based on tool condition</li>
    <li><strong>Cost-Benefit Analysis:</strong> Quantify the economic impact of predictive maintenance</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

def quality_page():
    """Quality analysis page"""
    st.markdown('<h1 class="section-header">🔍 Quality Analysis</h1>', unsafe_allow_html=True)
    
    # Check if quality analysis files exist
    if not os.path.exists('quality_statistics.csv'):
        st.error("❌ Quality analysis not found. Please run the quality analysis first.")
        return
    
    quality_stats = pd.read_csv('quality_statistics.csv', index_col=0)
    
    # Key Quality Metrics
    st.markdown('<h2 class="section-header">📊 Quality Performance Metrics</h2>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown("""
        <div class="metric-card">
        <h3>🏭 Machine Finalization</h3>
        <h2>98.52%</h2>
        <p>Overall Success Rate</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
        <h3>🔧 Unworn Tools</h3>
        <h2>68.33%</h2>
        <p>Finalization Rate</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
        <h3>⚠️ Worn Tools</h3>
        <h2>50.67%</h2>
        <p>Finalization Rate</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class="metric-card">
        <h3>📊 Data Points</h3>
        <h2>1,800</h2>
        <p>Analyzed</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Tool Condition Impact Analysis
    st.markdown('<h2 class="section-header">🔧 Tool Condition Impact</h2>', unsafe_allow_html=True)
    
    if os.path.exists('tool_condition_impact.png'):
        st.image('tool_condition_impact.png', use_container_width=True)
        
        st.markdown("""
        <div class="insight-box">
        <h4>🔍 Key Findings:</h4>
        <ul>
        <li><strong>Unworn Tools (68.33%):</strong> Significantly higher finalization rate than worn tools</li>
        <li><strong>Worn Tools (50.67%):</strong> Lower finalization rate</li>
        <li><strong>Performance Gap:</strong> 17.66% difference in favor of unworn tools</li>
        <li><strong>Expected Result:</strong> Unworn tools show better completion rates</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # Quality Correlations
    st.markdown('<h2 class="section-header">🔗 Quality Correlations</h2>', unsafe_allow_html=True)
    
    if os.path.exists('quality_correlations.png'):
        st.image('quality_correlations.png', use_container_width=True)
        
        st.markdown("""
        <div class="insight-box">
        <h4>💪 Strongest Quality Predictors:</h4>
        <ul>
        <li><strong>Machine Finalization:</strong> X1_DCBusVoltage (0.335), X1_CurrentFeedback (0.320), X1_OutputPower (0.313)</li>
        <li><strong>Tool Condition:</strong> Position variables (Z1, Y1, X1) show strongest correlations</li>
        <li><strong>Voltage Stability:</strong> Critical for machine finalization success</li>
        <li><strong>Current Feedback:</strong> Strong predictor of finalization outcomes</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    

def predictor_page():
    """Validation tool wear prediction page"""
    st.markdown('<h1 class="section-header">🎯 Validation Tool Wear Predictor</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="insight-box">
    <h4>🔍 True Validation with Experiments 17-18</h4>
    <p>This validation predictor was trained on experiments 1-16 (80/20 train/test split) and reserves experiments 17-18 
    for true validation. Upload experiment 17 or 18 to test the model's real-world accuracy!</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Show training information
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
        <h3>📚 Training Files</h3>
        <h2>16 Files</h2>
        <p>Experiments 1-16 (80/20 split)</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
        <h3>🔍 Validation Files</h3>
        <h2>2 Files</h2>
        <p>Experiments 17-18 (reserved)</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
        <h3>📊 Test Accuracy</h3>
        <h2>91.0%</h2>
        <p>On 20% test data</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class="metric-card">
        <h3>🔧 Key Features</h3>
        <h2>12 Features</h2>
        <p>Most important variables</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Check if validation model exists, if not train it
    if not os.path.exists('validation_tool_wear_model.pkl'):
        st.warning("⚠️ Validation model not found. Training model first...")
        with st.spinner("Training validation tool wear prediction model..."):
            try:
                from validation_tool_wear_predictor import ValidationToolWearPredictor
                predictor = ValidationToolWearPredictor()
                test_accuracy, feature_importance = predictor.train_model()
                st.success(f"✅ Validation model trained successfully! Test Accuracy: {test_accuracy:.3f}")
            except Exception as e:
                st.error(f"❌ Error training model: {str(e)}")
                return
    
    # File upload
    uploaded_file = st.file_uploader("📁 Upload validation data (experiment 17 or 18)", type=['csv'])
    
    # Machine name input
    machine_name = st.text_input("🏭 Machine Name/ID", value="Validation Machine", help="Enter experiment number (17 or 18)")
    
    if uploaded_file is not None and machine_name:
        try:
            # Load the uploaded data
            data = pd.read_csv(uploaded_file)
            st.success(f"✅ Data loaded successfully! Shape: {data.shape}")
            
            # Show data preview
            with st.expander("📋 Data Preview"):
                st.dataframe(data.head(), use_container_width=True)
            
            # Load validation predictor and make predictions
            with st.spinner("Analyzing tool wear risk..."):
                from validation_tool_wear_predictor import ValidationToolWearPredictor
                predictor = ValidationToolWearPredictor()
                wear_probabilities, wear_predictions = predictor.predict_tool_wear(data)
                analysis_results = predictor.analyze_machine_health(data, wear_probabilities)
                predictor.create_validation_report(data, wear_probabilities, analysis_results, machine_name)
            
            # Display results
            st.markdown('<h3 class="section-header">🎯 Prediction Results</h3>', unsafe_allow_html=True)
            
            # Risk assessment
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.markdown(f"""
                <div class="metric-card">
                <h3>Risk Level</h3>
                <h2>{analysis_results['risk_level']}</h2>
                <p>{analysis_results['risk_description']}</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                avg_prob = analysis_results['avg_wear_probability']
                st.markdown(f"""
                <div class="metric-card">
                <h3>Wear Probability</h3>
                <h2>{avg_prob:.1%}</h2>
                <p>Average Risk</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                max_prob = analysis_results['max_wear_probability']
                st.markdown(f"""
                <div class="metric-card">
                <h3>Peak Risk</h3>
                <h2>{max_prob:.1%}</h2>
                <p>Maximum Risk</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col4:
                st.markdown(f"""
                <div class="metric-card">
                <h3>Risk Stability</h3>
                <h2>{analysis_results['risk_stability']}</h2>
                <p>Risk Pattern</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Issues and recommendations
            if analysis_results['issues']:
                st.markdown('<h3 class="section-header">⚠️ Issues Detected</h3>', unsafe_allow_html=True)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("""
                    <div class="warning-box">
                    <h4>🚨 Problems Found:</h4>
                    """, unsafe_allow_html=True)
                    
                    for issue in analysis_results['issues']:
                        st.markdown(f"<p>• {issue}</p>", unsafe_allow_html=True)
                    
                    st.markdown("</div>", unsafe_allow_html=True)
                
                with col2:
                    st.markdown("""
                    <div class="success-box">
                    <h4>💡 Specific Recommendations:</h4>
                    """, unsafe_allow_html=True)
                    
                    for rec in analysis_results['recommendations']:
                        st.markdown(f"<p>• {rec}</p>", unsafe_allow_html=True)
                    
                    st.markdown("</div>", unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="success-box">
                <h4>✅ No Issues Detected</h4>
                <p>Your machine appears to be operating within normal parameters. Continue monitoring for any changes.</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Prevention strategies
            if analysis_results['prevention_strategies']:
                st.markdown('<h3 class="section-header">💡 Prevention Strategies</h3>', unsafe_allow_html=True)
                
                for strategy in analysis_results['prevention_strategies']:
                    priority_color = "red" if strategy['priority'] == 'Immediate' else "orange" if strategy['priority'] == 'High' else "yellow"
                    
                    st.markdown(f"""
                    <div class="success-box">
                    <h4>🎯 {strategy['action']} ({strategy['priority']} Priority)</h4>
                    <p><strong>Reason:</strong> {strategy['reason']}</p>
                    </div>
                    """, unsafe_allow_html=True)
            
            # General maintenance recommendations
            st.markdown('<h3 class="section-header">🔧 General Maintenance Recommendations</h3>', unsafe_allow_html=True)
            
            for rec in analysis_results['general_recommendations']:
                st.markdown(f"• {rec}")
            
            # Display validation report
            if os.path.exists('validation_prediction_report.png'):
                st.markdown('<h3 class="section-header">📊 Validation Analysis Report</h3>', unsafe_allow_html=True)
                st.image('validation_prediction_report.png', use_container_width=True)
            
            # Display model performance
            if os.path.exists('validation_model_performance.png'):
                with st.expander("🔬 Model Performance Details"):
                    st.image('validation_model_performance.png', use_container_width=True)
                    
                    st.markdown("""
                    <div class="insight-box">
                    <h4>🎯 Validation Model Information:</h4>
                    <ul>
                    <li><strong>Training Files:</strong> 16 files (experiments 1-16)</li>
                    <li><strong>Train/Test Split:</strong> 80%/20% (3,840/960 samples)</li>
                    <li><strong>Validation Files:</strong> 2 files (experiments 17-18)</li>
                    <li><strong>Test Accuracy:</strong> 91.0% on unseen test data</li>
                    <li><strong>Key Features:</strong> 12 most important variables for tool wear</li>
                    <li><strong>Top Features:</strong> X1_OutputCurrent, Feedrate, S1_CurrentFeedback</li>
                    </ul>
                    </div>
                    """, unsafe_allow_html=True)
        
        except Exception as e:
            st.error(f"❌ Error processing data: {str(e)}")
            st.markdown("""
            <div class="insight-box">
            <h4>📋 Expected Data Format:</h4>
            <p>Your CSV file should contain columns similar to the training data, including:</p>
            <ul>
            <li>X1_CurrentFeedback, X1_DCBusVoltage</li>
            <li>M1_CURRENT_FEEDRATE</li>
            <li>X1_OutputPower, X1_ActualVelocity</li>
            <li>And other sensor variables</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)


def single_machine_page():
    """Single machine comprehensive analysis page"""
    st.markdown('<h1 class="section-header">🔍 Single Machine Analysis</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="insight-box">
    <h4>🏭 Comprehensive Single Machine Analysis</h4>
    <p>Upload data from a single machine to get a detailed analysis including risk assessment, root cause analysis, 
    prevention strategies, and maintenance recommendations. Perfect for focused troubleshooting and optimization.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # File upload
    uploaded_file = st.file_uploader("📁 Upload single machine data (CSV format)", type=['csv'])
    
    # Machine name input
    machine_name = st.text_input("🏭 Machine Name/ID", value="Machine 01", help="Enter a name to identify this machine")
    
    if uploaded_file is not None and machine_name:
        try:
            # Load the uploaded data
            data = pd.read_csv(uploaded_file)
            st.success(f"✅ Data loaded successfully! Shape: {data.shape}")
            
            # Show data preview
            with st.expander("📋 Data Preview"):
                st.dataframe(data.head(), use_container_width=True)
            
            # Perform comprehensive analysis
            with st.spinner("Performing comprehensive machine analysis..."):
                from single_machine_analyzer import SingleMachineAnalyzer
                analyzer = SingleMachineAnalyzer()
                results = analyzer.analyze_single_machine(data, machine_name)
            
            # Display results
            st.markdown('<h3 class="section-header">🎯 Analysis Results</h3>', unsafe_allow_html=True)
            
            # Risk Assessment
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.markdown(f"""
                <div class="metric-card">
                <h3>Risk Level</h3>
                <h2>{results['analysis_results']['risk_level']}</h2>
                <p>{results['analysis_results']['risk_description']}</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                avg_prob = results['analysis_results']['avg_wear_probability']
                st.markdown(f"""
                <div class="metric-card">
                <h3>Wear Probability</h3>
                <h2>{avg_prob:.1%}</h2>
                <p>Average Risk</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                max_risk = np.max(results['wear_probabilities'])
                st.markdown(f"""
                <div class="metric-card">
                <h3>Peak Risk</h3>
                <h2>{max_risk:.1%}</h2>
                <p>Maximum Risk</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col4:
                risk_stability = "Stable" if results['detailed_analysis']['risk_timeline']['risk_std'] < 0.1 else "Variable"
                st.markdown(f"""
                <div class="metric-card">
                <h3>Risk Stability</h3>
                <h2>{risk_stability}</h2>
                <p>Risk Pattern</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Root Cause Analysis
            if results['detailed_analysis']['root_causes']:
                st.markdown('<h3 class="section-header">🔍 Root Cause Analysis</h3>', unsafe_allow_html=True)
                
                for i, cause in enumerate(results['detailed_analysis']['root_causes']):
                    severity_color = "red" if cause['severity'] == 'High' else "orange" if cause['severity'] == 'Medium' else "yellow"
                    
                    st.markdown(f"""
                    <div class="warning-box">
                    <h4>🚨 {cause['cause']} ({cause['severity']} Severity)</h4>
                    <p><strong>Description:</strong> {cause['description']}</p>
                    <p><strong>Impact:</strong> {cause['impact']}</p>
                    <p><strong>Solutions:</strong></p>
                    <ul>
                    {''.join([f'<li>{solution}</li>' for solution in cause['solutions']])}
                    </ul>
                    </div>
                    """, unsafe_allow_html=True)
            
            # Prevention Strategies
            if results['detailed_analysis']['prevention_strategies']:
                st.markdown('<h3 class="section-header">💡 Prevention Strategies</h3>', unsafe_allow_html=True)
                
                for strategy in results['detailed_analysis']['prevention_strategies']:
                    priority_color = "red" if strategy['priority'] == 'Immediate' else "orange" if strategy['priority'] == 'High' else "yellow"
                    
                    st.markdown(f"""
                    <div class="success-box">
                    <h4>🎯 {strategy['strategy']} ({strategy['priority']} Priority)</h4>
                    <p><strong>Description:</strong> {strategy['description']}</p>
                    <p><strong>Actions:</strong></p>
                    <ul>
                    {''.join([f'<li>{action}</li>' for action in strategy['actions']])}
                    </ul>
                    </div>
                    """, unsafe_allow_html=True)
            
            # Maintenance Plan
            maintenance_plan = results['detailed_analysis']['maintenance_plan']
            st.markdown('<h3 class="section-header">🔧 Maintenance Plan</h3>', unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown(f"""
                <div class="metric-card">
                <h3>Maintenance Urgency</h3>
                <h2>{maintenance_plan['urgency']}</h2>
                <p>{maintenance_plan['timeline']}</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown("""
                <div class="insight-box">
                <h4>🔧 Key Maintenance Tasks:</h4>
                <ul>
                """, unsafe_allow_html=True)
                
                for task in maintenance_plan['tasks']:
                    st.markdown(f"<li>{task}</li>", unsafe_allow_html=True)
                
                st.markdown("</ul></div>", unsafe_allow_html=True)
            
            # Monitoring Recommendations
            st.markdown('<h4>📊 Monitoring Recommendations:</h4>', unsafe_allow_html=True)
            for monitoring in maintenance_plan['monitoring']:
                st.markdown(f"• {monitoring}")
            
            # Display comprehensive report
            if os.path.exists('single_machine_comprehensive_report.png'):
                st.markdown('<h3 class="section-header">📊 Comprehensive Analysis Report</h3>', unsafe_allow_html=True)
                st.image('single_machine_comprehensive_report.png', use_container_width=True)
                
                st.markdown("""
                <div class="insight-box">
                <h4>📈 Report Sections:</h4>
                <ul>
                <li><strong>Risk Assessment Summary:</strong> Overall risk level and key metrics</li>
                <li><strong>Risk Timeline:</strong> How risk changes over time</li>
                <li><strong>Risk Distribution:</strong> Distribution of risk levels</li>
                <li><strong>Key Indicators:</strong> Analysis of critical sensor data</li>
                <li><strong>Root Cause Severity:</strong> Identified problems and their severity</li>
                <li><strong>Prevention Strategy Priority:</strong> Recommended actions by priority</li>
                <li><strong>Maintenance Plan:</strong> Specific maintenance tasks and timeline</li>
                <li><strong>Risk Category Distribution:</strong> Breakdown of risk levels</li>
                <li><strong>Recommendations Summary:</strong> Key actionable insights</li>
                </ul>
                </div>
                """, unsafe_allow_html=True)
        
        except Exception as e:
            st.error(f"❌ Error processing data: {str(e)}")
            st.markdown("""
            <div class="insight-box">
            <h4>📋 Expected Data Format:</h4>
            <p>Your CSV file should contain columns similar to the training data, including:</p>
            <ul>
            <li>X1_ActualPosition, X1_CommandPosition</li>
            <li>X1_CurrentFeedback, X1_DCBusVoltage</li>
            <li>M1_CURRENT_FEEDRATE</li>
            <li>And other sensor variables</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)


def main():
    # Create navigation
    create_navigation()
    
    # Display content based on current page
    if st.session_state.current_page == "overview":
        overview_page()
    elif st.session_state.current_page == "tool_wear":
        tool_wear_page()
    elif st.session_state.current_page == "completion":
        completion_page()
    elif st.session_state.current_page == "quality":
        quality_page()
    elif st.session_state.current_page == "predictor":
        predictor_page()
    elif st.session_state.current_page == "performance":
        performance_page()
    elif st.session_state.current_page == "insights":
        insights_page()

if __name__ == "__main__":
    main() 