import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from tool_wear_analysis import ToolWearAnalyzer

# Set page config
st.set_page_config(
    page_title="CNC Tool Wear Analysis Dashboard",
    page_icon="🔧",
    layout="wide",
    initial_sidebar_state="collapsed"
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
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin: 0.5rem 0;
    }
    .insight-box {
        background-color: #e8f4fd;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #ff7f0e;
        margin: 1rem 0;
    }
    .nav-button {
        background-color: #1f77b4;
        color: white;
        padding: 0.5rem 1rem;
        border: none;
        border-radius: 0.3rem;
        margin: 0.2rem;
        cursor: pointer;
        font-weight: bold;
    }
    .nav-button:hover {
        background-color: #0d5aa7;
    }
    .nav-button.active {
        background-color: #ff7f0e;
    }
    .page-container {
        padding: 2rem;
        background-color: white;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

def create_navigation():
    """Create navigation buttons at the top"""
    st.markdown("""
    <div style="text-align: center; padding: 1rem; background-color: #f8f9fa; border-radius: 0.5rem; margin-bottom: 2rem;">
        <h2 style="margin-bottom: 1rem;">🔧 CNC Tool Wear Analysis Dashboard</h2>
    </div>
    """, unsafe_allow_html=True)
    
    # Navigation buttons
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    
    with col1:
        if st.button("📊 Overview", key="overview"):
            st.session_state.page = "overview"
    with col2:
        if st.button("🔗 Correlations", key="correlations"):
            st.session_state.page = "correlations"
    with col3:
        if st.button("🎯 Features", key="features"):
            st.session_state.page = "features"
    with col4:
        if st.button("📈 Performance", key="performance"):
            st.session_state.page = "performance"
    with col5:
        if st.button("📋 Statistics", key="statistics"):
            st.session_state.page = "statistics"
    with col6:
        if st.button("💡 Insights", key="insights"):
            st.session_state.page = "insights"
    
    # Initialize page if not set
    if 'page' not in st.session_state:
        st.session_state.page = "overview"

def overview_page():
    """Overview page with key metrics and experiment info"""
    st.markdown('<h1 class="main-header">📊 Overview</h1>', unsafe_allow_html=True)
    
    # Check if analysis data exists
    if not os.path.exists('tool_wear_statistics.csv'):
        st.error("❌ Analysis data not found. Please run the tool wear analysis first.")
        return
    
    # Load data
    stats = pd.read_csv('tool_wear_statistics.csv', index_col=0)
    
    # Key Metrics Section
    st.markdown("## 🎯 Key Performance Metrics")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h3>🎯 ROC AUC Score</h3>
            <h2>0.998</h2>
            <p>Exceptional Performance</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h3>📈 Classification Accuracy</h3>
            <h2>98%</h2>
            <p>Excellent Detection</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
            <h3>🔬 Data Points Analyzed</h3>
            <h2>5,400</h2>
            <p>From 18 Experiments</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class="metric-card">
            <h3>🎛️ Key Features</h3>
            <h2>45</h2>
            <p>Selected Variables</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Experiment Overview
    st.markdown("## 🧪 Experiment Overview")
    
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

def correlations_page():
    """Correlation analysis page"""
    st.markdown('<h1 class="main-header">🔗 Correlation Analysis</h1>', unsafe_allow_html=True)
    
    if not os.path.exists('correlation_matrix.png'):
        st.error("❌ Correlation matrix not found. Please run the analysis first.")
        return
    
    st.markdown("### Top Correlations with Tool Wear")
    
    # Display correlation matrix
    st.image('correlation_matrix.png', use_container_width=True)
    
    # Top correlations explanation
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
    
    st.markdown("""
    <div class="insight-box">
    <h4>⚡ ACTUAL PERCENTAGE DIFFERENCES (Most Important):</h4>
    <ol>
    <li><strong>Y1 Current Feedback: 75.2%</strong> - Highest difference!</li>
    <li><strong>Current Feedrate: 29.6%</strong></li>
    <li><strong>S1 Current Feedback: -28.0%</strong></li>
    <li><strong>S1 DC Bus Voltage: -26.9%</strong></li>
    <li><strong>X1 DC Bus Voltage: -10.5%</strong></li>
    </ol>
    <p><strong>Note:</strong> Current feedback and bus voltage show the largest actual differences between worn and unworn tools!</p>
    </div>
    """, unsafe_allow_html=True)

def features_page():
    """Feature importance and analysis page"""
    st.markdown('<h1 class="main-header">🎯 Feature Analysis</h1>', unsafe_allow_html=True)
    
    if not os.path.exists('feature_importance.png'):
        st.error("❌ Feature importance data not found. Please run the analysis first.")
        return
    
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
        
        st.markdown("""
        <div class="insight-box">
        <h4>⚡ ACTUAL DATA DIFFERENCES (Raw Changes):</h4>
        <ol>
        <li><strong>Y1 Current Feedback: 75.2%</strong> - Highest actual difference!</li>
        <li><strong>S1 Current Feedback: -28.0%</strong> - Significant decrease</li>
        <li><strong>S1 DC Bus Voltage: -26.9%</strong> - Voltage drops with wear</li>
        <li><strong>X1 DC Bus Voltage: -10.5%</strong> - Consistent pattern</li>
        </ol>
        <p><strong>Key Insight:</strong> Current feedback and bus voltage show the largest actual changes, making them excellent direct indicators for monitoring!</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Feature distributions
    if os.path.exists('feature_distributions.png'):
        st.markdown("## 📊 Feature Distribution Analysis")
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

def performance_page():
    """Model performance page"""
    st.markdown('<h1 class="main-header">📈 Model Performance</h1>', unsafe_allow_html=True)
    
    if not os.path.exists('roc_curve.png'):
        st.error("❌ ROC curve data not found. Please run the analysis first.")
        return
    
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
    
    # Time series analysis
    if os.path.exists('time_series_comparison.png'):
        st.markdown("## ⏰ Time Series Patterns")
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

def statistics_page():
    """Detailed statistics page"""
    st.markdown('<h1 class="main-header">📋 Detailed Statistics</h1>', unsafe_allow_html=True)
    
    if not os.path.exists('tool_wear_statistics.csv'):
        st.error("❌ Statistics data not found. Please run the analysis first.")
        return
    
    # Load data
    stats = pd.read_csv('tool_wear_statistics.csv', index_col=0)
    
    # Show key statistics
    st.markdown("## 📊 Key Statistical Summary")
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
    
    # Full statistics
    st.markdown("## 📈 Complete Statistics Table")
    st.dataframe(stats, use_container_width=True)

def insights_page():
    """Research insights and recommendations page"""
    st.markdown('<h1 class="main-header">💡 Research Insights</h1>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="insight-box">
        <h4>✅ Research Questions Answered:</h4>
        
        <h5>1. "Are there specific patterns in current feedback before wear?"</h5>
        <p><strong>YES</strong> - Y1_CurrentFeedback is 75% higher in worn tools</p>
        
        <h5>2. "Can cutting forces explain tool wear differences?"</h5>
        <p><strong>YES</strong> - X1_OutputCurrent is the 2nd most important predictor</p>
        
        <h5>3. "How do feedrate and tool wear interact?"</h5>
        <p><strong>ANSWERED</strong> - M1_CURRENT_FEEDRATE is the most critical predictor</p>
        
        <h5>4. "Do voltage patterns indicate wear?"</h5>
        <p><strong>YES</strong> - S1 DC Bus Voltage shows -26.9% difference</p>
        
        <h5>5. "Are position deviations related to wear?"</h5>
        <p><strong>YES</strong> - Multiple position variables show strong correlations</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="insight-box">
        <h4>🎯 Predictive Maintenance Recommendations:</h4>
        
        <ol>
        <li><strong>Monitor feedrate reductions</strong> - Primary indicator</li>
        <li><strong>Track current feedback increases</strong> - Early warning signals</li>
        <li><strong>Watch for voltage spikes</strong> - Precursor to failure</li>
        <li><strong>Monitor position accuracy</strong> - Deviations indicate wear</li>
        <li><strong>Set automated thresholds</strong> based on correlation patterns</li>
        <li><strong>Implement real-time monitoring</strong> for Y1 current feedback</li>
        <li><strong>Create alert systems</strong> for voltage drops</li>
        </ol>
        </div>
        """, unsafe_allow_html=True)
    
    # Next steps
    st.markdown("## 🔬 Next Research Questions")
    
    st.markdown("""
    <div class="insight-box">
    <h4>Advanced Questions to Explore:</h4>
    <ol>
    <li><strong>Temporal Patterns:</strong> Do current feedback changes follow predictable time patterns before failure?</li>
    <li><strong>Variable Interactions:</strong> How do current feedback and feedrate interact to indicate different wear stages?</li>
    <li><strong>Wear Progression:</strong> Can we identify distinct phases of tool wear based on variable patterns?</li>
    <li><strong>Physical Mechanisms:</strong> Why does Y1 current feedback show 75% difference while X1 shows only 3%?</li>
    <li><strong>Real-time Scoring:</strong> Can we create a system that updates wear probability as new data comes in?</li>
    </ol>
    </div>
    """, unsafe_allow_html=True)

def main():
    # Create navigation
    create_navigation()
    
    # Check if analysis has been run
    if not os.path.exists('tool_wear_statistics.csv'):
        st.error("❌ Tool wear analysis data not found. Please run the analysis first.")
        
        if st.button("🔄 Run Tool Wear Analysis"):
            with st.spinner("Running comprehensive tool wear analysis..."):
                try:
                    analyzer = ToolWearAnalyzer()
                    analyzer.run_complete_analysis()
                    st.success("✅ Analysis completed successfully!")
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ Error running analysis: {str(e)}")
        return
    
    # Display appropriate page based on selection
    if st.session_state.page == "overview":
        overview_page()
    elif st.session_state.page == "correlations":
        correlations_page()
    elif st.session_state.page == "features":
        features_page()
    elif st.session_state.page == "performance":
        performance_page()
    elif st.session_state.page == "statistics":
        statistics_page()
    elif st.session_state.page == "insights":
        insights_page()

if __name__ == "__main__":
    main() 