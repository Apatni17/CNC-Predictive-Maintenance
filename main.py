import pandas as pd
import streamlit as st
from utils import CNC
from types import SimpleNamespace

def main():
    st.set_page_config(layout="wide", page_title="CNC Predictive Maintenance")
    st.title('🔧 CNC Predictive Maintenance')
    st.markdown("---")

    col1, col2 = st.columns([1, 3])

    with col1:
        st.header("⚙️ Configuration")
        uploaded_file = st.file_uploader('📂 Upload File', type=['csv'], help='Upload the dataset file in CSV format.')
        if uploaded_file is not None:
            data = pd.read_csv(uploaded_file)
        else:
            st.warning("Please upload a dataset file.")
            return
        model = st.selectbox('🛠️ Model', [
            'Logistic Regression', 
            'Gaussian Naive Bayes', 
            'K-Nearest Neighbors',
            'Decision Tree',
            'Random Forest',
            'eXtreme Gradient Boosting',
            ])
        sampler = st.selectbox('📊 Sampler', [
            'None',
            'SMOTE',
            'ADASYN',
            'BorderlineSMOTE',
            'RandomOverSampler',
            'RandomUnderSampler',
            ])
        future_steps = st.number_input('🔮 Future Steps', value=1, min_value=1, max_value=10)
        window_size = st.number_input('🔢 Window Size', value=5, min_value=1, max_value=1200)
        test_size = st.number_input('📏 Test Size', value=200, min_value=1, max_value=1200)
        seed = st.number_input('🌱 Seed', value=0, min_value=0)
        corr_threshold = st.slider('📈 Correlation Threshold', 0.0, 1.0, 0.9)

        args = SimpleNamespace(
            data=data,
            model=model,
            sampler=sampler,
            future_steps=future_steps,
            window_size=window_size,
            test_size=test_size,
            seed=seed,
            corr_threshold=corr_threshold,
        )

        cnc = CNC(args)
        st.markdown("---")
        
        with st.container():
            execute_all_button = st.button('🚀 Execute All', key='execute_all', use_container_width=True)
            
        if execute_all_button:
            fig = cnc.pre_process()
            st.success("Pre-processing done. ✅")
            with col2:
                st.pyplot(fig)

            fig = cnc.train()
            st.success("Training done. ✅")
            if fig is not None:
                with col2:
                    st.pyplot(fig)

            figs = cnc.evaluate()
            st.success("Evaluation done. ✅")
            with col2:
                for fig in figs:
                    st.pyplot(fig)

        with col2:
            st.markdown("### Results 📊")
            st.markdown("The results of the CNC predictive maintenance will be displayed here after executing all steps.")

if __name__ == "__main__":
    main()
