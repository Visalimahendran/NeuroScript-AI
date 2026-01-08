import streamlit as st
import cv2
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
import threading
import queue

from src.real_time_acquisition import RealTimeHandwritingCapture
from src.real_time_analysis import RealTimeAnalysisPipeline

class RealTimeMentalHealthApp:
    """Real-time mental health assessment web application"""
    
    def __init__(self):
        st.set_page_config(
            page_title="Real-Time Mental Health Assessment",
            page_icon="🧠",
            layout="wide"
        )
        
        # Initialize components
        self.capture = None
        self.analysis_pipeline = None
        
        # State variables
        self.is_capturing = False
        self.is_analyzing = False
        
        # Data buffers
        self.latest_frame = None
        self.latest_analysis = None
        self.visualization_data = {
            'npi_history': [],
            'mh_history': [],
            'timestamps': []
        }
        
        # Threading
        self.frame_queue = queue.Queue(maxsize=5)
        self.update_thread = None
        
    def run(self):
        """Run the application"""
        st.title("🧠 Real-Time Mental Health Assessment System")
        st.markdown("### Live Handwriting Analysis with Neural Pressure Estimation")
        
        # Sidebar
        with st.sidebar:
            st.header("Controls")
            
            # Mode selection
            mode = st.radio(
                "Analysis Mode",
                ["Live Webcam", "Upload Video", "Digital Writing"],
                index=0
            )
            # ==================== LIVE WEBCAM MODE ====================
            if mode == "Live Webcam":
                st.subheader("📹 Live Webcam Settings")
                camera_id = st.number_input("Camera ID", value=0, min_value=0)
                resolution = st.selectbox("Resolution", ["640x480", "1280x720", "1920x1080"], index=1)
                
                
                
            # Settings
            st.subheader("Settings")
            fps = st.slider("FPS", 10, 60, 30)
            analysis_duration = st.slider("Analysis Duration (min)", 1, 30, 5)
            
            # Control buttons
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("▶ Start Analysis", type="primary"):
                    self.start_analysis(mode, fps)
            
            with col2:
                if st.button("⏹ Stop Analysis"):
                    self.stop_analysis()
            
            # Model selection
            st.subheader("Model Selection")
            model_type = st.selectbox(
                "Model Type",
                ["Hybrid CNN-RNN", "CNN Only", "RNN Only"],
                index=0
            )
            
            # Display info
            st.subheader("System Status")
            status_col1, status_col2 = st.columns(2)
            
            with status_col1:
                st.metric("Capture", "Active" if self.is_capturing else "Inactive")
            
            with status_col2:
                st.metric("Analysis", "Active" if self.is_analyzing else "Inactive")
        
        # Main content area
        if self.is_capturing and self.is_analyzing:
            self.display_real_time_analysis()
        else:
            self.display_welcome_screen()
    
    def start_analysis(self, mode, fps):
        """Start real-time analysis"""
        try:
            # Initialize capture
            if mode == "Live Webcam":
                self.capture = RealTimeHandwritingCapture(camera_id=0, fps=fps)
                self.capture.start_capture()
                self.is_capturing = True
                
                # Start update thread
                self.update_thread = threading.Thread(target=self._update_loop)
                self.update_thread.start()
            
            # Initialize analysis pipeline
            self.analysis_pipeline = RealTimeAnalysisPipeline()
            self.analysis_pipeline.start_analysis()
            self.is_analyzing = True
            
            st.success("Analysis started successfully!")
            
        except Exception as e:
            st.error(f"Error starting analysis: {e}")
    
    def stop_analysis(self):
        """Stop analysis"""
        if self.capture:
            self.capture.stop_capture()
            self.is_capturing = False
        
        if self.analysis_pipeline:
            self.analysis_pipeline.stop_analysis()
            self.is_analyzing = False
        
        if self.update_thread:
            self.update_thread.join()
        
        st.info("Analysis stopped")
    
    def _update_loop(self):
        """Update loop for real-time data"""
        while self.is_capturing and self.capture:
            # Get latest frame
            frame = self.capture.get_latest_frame()
            if frame is not None:
                self.latest_frame = frame
            
            # Get stroke data and analyze
            stroke_data = self.capture.get_stroke_data()
            if stroke_data and self.analysis_pipeline:
                self.analysis_pipeline.add_stroke_data(stroke_data)
                self.latest_analysis = self.analysis_pipeline.get_current_analysis()
                
                # Update visualization data
                viz_data = self.analysis_pipeline.get_visualization_data()
                if viz_data:
                    self.visualization_data = viz_data
            
            time.sleep(0.1)  # 10 Hz update rate
    
    def display_real_time_analysis(self):
        """Display real-time analysis dashboard"""
        
        # Create columns for layout
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Live video feed
            st.subheader("Live Feed")
            video_placeholder = st.empty()
            
            if self.latest_frame is not None:
                # Convert BGR to RGB for display
                rgb_frame = cv2.cvtColor(self.latest_frame, cv2.COLOR_BGR2RGB)
                video_placeholder.image(rgb_frame, channels="RGB", use_column_width=True)
        
        with col2:
            # Real-time metrics
            st.subheader("Current Metrics")
            
            if self.latest_analysis:
                analysis = self.latest_analysis
                
                # Neural Pressure Index
                npi_score = analysis['neural_pressure']['npi_score']
                npi_category = analysis['neural_pressure']['npi_category']
                
                st.metric(
                    "Neural Pressure Index (NPI)",
                    f"{npi_score:.1f}",
                    npi_category,
                    delta_color="inverse"
                )
                
                # Mental Health Risk
                mh_score = analysis['mental_health']['risk_score']
                mh_level = analysis['mental_health']['risk_level']
                
                st.metric(
                    "Mental Health Risk",
                    f"{mh_score:.1f}",
                    mh_level,
                    delta_color="inverse"
                )
                
                # Combined Risk
                combined_score = analysis['combined_risk']['score']
                combined_level = analysis['combined_risk']['level']
                
                st.metric(
                    "Combined Assessment",
                    f"{combined_score:.1f}",
                    combined_level,
                    delta_color="inverse"
                )
                
                # Confidence
                npi_confidence = analysis['neural_pressure']['npi_confidence']
                mh_confidence = analysis['mental_health']['confidence']
                
                col_conf1, col_conf2 = st.columns(2)
                with col_conf1:
                    st.metric("NPI Confidence", f"{npi_confidence:.1%}")
                with col_conf2:
                    st.metric("MH Confidence", f"{mh_confidence:.1%}")
        
        # Detailed analysis section
        st.markdown("---")
        
        if self.latest_analysis:
            self.display_detailed_analysis()
        
        # Visualization section
        st.markdown("---")
        st.subheader("Trend Analysis")
        self.display_trend_visualizations()
    
    def display_detailed_analysis(self):
        """Display detailed analysis results"""
        analysis = self.latest_analysis
        
        # Create tabs for different analysis aspects
        tab1, tab2, tab3, tab4 = st.tabs([
            "Neural Pressure",
            "Mental Health",
            "Motor Features",
            "Recommendations"
        ])
        
        with tab1:
            # Neural Pressure details
            npi = analysis['neural_pressure']
            
            st.subheader("Neural Pressure Analysis")
            
            # Gauge chart for NPI
            fig_npi = go.Figure(go.Indicator(
                mode="gauge+number",
                value=npi['npi_score'],
                title={'text': "Neural Pressure Index"},
                domain={'x': [0, 1], 'y': [0, 1]},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 25], 'color': "green"},
                        {'range': [25, 50], 'color': "lightgreen"},
                        {'range': [50, 75], 'color': "orange"},
                        {'range': [75, 100], 'color': "red"}
                    ],
                    'threshold': {
                        'line': {'color': "black", 'width': 4},
                        'thickness': 0.75,
                        'value': npi['npi_score']
                    }
                }
            ))
            
            fig_npi.update_layout(height=300)
            st.plotly_chart(fig_npi, use_container_width=True)
            
            # Contributing factors
            st.subheader("Contributing Factors")
            if 'contributing_factors' in npi:
                for factor in npi['contributing_factors']:
                    st.progress(
                        factor['contribution'] / 100,
                        text=f"{factor['factor']}: {factor['contribution']:.1f}%"
                    )
            
            # Interpretation
            st.subheader("Interpretation")
            st.info(npi['interpretation'])
        
        with tab2:
            # Mental Health details
            mh = analysis['mental_health']
            
            st.subheader("Mental Health Assessment")
            
            # Display prediction
            col_pred1, col_pred2, col_pred3 = st.columns(3)
            
            with col_pred1:
                st.metric("Prediction", mh['prediction'])
            
            with col_pred2:
                st.metric("Confidence", f"{mh['confidence']:.1%}")
            
            with col_pred3:
                st.metric("Risk Level", mh['risk_level'])
            
            # Risk score gauge
            fig_mh = go.Figure(go.Indicator(
                mode="gauge+number",
                value=mh['risk_score'],
                title={'text': "Mental Health Risk Score"},
                domain={'x': [0, 1], 'y': [0, 1]},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 30], 'color': "green"},
                        {'range': [30, 70], 'color': "orange"},
                        {'range': [70, 100], 'color': "red"}
                    ]
                }
            ))
            
            fig_mh.update_layout(height=250)
            st.plotly_chart(fig_mh, use_container_width=True)
        
        with tab3:
            # Motor features
            features = analysis['neuromotor_features']
            
            st.subheader("Neuro-Motor Features")
            
            # Key features table
            key_features = {
                'Tremor Mean': features.get('tremor_mean', 0),
                'Pressure Variability': features.get('pressure_variability', 0),
                'Curvature Instability': features.get('curvature_instability', 0),
                'Pause Ratio': features.get('pause_ratio', 0),
                'Velocity CV': features.get('velocity_cv', 0),
                'Stroke Consistency': features.get('stroke_consistency', 0)
            }
            
            # Display as metrics
            cols = st.columns(3)
            for idx, (name, value) in enumerate(key_features.items()):
                with cols[idx % 3]:
                    st.metric(name, f"{value:.3f}")
            
            # Feature radar chart
            st.subheader("Feature Radar")
            self.display_feature_radar(features)
        
        with tab4:
            # Recommendations
            st.subheader("Personalized Recommendations")
            
            if 'combined_risk' in analysis:
                st.info(analysis['combined_risk']['recommendation'])
            
            # Generate additional recommendations
            if self.analysis_pipeline:
                recommendations = self.analysis_pipeline._generate_recommendations()
                
                if recommendations:
                    st.subheader("Specific Recommendations")
                    for i, rec in enumerate(recommendations, 1):
                        st.write(f"{i}. {rec}")
            
            # Action plan
            st.subheader("Suggested Action Plan")
            
            action_items = [
                "Monitor handwriting patterns regularly",
                "Practice relaxation techniques",
                "Maintain consistent sleep schedule",
                "Engage in regular physical activity",
                "Consider professional evaluation if patterns persist"
            ]
            
            for item in action_items:
                st.checkbox(item)
    
    def display_feature_radar(self, features):
        """Display feature radar chart"""
        # Select features for radar
        radar_features = {
            'Tremor': features.get('tremor_mean', 0),
            'Pressure': features.get('pressure_variability', 0),
            'Curvature': features.get('curvature_instability', 0),
            'Pause': features.get('pause_ratio', 0),
            'Velocity': features.get('velocity_cv', 0),
            'Consistency': 1 - min(features.get('stroke_consistency', 0.5), 1)
        }
        
        # Normalize values for radar
        max_val = max(radar_features.values()) if radar_features.values() else 1
        normalized = {k: v / (max_val + 1e-10) for k, v in radar_features.items()}
        
        fig = go.Figure(data=go.Scatterpolar(
            r=list(normalized.values()),
            theta=list(normalized.keys()),
            fill='toself',
            line_color='blue'
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )
            ),
            showlegend=False,
            height=300
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def display_trend_visualizations(self):
        """Display trend visualizations"""
        
        if not self.visualization_data['npi_history']:
            st.info("Collecting data for trend analysis...")
            return
        
        # Create time series plot
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Neural Pressure Index Over Time', 'Mental Health Risk Over Time'),
            vertical_spacing=0.15
        )
        
        # NPI plot
        fig.add_trace(
            go.Scatter(
                x=self.visualization_data['timestamps'],
                y=self.visualization_data['npi_history'],
                mode='lines+markers',
                name='NPI',
                line=dict(color='red', width=2)
            ),
            row=1, col=1
        )
        
        # MH Risk plot
        fig.add_trace(
            go.Scatter(
                x=self.visualization_data['timestamps'],
                y=self.visualization_data['mh_history'],
                mode='lines+markers',
                name='MH Risk',
                line=dict(color='blue', width=2)
            ),
            row=2, col=1
        )
        
        # Update layout
        fig.update_layout(
            height=600,
            showlegend=True,
            hovermode='x unified'
        )
        
        fig.update_xaxes(title_text="Time", row=2, col=1)
        fig.update_yaxes(title_text="NPI Score", row=1, col=1)
        fig.update_yaxes(title_text="Risk Score", row=2, col=1)
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Correlation analysis
        if len(self.visualization_data['npi_history']) > 10:
            correlation = np.corrcoef(
                self.visualization_data['npi_history'],
                self.visualization_data['mh_history']
            )[0, 1]
            
            st.metric(
                "NPI-MH Correlation",
                f"{correlation:.3f}",
                "Strong correlation" if abs(correlation) > 0.7 else 
                "Moderate correlation" if abs(correlation) > 0.3 else 
                "Weak correlation"
            )
    
    def display_welcome_screen(self):
        """Display welcome screen when not analyzing"""
        
        st.header("Welcome to Real-Time Mental Health Assessment")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            ### 🎯 How It Works
            
            1. **Start Analysis** - Click the start button in the sidebar
            2. **Write Naturally** - Use paper and webcam or digital writing
            3. **Real-Time Analysis** - Get instant neural pressure and mental health assessment
            4. **Review Results** - View detailed analysis and recommendations
            
            ### 🔬 Features
            
            - **Neural Pressure Index (NPI)** - Estimated neural load from handwriting
            - **Real-Time Motor Analysis** - Tremor, pressure, curvature, pause patterns
            - **Mental Health Risk Assessment** - Stress, anxiety, depression indicators
            - **Trend Analysis** - Monitor changes over time
            - **Personalized Recommendations** - Actionable insights
            """)
        
        with col2:
            st.image("assets/system_diagram.png", caption="System Architecture")
            
            st.markdown("""
            ### 📊 What We Analyze
            
            - **Tremor & Micro-Jitter** - Fine motor control
            - **Pressure Variability** - Muscle tension
            - **Stroke Consistency** - Motor planning
            - **Hesitation Patterns** - Cognitive load
            - **Curvature Instability** - Spatial control
            """)
        
        # Quick start guide
        st.markdown("---")
        st.subheader("🚀 Quick Start Guide")
        
        guide_cols = st.columns(4)
        
        with guide_cols[0]:
            st.markdown("""
            **1. Setup**
            - Ensure good lighting
            - Position webcam clearly
            - Have paper and pen ready
            """)
        
        with guide_cols[1]:
            st.markdown("""
            **2. Calibration**
            - Click 'Start Analysis'
            - Write normally for 30 seconds
            - System auto-calibrates
            """)
        
        with guide_cols[2]:
            st.markdown("""
            **3. Analysis**
            - Write naturally
            - System analyzes in real-time
            - View live metrics
            """)
        
        with guide_cols[3]:
            st.markdown("""
            **4. Results**
            - Review detailed analysis
            - Check recommendations
            - Monitor trends
            """)
        
        # Safety disclaimer
        st.markdown("---")
        st.warning("""
        ⚠ **Important Disclaimer**
        
        This tool provides assessment based on handwriting patterns and is not a substitute 
        for professional medical diagnosis. Always consult healthcare professionals for 
        medical advice.
        """)

def main():
    """Main entry point"""
    app = RealTimeMentalHealthApp()
    app.run()

if __name__ == "__main__":
    main()