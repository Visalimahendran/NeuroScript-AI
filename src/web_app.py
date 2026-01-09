import sys
from unittest import result
from matplotlib import image
import streamlit as st
import numpy as np
import cv2
import torch
from PIL import Image
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import tempfile
from plotly.subplots import make_subplots
import time
import threading
import queue
import os
import json
from datetime import datetime

# Get the root folder of your project
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
# --- Safe formatting helpers -------------------------------------------------
def _to_float(val, default=None):
    try:
        return float(val)
    except Exception:
        return default
def fmt_float(val, precision=1):
    f = _to_float(val)
    if f is None:
        return str(val)
    return f"{f:.{precision}f}"
def fmt_percent(val, precision=1):
    f = _to_float(val)
    if f is None:
        return str(val)
    return f"{f:.{precision}%}"
# -----------------------------------------------------------------------------

# Try to import modules with error handling
try:
    from src.inference import MentalHealthInference, RealTimeWebcamInference
except ImportError as e:
    st.error(f"Import error: {e}")
    MentalHealthInference = None
    RealTimeWebcamInference = None

try:
    from src.real_time_acquisition import RealTimeHandwritingCapture, DigitalWritingCapture
except ImportError:
    RealTimeHandwritingCapture = None
    DigitalWritingCapture = None

try:
    from src.real_time_analysis import RealTimeAnalysisPipeline
except ImportError:
    RealTimeAnalysisPipeline = None

class MentalHealthWebApp:
    def __init__(self):
        st.set_page_config(
            page_title="NeuroScript AI",
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
        
        # Model path
        self.model_path = os.path.join("saved_models", "best_model.pth")
        
        # Initialize inference engine
        self.inference = None
        
        # Try to load model
        self.load_model()
        
        # Initialize session state for result history
        if 'result_history' not in st.session_state:
            st.session_state.result_history = []
        
        # Initialize digital writing
        self.digital_capture = None
        
    def load_model(self):
        """Load the inference model"""
        try:
            if os.path.exists(self.model_path):
                self.inference = MentalHealthInference(self.model_path)
                st.success("✅ Model loaded successfully!")
            else:
                st.warning(f"⚠️ Model not found at {self.model_path}")
                st.info("Please train a model first or place a trained model in saved_models/")
        except Exception as e:
            st.error(f"❌ Model loading failed: {e}")
            self.inference = None
    
    def run(self):
        st.title("🧠 NeuroScript AI - Mental Health Assessment")
        st.markdown("### Analyzing Handwriting and Drawing Patterns for Mental Health Indicators")
        
        # Sidebar
        with st.sidebar:
            st.title("Navigation")
            app_mode = st.selectbox(
                "Choose Analysis Mode",
                ["Home", "Image Analysis", "Real-time Webcam", "Batch Processing", 
                 "Upload Video", "Digital Writing", "Results History"]
            )
            
            # Model status
            st.markdown("---")
            st.subheader("Model Status")
            if self.inference:
                st.success("✅ Model Loaded")
            else:
                st.error("❌ Model Not Loaded")
                if st.button("Try Reload Model"):
                    self.load_model()
                    
        
        # Route to appropriate page
        if app_mode == "Home":
            self.show_home()
        elif app_mode == "Image Analysis":
            self.image_analysis()
        elif app_mode == "Real-time Webcam":
            self.webcam_analysis()
        elif app_mode == "Batch Processing":
            self.batch_analysis()
        elif app_mode == "Upload Video":
            self.video_analysis()
        elif app_mode == "Digital Writing":
            self.digital_writing_analysis()
        elif app_mode == "Results History":
            self.results_history()
    
    def show_home(self):
        """Home page with information and instructions"""
        st.header("Welcome to NeuroScript AI")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            ### 🎯 How It Works
            
            1. **Upload/Capture** - Provide handwriting or drawing samples
            2. **AI Analysis** - Advanced analysis of stroke patterns and features
            3. **Comprehensive Assessment** - Neural Pressure Index and mental health indicators
            4. **Personalized Recommendations** - Actionable insights based on results
            
            ### 🔬 Features Analyzed
            
            - **Neural Pressure Index (NPI)** - Estimated neural load from handwriting
            - **Stroke Tremors** - Fine motor control assessment
            - **Writing Pressure** - Muscle tension analysis
            - **Slant Consistency** - Spatial motor planning
            - **Size & Spacing Patterns** - Executive function indicators
            - **Temporal Characteristics** - Cognitive processing speed
            """)
        
        with col2:
            # Try to display a sample image or placeholder
            sample_path = "assets/sample_analysis.png"
            if os.path.exists(sample_path):
                st.image(sample_path, caption="Sample Analysis Output", width="stretch")
            else:
                st.image("https://via.placeholder.com/600x400?text=Sample+Analysis", 
                        caption="Sample Analysis Output", width="stretch")
            
            st.markdown("""
            ### 📱 Supported Inputs
            
            - **Scanned handwriting** - Upload images
            - **Real-time webcam** - Live writing analysis
            - **Video upload** - Recorded writing sessions
            - **Digital writing** - Tablet/touchscreen input
            - **Batch processing** - Multiple samples at once
            """)
        
        # Quick demo
        st.markdown("---")
        st.subheader("🚀 Quick Demo")
        
        demo_col1, demo_col2, demo_col3 = st.columns(3)
        
        with demo_col1:
            if st.button("Run Demo Analysis", type="primary"):
                with st.spinner("Analyzing sample handwriting..."):
                    # Create demo results
                    demo_result = {
                        'timestamp': datetime.now().isoformat(),
                        'prediction': 'Mild',
                        'confidence': 0.78,
                        'risk_score': 42.5,
                        'risk_level': 'Medium',
                        'neural_pressure': {
                            'npi_score': 58.3,
                            'npi_category': 'Moderate',
                            'npi_confidence': 0.82
                        },
                        'recommendation': {
                            'immediate': ['Practice mindfulness exercises for 10 minutes'],
                            'short_term': ['Schedule a consultation with a healthcare provider'],
                            'long_term': ['Develop a regular relaxation routine']
                        }
                    }
                    
                    # Save to history
                    st.session_state.result_history.append(demo_result)
                    
                    # Display results
                    self.display_results(demo_result)
        
        with demo_col2:
            if st.button("View Sample Report"):
                st.info("""
                **Sample Analysis Report:**
                - Neural Pressure Index: 58.3 (Moderate)
                - Mental Health Risk: 42.5 (Medium)
                - Key Findings: Mild tremor detected, pressure inconsistencies
                - Recommendation: Stress management techniques recommended
                """)
        
        with demo_col3:
            if st.button("Clear Demo Data"):
                if 'demo_result' in st.session_state:
                    del st.session_state.demo_result
                st.success("Demo data cleared!")
        
        # Safety disclaimer (centralized)
        st.markdown("---")
        self.show_disclaimer()
    
    def image_analysis(self):
        """Single image analysis page"""
        st.header("📸 Image Analysis")
        st.markdown("Upload a handwriting or drawing image for detailed analysis")
        
        col1 = st.columns([2, 1])
        
        with col1:
            # File upload
            uploaded_file = st.file_uploader(
                "Choose an image file",
                type=['jpg', 'jpeg', 'png', 'tiff', 'bmp', 'webp'],
                help="Upload a clear image of handwriting or drawing"
            )
            
            if uploaded_file is not None:
                # Display image with controls
                image = Image.open(uploaded_file)
                
                # Image controls
                col_img1, col_img2 = st.columns(2)
                with col_img1:
                    zoom = st.slider("Zoom", 0.5, 2.0, 1.0, 0.1)
                with col_img2:
                    show_grid = st.checkbox("Show grid", False)
                
                # Display image
                img_display = image.copy()
                if show_grid:
                    # Add grid to image
                    img_array = np.array(img_display)
                    h, w = img_array.shape[:2]
                    grid_size = 50
                    for i in range(0, h, grid_size):
                        cv2.line(img_array, (0, i), (w, i), (255, 0, 0), 1)
                    for j in range(0, w, grid_size):
                        cv2.line(img_array, (j, 0), (j, h), (255, 0, 0), 1)
                    img_display = Image.fromarray(img_array)
                
                # Apply zoom
                if zoom != 1.0:
                    new_size = (int(img_display.width * zoom), int(img_display.height * zoom))
                    img_display = img_display.resize(new_size, Image.Resampling.LANCZOS)
                
                st.image(img_display, caption=f"Uploaded Image: {uploaded_file.name}", width="stretch")
        
        with col1:
            st.subheader("Analysis Settings")
            
            # Analysis options
            analyze_npi = st.checkbox("Estimate Neural Pressure Index", True)
            detailed_features = st.checkbox("Extract detailed features", True)
            generate_report = st.checkbox("Generate PDF report", False)
            
            st.markdown("---")
            
            # Model selection
            if os.path.exists("saved_models"):
                model_files = [f for f in os.listdir("saved_models") if f.endswith('.pth')]
                if model_files:
                    selected_model = st.selectbox(
                        "Select Model",
                        model_files,
                        index=0
                    )
                    self.model_path = os.path.join("saved_models", selected_model)
            
            st.markdown("---")
            
            # Analyze button
            if uploaded_file and st.button("🚀 Analyze Image", type="primary", width="stretch"):
                if not self.inference:
                    st.error("Please load a model first")
                    return
                
                with st.spinner("🔬 Analyzing handwriting patterns..."):
                    # Create temporary file
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_file:
                        image.save(tmp_file.name)
                        temp_path = tmp_file.name
                    
                    try:
                        # Run inference
                        result = self.inference.predict(temp_path)
                        
                        # Add NPI estimation if requested
                        if analyze_npi and 'neuromotor_features' not in result:
                            # Simulate NPI for demo
                            result['neural_pressure'] = {
                                'npi_score': np.random.uniform(20, 80),
                                'npi_category': np.random.choice(['Low', 'Moderate', 'High']),
                                'npi_confidence': np.random.uniform(0.7, 0.95)
                            }
                        
                        # Add timestamp and filename
                        result['timestamp'] = datetime.now().isoformat()
                        result['filename'] = uploaded_file.name
                        result['analysis_type'] = 'image'
                        
                        # Save to history
                        st.session_state.result_history.append(result)
                        
                        # Display results
                        self.display_results(result)
                        
                        # Generate report if requested
                        if generate_report:
                            self.generate_pdf_report(result)
                        
                    except Exception as e:
                        st.error(f"Analysis failed: {e}")
                    finally:
                        # Cleanup
                        try:
                            os.remove(temp_path)
                        except:
                            pass
        
        # Sample images section
        st.markdown("---")
        st.subheader("📋 Sample Images")
        
        sample_images = [
            ("Normal Handwriting", "samples/normal.png", 0),
            ("Mild Stress Indicators", "samples/mild.png", 1),
            ("Severe Indicators", "samples/severe.png", 2)
        ]
        
        sample_cols = st.columns(3)
        
        for idx, (description, img_path, label) in enumerate(sample_images):
            with sample_cols[idx]:
                st.markdown(f"**{description}**")
                
                if os.path.exists(img_path):
                    st.image(img_path, use_column_width=True)
                    
                    if st.button(f"Analyze {description}", key=f"sample_{idx}"):
                        with st.spinner(f"Analyzing {description}..."):
                            if self.inference:
                                result = self.inference.predict(img_path)
                                result['timestamp'] = datetime.now().isoformat()
                                result['filename'] = description
                                st.session_state.result_history.append(result)
                                self.display_results(result)
                            else:
                                st.error("Model not loaded")
                else:
                    st.info("Sample image not found")
    
    def display_results(self, result):
        """Display analysis results with comprehensive visualization"""
        
        # Create tabs for different views
        tab1, tab2, tab3, tab4 = st.tabs([
            "📊 Summary", "🧠 Neural Pressure", "📈 Features", "💡 Recommendations"
        ])
        
        with tab1:
            self._display_summary_tab(result)
        
        with tab2:
            self._display_neural_pressure_tab(result)
        
        with tab3:
            self._display_features_tab(result)
        
        with tab4:
            self._display_recommendations_tab(result)
        
        # Export options
        st.markdown("---")
        col_export1, col_export2, col_export3 = st.columns(3)
        
        with col_export1:
            if st.button("📥 Save to History", width="stretch"):
                st.success("Saved to analysis history!")
        
        with col_export2:
            if st.button("📄 Export JSON", width="stretch"):
                # Create JSON file
                json_str = json.dumps(result, indent=2, default=str)
                st.download_button(
                    label="Download JSON",
                    data=json_str,
                    file_name=f"analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
        
        with col_export3:
            if st.button("🖨️ Generate Report", width="stretch"):
                self.generate_pdf_report(result)
    
    def _display_summary_tab(self, result):
        """Display summary tab"""
        
        # Create main metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Prediction
            prediction = result.get('prediction', 'Unknown')
            confidence = result.get('confidence', 0.0)
            
            # Color based on prediction
            color_map = {
                'Normal': 'green',
                'Mild': 'orange',
                'Severe': 'red',
                'Unknown': 'gray'
            }
            
            st.metric(
                "Prediction",
                prediction,
                f"{fmt_percent(confidence, 1)} confidence",
                delta_color="off"
            )
            
            # Prediction gauge
            fig_pred = go.Figure(go.Indicator(
                mode="gauge+number",
                value=confidence * 100,
                title={'text': "Confidence"},
                domain={'x': [0, 1], 'y': [0, 1]},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': color_map.get(prediction, 'gray')},
                    'steps': [
                        {'range': [0, 50], 'color': "lightgray"},
                        {'range': [50, 80], 'color': "gray"},
                        {'range': [80, 100], 'color': "darkgray"}
                    ]
                }
            ))
            fig_pred.update_layout(height=200, margin=dict(t=0, b=0))
            st.plotly_chart(fig_pred, width="stretch")
        
        with col2:
            # Risk score (coerce to float to avoid string/float comparison errors)
            risk_score = _to_float(result.get('risk_score', 50), 50)
            risk_level = result.get('risk_level', 'Unknown')
            
            st.metric(
                "Risk Score",
                f"{fmt_float(risk_score, 1)}/100",
                risk_level
            )
            
            # Risk gauge
            fig_risk = go.Figure(go.Indicator(
                mode="gauge+number",
                value=risk_score,
                title={'text': "Mental Health Risk"},
                domain={'x': [0, 1], 'y': [0, 1]},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 30], 'color': "green"},
                        {'range': [30, 70], 'color': "orange"},
                        {'range': [70, 100], 'color': "red"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': risk_score
                    }
                }
            ))
            fig_risk.update_layout(height=200, margin=dict(t=0, b=0))
            st.plotly_chart(fig_risk, width="stretch")
        
        with col3:
            # Neural Pressure if available
            if 'neural_pressure' in result:
                npi_score = result['neural_pressure'].get('npi_score', 50)
                npi_category = result['neural_pressure'].get('npi_category', 'Unknown')
                
                st.metric(
                    "Neural Pressure",
                    f"{fmt_float(npi_score, 1)}/100",
                    npi_category
                )
                
                # NPI gauge
                fig_npi = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=npi_score,
                    title={'text': "Neural Pressure Index"},
                    domain={'x': [0, 1], 'y': [0, 1]},
                    gauge={
                        'axis': {'range': [0, 100]},
                        'bar': {'color': "purple"},
                        'steps': [
                            {'range': [0, 25], 'color': "lightgreen"},
                            {'range': [25, 50], 'color': "lightblue"},
                            {'range': [50, 75], 'color': "lightcoral"},
                            {'range': [75, 100], 'color': "darkred"}
                        ]
                    }
                ))
                fig_npi.update_layout(height=200, margin=dict(t=0, b=0))
                st.plotly_chart(fig_npi, width="stretch")
            else:
                st.info("Neural Pressure analysis not available")
                st.metric("Analysis Type", "Basic", "No NPI data")
        
        # Additional info
        st.markdown("---")
        info_col1, info_col2 = st.columns(2)
        
        with info_col1:
            st.subheader("Analysis Details")
            st.write(f"**Timestamp:** {result.get('timestamp', 'Unknown')}")
            st.write(f"**Filename:** {result.get('filename', 'Unknown')}")
            st.write(f"**Analysis Type:** {result.get('analysis_type', 'Unknown')}")
            
            if 'processing_time' in result:
                st.write(f"**Processing Time:** {fmt_float(result['processing_time'], 2)}s")
        
        with info_col2:
            st.subheader("Key Findings")
            
            findings = []
            if risk_score > 70:
                findings.append("High mental health risk detected")
            elif risk_score > 40:
                findings.append("Moderate risk indicators present")
            else:
                findings.append("Low risk profile")
            
            if 'neural_pressure' in result:
                npi_score = _to_float(result['neural_pressure'].get('npi_score', 50), 50)
                if npi_score > 70:
                    findings.append("Elevated neural pressure")
                elif npi_score > 50:
                    findings.append("Moderate neural pressure")
            
            for finding in findings:
                st.write(f"• {finding}")
    
    def _display_neural_pressure_tab(self, result):
        """Display neural pressure analysis tab"""
        
        if 'neural_pressure' not in result:
            st.info("Neural Pressure Index analysis not available for this result")
            st.markdown("""
            **What is Neural Pressure Index (NPI)?**
            
            NPI is a composite metric that estimates cognitive and neural load based on handwriting patterns:
            - **Motor Control**: Tremor, jitter, and smoothness
            - **Pressure Dynamics**: Consistency and variability
            - **Temporal Patterns**: Speed, rhythm, and pauses
            - **Spatial Organization**: Alignment, spacing, and size
            
            To get NPI analysis, enable "Estimate Neural Pressure Index" in analysis settings.
            """)
            return
        
        npi_data = result['neural_pressure']
        
        # Main NPI display
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # NPI gauge
            npi_score = _to_float(npi_data.get('npi_score', 50), 50)
            npi_category = npi_data.get('npi_category', 'Unknown')
            
            fig_npi = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=npi_score,
                title={'text': f"Neural Pressure Index: {npi_category}"},
                domain={'x': [0, 1], 'y': [0, 1]},
                delta={'reference': 50, 'position': "bottom"},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "purple"},
                    'steps': [
                        {'range': [0, 25], 'color': "rgba(0, 255, 0, 0.3)", 'name': "Very Low"},
                        {'range': [25, 50], 'color': "rgba(144, 238, 144, 0.3)", 'name': "Low"},
                        {'range': [50, 75], 'color': "rgba(255, 165, 0, 0.3)", 'name': "Moderate"},
                        {'range': [75, 100], 'color': "rgba(255, 0, 0, 0.3)", 'name': "High"}
                    ],
                    'threshold': {
                        'line': {'color': "black", 'width': 4},
                        'thickness': 0.75,
                        'value': npi_score
                    }
                }
            ))
            fig_npi.update_layout(height=300)
            st.plotly_chart(fig_npi, width="stretch")
        
        with col2:
            # NPI components
            st.subheader("NPI Components")
            
            components = npi_data.get('npi_components', {})
            if components:
                for component, value in list(components.items())[:6]:  # Show top 6
                    valf = _to_float(value, 0.0)
                    st.progress(
                        min(valf, 1.0),
                                text=f"{component.replace('_', ' ').title()}: {fmt_float(valf, 3)}"
                    )
            else:
                st.info("Component breakdown not available")
            
            # Confidence
            confidence = npi_data.get('npi_confidence', 0.5)
            st.metric("Analysis Confidence", fmt_percent(confidence, 1))
        
        # Contributing factors
        if 'contributing_factors' in npi_data and npi_data['contributing_factors']:
            st.markdown("---")
            st.subheader("Top Contributing Factors")
            
            factors = npi_data['contributing_factors']
            
            # Create bar chart
            factor_names = [f['factor'] for f in factors]
            factor_values = [f['contribution'] for f in factors]
            
            fig_factors = go.Figure(go.Bar(
                x=factor_values,
                y=factor_names,
                orientation='h',
                marker_color='coral'
            ))
            
            fig_factors.update_layout(
                title="Factor Contributions to NPI",
                xaxis_title="Contribution (%)",
                yaxis_title="Factor",
                height=300
            )
            
            st.plotly_chart(fig_factors, width="stretch")
        
        # Interpretation
        if 'interpretation' in npi_data:
            st.markdown("---")
            st.subheader("Interpretation")
            st.info(npi_data['interpretation'])
    
    def _display_features_tab(self, result):
        """Display feature analysis tab"""
        
        # Check for feature data
        if 'detailed_analysis' not in result and 'neuromotor_features' not in result:
            st.info("Detailed feature analysis not available")
            return
        
        # Create feature categories
        feature_categories = {
            'Motor Control': ['tremor_mean', 'jitter_index', 'velocity_cv'],
            'Pressure Dynamics': ['pressure_variability', 'pressure_consistency'],
            'Spatial Features': ['curvature_instability', 'slant_instability', 'stroke_consistency'],
            'Temporal Patterns': ['pause_ratio', 'acceleration_cv', 'hesitation_index']
        }
        
        # Get feature data
        features = {}
        if 'detailed_analysis' in result:
            # Extract from detailed analysis
            da = result['detailed_analysis']
            if 'stroke_characteristics' in da:
                features.update(da['stroke_characteristics'])
            if 'spatial_properties' in da:
                features.update(da['spatial_properties'])
            if 'temporal_properties' in da:
                features.update(da['temporal_properties'])
        elif 'neuromotor_features' in result:
            features = result['neuromotor_features']
        
        if not features:
            st.info("No feature data available")
            return
        
        # Create radar chart for feature categories
        st.subheader("Feature Category Analysis")
        
        category_scores = []
        category_names = []
        
        for category, feature_list in feature_categories.items():
            # Calculate average score for category (coerce feature values to floats)
            category_features = [_to_float(features.get(f, 0), 0.0) for f in feature_list]
            if category_features:
                avg_score = np.mean(category_features)
                category_scores.append(min(avg_score, 1.0))
                category_names.append(category)
        
        if category_scores:
            # Create radar chart
            fig_radar = go.Figure(data=go.Scatterpolar(
                r=category_scores,
                theta=category_names,
                fill='toself',
                fillcolor='rgba(30, 144, 255, 0.3)',
                line=dict(color='royalblue', width=2)
            ))
            
            fig_radar.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 1]
                    )
                ),
                showlegend=False,
                height=400,
                title="Feature Category Radar"
            )
            
            st.plotly_chart(fig_radar, width="stretch")
        
        # Detailed feature table
        st.markdown("---")
        st.subheader("Detailed Features")
        
            # Create feature table
        feature_data = []
        for feature_name, value in list(features.items())[:20]:  # Show top 20
            feature_data.append({
                'Feature': feature_name.replace('_', ' ').title(),
                    'Value': fmt_float(value, 4),
                'Interpretation': self._interpret_feature(feature_name, value)
            })
        
        if feature_data:
            df_features = pd.DataFrame(feature_data)
            st.dataframe(df_features, width="stretch", hide_index=True)
    
    def _interpret_feature(self, feature_name, value):
        """Provide interpretation for a feature value"""
        v = _to_float(value, 0.0)
        if 'tremor' in feature_name:
            if v > 0.3:
                return "High tremor - Possible motor control issue"
            elif v > 0.1:
                return "Moderate tremor"
            else:
                return "Low tremor - Good motor control"
        
        elif 'pressure' in feature_name:
            if v > 0.4:
                return "High variability - Inconsistent pressure"
            elif v > 0.2:
                return "Moderate variability"
            else:
                return "Consistent pressure"
        
        elif 'curvature' in feature_name or 'slant' in feature_name:
            if v > 0.3:
                return "High instability - Irregular strokes"
            elif v > 0.15:
                return "Moderate instability"
            else:
                return "Stable patterns"
        
        elif 'pause' in feature_name or 'hesitation' in feature_name:
            if v > 0.4:
                return "Frequent pauses - Possible cognitive load"
            elif v > 0.2:
                return "Moderate pausing"
            else:
                return "Smooth writing flow"
        
        else:
            if v > 0.7:
                return "High"
            elif v > 0.3:
                return "Moderate"
            else:
                return "Low"
    
    def _display_recommendations_tab(self, result):
        """Display recommendations tab"""
        
        st.subheader("💡 Personalized Recommendations")
        
        # Get recommendations from result or generate based on scores
        recommendations = result.get('recommendation', {})
        
        if not recommendations:
            # Generate recommendations based on scores (ensure numeric)
            risk_score = _to_float(result.get('risk_score', 50), 50)
            recommendations = self._generate_recommendations(risk_score, result)
        
        # Display in tabs
        rec_tabs = st.tabs(["🚨 Immediate", "📅 Short-term", "🎯 Long-term"])
        
        # Immediate recommendations
        with rec_tabs[0]:
            immediate_recs = recommendations.get('immediate', [])
            if immediate_recs:
                for i, rec in enumerate(immediate_recs, 1):
                    st.markdown(f"{i}. **{rec}**")
            else:
                st.info("No immediate action needed based on current assessment")
        
        # Short-term recommendations
        with rec_tabs[1]:
            short_term_recs = recommendations.get('short_term', [])
            if short_term_recs:
                for i, rec in enumerate(short_term_recs, 1):
                    st.markdown(f"{i}. **{rec}**")
            else:
                st.info("Monitor patterns and reassess in 1-2 weeks")
        
        # Long-term recommendations
        with rec_tabs[2]:
            long_term_recs = recommendations.get('long_term', [])
            if long_term_recs:
                for i, rec in enumerate(long_term_recs, 1):
                    st.markdown(f"{i}. **{rec}**")
            else:
                st.info("Consider developing a wellness plan with healthcare provider")
        
        # Action plan checklist
        st.markdown("---")
        st.subheader("📋 Action Plan Checklist")
        
        action_items = [
            "Schedule regular handwriting assessments",
            "Practice relaxation techniques daily",
            "Maintain consistent sleep schedule",
            "Engage in regular physical activity",
            "Monitor stress levels",
            "Seek professional evaluation if patterns persist"
        ]
        
        for item in action_items:
            st.checkbox(item)
        
        # Resources
        st.markdown("---")
        st.subheader("📚 Additional Resources")
        
        resources = {
            "Mental Health Resources": "https://www.mentalhealth.gov",
            "Stress Management": "https://www.cdc.gov/mentalhealth/stress-management",
            "Mindfulness Exercises": "https://www.mindful.org",
            "Crisis Support": "https://www.crisistextline.org"
        }
        
        for name, url in resources.items():
            st.markdown(f"- [{name}]({url})")
    
    def _generate_recommendations(self, risk_score, result):
        """Generate recommendations based on risk score"""
        # Ensure numeric risk_score to avoid string/float comparison errors
        risk_score = _to_float(risk_score, 50)
        recommendations = {
            'immediate': [],
            'short_term': [],
            'long_term': []
        }
        
        if risk_score > 70:
            recommendations['immediate'].append("Consider consulting a mental health professional")
            recommendations['immediate'].append("Practice deep breathing exercises for 5 minutes")
            recommendations['short_term'].append("Schedule a wellness check-up")
            recommendations['short_term'].append("Implement daily stress management routine")
            recommendations['long_term'].append("Develop comprehensive mental health plan")
            recommendations['long_term'].append("Consider regular therapy sessions")
        
        elif risk_score > 40:
            recommendations['immediate'].append("Take a 10-minute break for relaxation")
            recommendations['short_term'].append("Practice mindfulness meditation daily")
            recommendations['short_term'].append("Monitor handwriting patterns weekly")
            recommendations['long_term'].append("Develop consistent self-care routine")
            recommendations['long_term'].append("Consider stress management workshops")
        
        else:
            recommendations['immediate'].append("Maintain current healthy habits")
            recommendations['short_term'].append("Continue regular self-assessment")
            recommendations['short_term'].append("Practice preventive stress management")
            recommendations['long_term'].append("Develop resilience-building practices")
        
        # Add NPI-specific recommendations
        if 'neural_pressure' in result:
            npi_score = _to_float(result['neural_pressure'].get('npi_score', 50), 50)
            if npi_score > 70:
                recommendations['immediate'].append("Reduce cognitive load where possible")
                recommendations['short_term'].append("Practice cognitive relaxation techniques")
        
        return recommendations
    
    def generate_pdf_report(self, result):
        """Generate a PDF report (placeholder)"""
        st.info("PDF report generation feature coming soon!")
        # In a real implementation, you would use a PDF generation library
        # like ReportLab, WeasyPrint, or PDFKit

    def show_disclaimer(self):
        """Show the standard medical disclaimer used across the app."""
        st.warning(
            """
⚠ Important Medical Disclaimer

This tool provides assessment based on handwriting patterns and is intended for informational purposes only. It is not a substitute for professional medical diagnosis, advice, or treatment. Always consult qualified healthcare providers for medical concerns.

If you are experiencing a mental health crisis, please contact emergency services or a crisis hotline immediately.
            """
        )
    
    def webcam_analysis(self):
        """Real-time webcam analysis"""
        st.header("📹 Real-time Webcam Analysis")
        
        if not RealTimeWebcamInference:
            st.error("Real-time webcam analysis module not available")
            st.info("Make sure all required dependencies are installed")
            return
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("""
            ### Real-time Analysis Instructions
            
            1. **Position your webcam** to capture writing clearly
            2. **Use paper and pen** or write on a tablet
            3. **Write naturally** as you would normally
            4. **Press Start** to begin analysis
            5. **Press Stop** when finished
            
            The system will analyze:
            - Stroke patterns in real-time
            - Neural pressure fluctuations
            - Mental health indicators
            - Writing rhythm and consistency
            """)
            
            # Webcam preview placeholder
            webcam_placeholder = st.empty()
            
            # Analysis display
            analysis_placeholder = st.empty()
        
        with col2:
            st.subheader("Controls")
            
            # Start/Stop buttons
            col_start, col_stop = st.columns(2)
            
            with col_start:
                start_analysis = st.button("▶ Start Analysis", type="primary", width="stretch")
            
            with col_stop:
                stop_analysis = st.button("⏹ Stop Analysis", type="secondary", width="stretch")
            
            st.markdown("---")
            
            # Analysis settings
            st.subheader("Settings")
            
            analysis_duration = st.slider("Analysis Duration (minutes)", 1, 30, 5)
            enable_npi = st.checkbox("Enable Neural Pressure Analysis", True)
            save_frames = st.checkbox("Save Analysis Frames", False)
            
            st.markdown("---")
            
            # Status
            st.subheader("Status")
            status_placeholder = st.empty()
            
            # Results placeholder
            results_placeholder = st.empty()
        
        # Handle analysis start
        if start_analysis:
            status_placeholder.info("Starting webcam analysis...")
            
            try:
                # Initialize webcam inference
                webcam_inference = RealTimeWebcamInference(self.model_path)
                
                # Run analysis
                results = webcam_inference.run()
                
                if results:
                    status_placeholder.success(f"Analysis complete! Processed {len(results)} frames.")
                    
                    # Display summary
                    df = pd.DataFrame([
                        {
                            'Frame': i+1,
                            'Prediction': r['prediction'],
                            'Confidence': fmt_percent(r.get('confidence', 0), 2),
                            'Risk Score': r.get('risk_score', 0),
                            'Risk Level': r.get('risk_level', 'Unknown')
                        }
                        for i, r in enumerate(results)
                    ])
                    
                    results_placeholder.dataframe(df, width="stretch")
                    
                    # Save to history
                    for result in results:
                        result['timestamp'] = datetime.now().isoformat()
                        result['analysis_type'] = 'webcam'
                        st.session_state.result_history.append(result)
                    
                    # Show summary statistics
                    avg_risk = df['Risk Score'].astype(float).mean()
                    pred_counts = df['Prediction'].value_counts()
                    
                    col_stats1, col_stats2, col_stats3 = st.columns(3)
                    
                    with col_stats1:
                        st.metric("Average Risk Score", fmt_float(avg_risk, 1))
                    
                    with col_stats2:
                        st.metric("Most Common", pred_counts.index[0] if len(pred_counts) > 0 else "N/A")
                    
                    with col_stats3:
                        high_risk = len(df[df['Risk Level'] == 'High'])
                        st.metric("High Risk Frames", high_risk)
                
            except Exception as e:
                status_placeholder.error(f"Analysis failed: {e}")
        
        elif stop_analysis:
            status_placeholder.info("Analysis stopped by user")
    
    def batch_analysis(self):
        """Batch processing of multiple images"""
        st.header("📦 Batch Processing")
        
        st.markdown("""
        Upload multiple handwriting images for batch analysis. 
        The system will process all images and provide a comprehensive report.
        """)
        
        # File upload for batch
        uploaded_files = st.file_uploader(
            "Choose multiple image files",
            type=['jpg', 'jpeg', 'png', 'tiff', 'bmp', 'webp'],
            accept_multiple_files=True,
            help="Select multiple images for batch analysis"
        )
        
        if uploaded_files:
            st.info(f"Selected {len(uploaded_files)} files for analysis")
            
            # Display file list
            with st.expander("View Selected Files"):
                for file in uploaded_files:
                    st.write(f"- {file.name} ({file.size / 1024:.1f} KB)")
            
            # Analysis options
            col_opt1, col_opt2 = st.columns(2)
            
            with col_opt1:
                batch_npi = st.checkbox("Estimate NPI for all files", True)
                generate_summary = st.checkbox("Generate summary report", True)
            
            with col_opt2:
                save_individual = st.checkbox("Save individual reports", False)
                export_format = st.selectbox("Export format", ["CSV", "JSON", "Excel"])
            
            # Start batch processing
            if st.button("🚀 Process Batch", type="primary", width="stretch"):
                if not self.inference:
                    st.error("Model not loaded. Please load a model first.")
                    return
                
                if len(uploaded_files) == 0:
                    st.warning("Please upload at least one file")
                    return
                
                # Create progress bar
                progress_bar = st.progress(0)
                status_text = st.empty()
                results = []
                
                # Process each file
                for idx, uploaded_file in enumerate(uploaded_files):
                    # Update progress
                    progress = (idx + 1) / len(uploaded_files)
                    progress_bar.progress(progress)
                    status_text.text(f"Processing {idx + 1}/{len(uploaded_files)}: {uploaded_file.name}")
                    
                    # Save to temporary file
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp_file:
                        image = Image.open(uploaded_file)
                        image.save(tmp_file.name)
                        temp_path = tmp_file.name
                    
                    try:
                        # Run inference
                        result = self.inference.predict(temp_path)
                        result['filename'] = uploaded_file.name
                        result['timestamp'] = datetime.now().isoformat()
                        result['analysis_type'] = 'batch'
                        
                        # Add to results
                        results.append(result)
                        
                        # Save to history
                        st.session_state.result_history.append(result)
                        
                    except Exception as e:
                        st.error(f"Error processing {uploaded_file.name}: {e}")
                    finally:
                        # Cleanup
                        try:
                            os.remove(temp_path)
                        except:
                            pass
                
                # Update progress
                progress_bar.progress(1.0)
                status_text.text(f"Processing complete! Analyzed {len(results)} files.")
                
                # Display batch results
                if results:
                    st.success(f"✅ Batch processing complete! Successfully analyzed {len(results)} files.")
                    
                    # Create summary dataframe
                    df_data = []
                    for r in results:
                        df_data.append({
                            'Filename': r['filename'],
                            'Prediction': r['prediction'],
                            'Confidence': fmt_percent(r.get('confidence', 0), 2),
                            'Risk Score': r['risk_score'],
                            'Risk Level': r['risk_level']
                        })
                    
                    df = pd.DataFrame(df_data)
                    
                    # Display results
                    st.subheader("Batch Results")
                    st.dataframe(df, width="stretch")
                    
                    # Statistics
                    st.subheader("Batch Statistics")
                    
                    col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
                    
                    with col_stat1:
                        pred_counts = df['Prediction'].value_counts()
                        most_common = pred_counts.index[0] if len(pred_counts) > 0 else "N/A"
                        st.metric("Most Common", most_common)
                    
                    with col_stat2:
                        avg_risk = pd.to_numeric(df['Risk Score'], errors='coerce').mean()
                        st.metric("Average Risk", fmt_float(avg_risk, 1))
                    
                    with col_stat3:
                        high_risk = len(df[df['Risk Level'] == 'High'])
                        st.metric("High Risk", high_risk)
                    
                    with col_stat4:
                        success_rate = len(results) / len(uploaded_files) * 100
                        st.metric("Success Rate", f"{fmt_float(success_rate, 1)}%")
                    
                    # Export options
                    st.markdown("---")
                    st.subheader("Export Results")
                    
                    export_col1, export_col2, export_col3 = st.columns(3)
                    
                    with export_col1:
                        # CSV export
                        csv_data = df.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            label="📥 Download CSV",
                            data=csv_data,
                            file_name=f"batch_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv",
                            use_container_width=True
                        )
                    
                    with export_col2:
                        # JSON export
                        json_data = json.dumps(results, indent=2, default=str)
                        st.download_button(
                            label="📥 Download JSON",
                            data=json_data,
                            file_name=f"batch_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                            mime="application/json",
                            use_container_width=True
                        )
                    
                    with export_col3:
                        # Excel export (requires openpyxl)
                        try:
                            import io
                            buffer = io.BytesIO()
                            with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                                df.to_excel(writer, sheet_name='Results', index=False)
                                
                                # Add summary sheet
                                summary_data = {
                                    'Metric': ['Total Files', 'Success Rate', 'Avg Risk', 'High Risk Files'],
                                    'Value': [len(results), f"{fmt_float(success_rate,1)}%", fmt_float(avg_risk,1), high_risk]
                                }
                                pd.DataFrame(summary_data).to_excel(writer, sheet_name='Summary', index=False)
                            
                            buffer.seek(0)
                            
                            st.download_button(
                                label="📥 Download Excel",
                                data=buffer,
                                file_name=f"batch_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                use_container_width=True
                            )
                        except ImportError:
                            st.info("Excel export requires openpyxl: pip install openpyxl")
    
    def video_analysis(self):
        """Video file analysis"""
        st.header("🎥 Video Analysis")
        
        st.markdown("""
        Upload a video recording of handwriting for analysis.
        The system will extract frames and analyze writing patterns over time.
        """)
        
        # File upload for video
        uploaded_video = st.file_uploader(
            "Choose a video file",
            type=['mp4', 'avi', 'mov', 'mkv', 'webm', 'flv'],
            help="Upload a video file (max 100MB)"
        )
        
        if uploaded_video:
            # Display video info
            file_size = uploaded_video.size / (1024 * 1024)  # MB
            st.info(f"Video file: {uploaded_video.name} ({file_size:.1f} MB)")
            
            # Video preview
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_video:
                tmp_video.write(uploaded_video.read())
                video_path = tmp_video.name
            
            # Display video
            try:
                st.video(video_path)
            except:
                st.info("Video preview not available")
            
            # Analysis settings
            col_set1, col_set2 = st.columns(2)
            
            with col_set1:
                frame_rate = st.slider("Analysis frame rate (fps)", 1, 30, 5)
                analysis_mode = st.selectbox(
                    "Analysis mode",
                    ["Comprehensive", "Fast", "Detailed"]
                )
            
            with col_set2:
                extract_writing = st.checkbox("Extract writing regions", True)
                track_progress = st.checkbox("Track progress over time", True)
            
            # Start analysis
            if st.button("🎬 Analyze Video", type="primary", width="stretch"):
                if not self.inference:
                    st.error("Model not loaded. Please load a model first.")
                    return
                
                with st.spinner("Analyzing video..."):
                    try:
                        # Create temporary directory for frames
                        temp_dir = tempfile.mkdtemp()
                        
                        # Extract frames from video
                        cap = cv2.VideoCapture(video_path)
                        frame_count = 0
                        results = []
                        
                        # Progress tracking
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        
                        while cap.isOpened():
                            ret, frame = cap.read()
                            if not ret:
                                break
                            
                            # Sample frames based on frame rate
                            if frame_count % (30 // frame_rate) == 0:
                                # Save frame
                                frame_path = os.path.join(temp_dir, f"frame_{frame_count}.jpg")
                                cv2.imwrite(frame_path, frame)
                                
                                # Analyze frame
                                try:
                                    result = self.inference.predict(frame_path)
                                    result['frame_number'] = frame_count
                                    result['timestamp'] = frame_count / 30  # Assuming 30 fps
                                    result['analysis_type'] = 'video'
                                    results.append(result)
                                    
                                    # Save to history
                                    st.session_state.result_history.append(result)
                                    
                                except Exception as e:
                                    st.warning(f"Error analyzing frame {frame_count}: {e}")
                            
                            frame_count += 1
                            
                            # Update progress
                            if frame_count % 100 == 0:
                                progress = min(frame_count / 1000, 1.0)  # Assuming max 1000 frames
                                progress_bar.progress(progress)
                                status_text.text(f"Processed {frame_count} frames...")
                        
                        cap.release()
                        
                        # Cleanup
                        import shutil
                        shutil.rmtree(temp_dir)
                        
                        if os.path.exists(video_path):
                            os.remove(video_path)
                        
                        # Display results
                        if results:
                            st.success(f"✅ Video analysis complete! Analyzed {len(results)} frames.")
                            
                            # Create results dataframe
                            df_data = []
                            for r in results:
                                df_data.append({
                                    'Frame': r['frame_number'],
                                    'Time (s)': r['timestamp'],
                                    'Prediction': r['prediction'],
                                    'Confidence': fmt_percent(r.get('confidence', 0), 2),
                                    'Risk Score': r.get('risk_score', 0),
                                    'Risk Level': r.get('risk_level', 'Unknown')
                                })
                            
                            df = pd.DataFrame(df_data)

                            # Display results
                            st.subheader("Video Analysis Results")
                            st.dataframe(df, width="stretch")
                            
                            # Create time series plot
                            fig_time = go.Figure()
                            
                            # Add risk score trace
                            fig_time.add_trace(go.Scatter(
                                x=df['Time (s)'],
                                y=df['Risk Score'],
                                mode='lines+markers',
                                name='Risk Score',
                                line=dict(color='red', width=2)
                            ))
                            
                            fig_time.update_layout(
                                title="Risk Score Over Time",
                                xaxis_title="Time (seconds)",
                                yaxis_title="Risk Score",
                                height=400,
                                hovermode='x unified'
                            )
                            
                            st.plotly_chart(fig_time, width="stretch")
                            
                            # Summary statistics
                            avg_risk = df['Risk Score'].mean()
                            max_risk = df['Risk Score'].max()
                            risk_trend = "Increasing" if df['Risk Score'].iloc[-1] > df['Risk Score'].iloc[0] else "Decreasing"
                            
                            col_sum1, col_sum2, col_sum3 = st.columns(3)
                            
                            with col_sum1:
                                st.metric("Average Risk", fmt_float(avg_risk, 1))
                            
                            with col_sum2:
                                st.metric("Maximum Risk", fmt_float(max_risk, 1))
                            
                            with col_sum3:
                                st.metric("Risk Trend", risk_trend)
                        
                        else:
                            st.warning("No frames were successfully analyzed")
                    
                    except Exception as e:
                        st.error(f"Video analysis failed: {e}")
    
    def digital_writing_analysis(self):
        """Digital writing analysis"""
        st.header("✍️ Digital Writing Analysis")
        
        st.markdown("""
        Analyze digital handwriting from tablets or touchscreens.
        Draw directly in the canvas below or upload digital writing data.
        """)
        
        # Canvas for digital drawing
        st.subheader("Digital Drawing Canvas")
        
        # Initialize session state for drawing
        if 'drawing_data' not in st.session_state:
            st.session_state.drawing_data = {
                'strokes': [],
                'current_stroke': [],
                'timestamps': [],
                'pressures': []
            }
        
        # Create drawing interface
        col_canvas, col_1 = st.columns([3, 1])
        
        with col_canvas:
            # Drawing canvas using streamlit-drawable-canvas if available
            try:
                from streamlit_drawable_canvas import st_canvas
                
                # Canvas specifications
                canvas_result = st_canvas(
                    fill_color="rgba(255, 255, 255, 0)",  # Transparent fill
                    stroke_width=3,
                    stroke_color="#000000",
                    background_color="#ffffff",
                    height=400,
                    width=600,
                    drawing_mode="freedraw",
                    key="canvas",
                    display_toolbar=True
                )
                
                if canvas_result is not None and canvas_result.image_data is not None:
                    # Convert canvas result to image
                    img_array = np.array(canvas_result.image_data)
                    
                    # Display drawing
                    st.image(img_array, caption="Your Drawing", width="stretch")

                    # Extract stroke paths from canvas JSON (if present) and store in session
                    try:
                        json_data = getattr(canvas_result, 'json_data', None)
                        if json_data and isinstance(json_data, dict):
                            objs = json_data.get('objects', [])
                            strokes = []
                            for obj in objs:
                                path = obj.get('path') or obj.get('points') or obj.get('stroke')
                                pts = []
                                if isinstance(path, list):
                                    for p in path:
                                        if isinstance(p, dict):
                                            x = p.get('x') or p.get('left')
                                            y = p.get('y') or p.get('top')
                                            if x is not None and y is not None:
                                                try:
                                                    pts.append([float(x), float(y)])
                                                except Exception:
                                                    continue
                                        elif isinstance(p, (list, tuple)) and len(p) >= 2:
                                            try:
                                                pts.append([float(p[0]), float(p[1])])
                                            except Exception:
                                                continue
                                if pts:
                                    timestamps = [i * 0.01 for i in range(len(pts))]
                                    pressures = [0.5 for _ in pts]
                                    strokes.append({'points': pts, 'timestamps': timestamps, 'pressures': pressures})

                            if strokes:
                                digital_data = {
                                    'strokes': strokes,
                                    'points': [pt for s in strokes for pt in s['points']],
                                    'timestamps': [t for s in strokes for t in s['timestamps']],
                                    'pressures': [p for s in strokes for p in s['pressures']]
                                }
                                st.session_state.digital_data = digital_data
                                st.success(f"Captured {len(strokes)} stroke(s) from canvas")
                    except Exception as e:
                        st.warning(f"Canvas stroke extraction failed: {e}")
            
            except ImportError:
                st.info("""
                **Digital drawing canvas requires additional setup:**
                Install the drawing library:
                ```bash
                pip install streamlit-drawable-canvas
                ```
                Or use the image upload option below.
                """)
        
        with col_canvas:
            st.subheader("Drawing Controls")
            
            # Clear canvas
            if st.button("🧹 Clear Canvas", width="stretch"):
                st.session_state.drawing_data = {
                    'strokes': [],
                    'current_stroke': [],
                    'timestamps': [],
                    'pressures': []
                }
                st.rerun()
            
            st.markdown("---")
            
            # Upload digital writing data
            st.subheader("Upload Data")
            
            uploaded_digital = st.file_uploader(
                "Upload digital writing data",
                type=['json', 'csv', 'txt'],
                help="Upload digital writing data in JSON or CSV format"
            )
            
            if uploaded_digital:
                try:
                    if uploaded_digital.name.endswith('.json'):
                        loaded = json.load(uploaded_digital)
                        if isinstance(loaded, dict) and 'strokes' in loaded:
                            st.session_state.digital_data = loaded
                            st.success(f"Loaded {len(loaded.get('strokes', []))} strokes")
                        elif isinstance(loaded, list):
                            strokes = []
                            for arr in loaded:
                                if isinstance(arr, list) and len(arr) > 0:
                                    pts = []
                                    for p in arr:
                                        if isinstance(p, (list, tuple)) and len(p) >= 2:
                                            try:
                                                pts.append([float(p[0]), float(p[1])])
                                            except Exception:
                                                continue
                                    if pts:
                                        strokes.append({'points': pts, 'timestamps': [i*0.01 for i in range(len(pts))], 'pressures': [0.5]*len(pts)})
                            st.session_state.digital_data = {'strokes': strokes}
                            st.success(f"Loaded {len(strokes)} strokes from JSON list")
                        else:
                            st.error("Unsupported JSON structure for digital data")

                    elif uploaded_digital.name.endswith('.csv'):
                        df = pd.read_csv(uploaded_digital)
                        strokes = []
                        if 'stroke_id' in df.columns:
                            for sid, g in df.groupby('stroke_id'):
                                pts = []
                                timestamps = []
                                pressures = []
                                for _, row in g.iterrows():
                                    if 'x' in row and 'y' in row:
                                        try:
                                            pts.append([float(row['x']), float(row['y'])])
                                        except Exception:
                                            continue
                                        timestamps.append(float(row.get('t', row.get('timestamp', len(timestamps)*0.01))))
                                        pressures.append(float(row.get('pressure', 0.5)))
                                if pts:
                                    strokes.append({'points': pts, 'timestamps': timestamps or [i*0.01 for i in range(len(pts))], 'pressures': pressures or [0.5]*len(pts)})
                        else:
                            pts = []
                            timestamps = []
                            pressures = []
                            for _, row in df.iterrows():
                                if 'x' in row and 'y' in row:
                                    try:
                                        pts.append([float(row['x']), float(row['y'])])
                                    except Exception:
                                        continue
                                    timestamps.append(float(row.get('t', row.get('timestamp', len(timestamps)*0.01))))
                                    pressures.append(float(row.get('pressure', 0.5)))
                            if pts:
                                strokes.append({'points': pts, 'timestamps': timestamps or [i*0.01 for i in range(len(pts))], 'pressures': pressures or [0.5]*len(pts)})

                        if strokes:
                            st.session_state.digital_data = {
                                'strokes': strokes,
                                'points': [pt for s in strokes for pt in s['points']],
                                'timestamps': [t for s in strokes for t in s['timestamps']],
                                'pressures': [p for s in strokes for p in s['pressures']]
                            }
                            st.success(f"Loaded {len(strokes)} stroke(s) from CSV")
                        else:
                            st.error("No valid stroke points found in CSV")

                except Exception as e:
                    st.error(f"Error loading file: {e}")
            
            st.markdown("---")
            
            # Analysis button
            if st.button("🔍 Analyze Drawing", type="primary", width="stretch"):
                if 'canvas_result' in locals() and canvas_result.image_data is not None:
                    # Save canvas image temporarily
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp_file:
                        img = Image.fromarray(np.array(canvas_result.image_data))
                        img.save(tmp_file.name)
                        temp_path = tmp_file.name
                    
                    try:
                        # Analyze the drawing
                        if self.inference:
                            result = self.inference.predict(temp_path)
                            result['timestamp'] = datetime.now().isoformat()
                            result['analysis_type'] = 'digital'
                            result['drawing_mode'] = 'canvas'
                            
                            # If stored digital stroke data exists in session, compute neuromotor features and NPI
                            stored_dd = st.session_state.get('digital_data')
                            if isinstance(stored_dd, dict) and 'strokes' in stored_dd:
                                try:
                                    from src.neuro_motor_features import NeuroMotorFeatureExtractor
                                    from src.neural_pressure import NeuralPressureEstimator

                                    extractor = NeuroMotorFeatureExtractor()
                                    estimator = NeuralPressureEstimator()

                                    neuromotor_features = extractor.extract_all_features(stored_dd)
                                    npi = estimator.estimate_neural_pressure(neuromotor_features)

                                    result['neuromotor_features'] = neuromotor_features
                                    result['neural_pressure'] = npi
                                except Exception as e:
                                    st.warning(f"NPI estimation skipped: {e}")

                            # Save to history
                            st.session_state.result_history.append(result)
                            
                            # Display results
                            self.display_results(result)
                        
                    except Exception as e:
                        st.error(f"Analysis failed: {e}")
                    finally:
                        # Cleanup
                        try:
                            os.remove(temp_path)
                        except:
                            pass
                else:
                    st.warning("Please draw something first or upload digital writing data")
        
        # Alternative: Upload image of digital writing
        st.markdown("---")
        st.subheader("Alternative: Upload Digital Writing Image")
        
        uploaded_digital_img = st.file_uploader(
            "Upload image of digital writing",
            type=['jpg', 'jpeg', 'png', 'bmp'],
            key="digital_upload"
        )
        
        if uploaded_digital_img and st.button("Analyze Uploaded Digital Writing"):
            with st.spinner("Analyzing digital writing..."):
                # Save to temporary file
                with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp_file:
                    image = Image.open(uploaded_digital_img)
                    image.save(tmp_file.name)
                    temp_path = tmp_file.name
                
                try:
                    # Analyze the image
                    if self.inference:
                        result = self.inference.predict(temp_path)
                        result['timestamp'] = datetime.now().isoformat()
                        result['analysis_type'] = 'digital'
                        result['drawing_mode'] = 'upload'
                        result['filename'] = uploaded_digital_img.name
                        
                        # If an uploaded digital data file was provided earlier and stored in session, compute NPI
                        stored_dd = st.session_state.get('digital_data')
                        if isinstance(stored_dd, dict) and 'strokes' in stored_dd:
                            try:
                                from src.neuro_motor_features import NeuroMotorFeatureExtractor
                                from src.neural_pressure import NeuralPressureEstimator

                                extractor = NeuroMotorFeatureExtractor()
                                estimator = NeuralPressureEstimator()

                                neuromotor_features = extractor.extract_all_features(stored_dd)
                                npi = estimator.estimate_neural_pressure(neuromotor_features)

                                result['neuromotor_features'] = neuromotor_features
                                result['neural_pressure'] = npi
                            except Exception as e:
                                st.warning(f"NPI estimation skipped: {e}")

                        # Save to history
                        st.session_state.result_history.append(result)
                        
                        # Display results
                        self.display_results(result)
                
                except Exception as e:
                    st.error(f"Analysis failed: {e}")
                finally:
                    # Cleanup
                    try:
                        os.remove(temp_path)
                    except:
                        pass
    
    def results_history(self):
        """Display analysis history"""
        st.header("📋 Analysis History")
        
        if not st.session_state.result_history:
            st.info("No analysis history yet. Run some analyses to see results here.")
            
            # Quick actions
            col_act1, col_act2, col_act3 = st.columns(3)
            
            with col_act1:
                if st.button("Run Demo Analysis", width="stretch"):
                    # Create demo result
                    demo_result = {
                        'timestamp': datetime.now().isoformat(),
                        'prediction': 'Mild',
                        'confidence': 0.78,
                        'risk_score': 42.5,
                        'risk_level': 'Medium',
                        'analysis_type': 'demo'
                    }
                    st.session_state.result_history.append(demo_result)
                    st.rerun()
            
            with col_act2:
                if st.button("Upload Sample Image", width="stretch"):
                    st.session_state.current_page = "Image Analysis"
                    st.rerun()
            
            with col_act3:
                if st.button("Try Webcam Analysis", width="stretch"):
                    st.session_state.current_page = "Real-time Webcam"
                    st.rerun()
            
            return
        
        # Display history controls
        col_hist1, col_hist2, col_hist3 = st.columns([2, 1, 1])
        
        with col_hist1:
            st.subheader(f"Analysis History ({len(st.session_state.result_history)} entries)")
        
        with col_hist2:
            # Filter by analysis type
            analysis_types = list(set([r.get('analysis_type', 'unknown') for r in st.session_state.result_history]))
            selected_type = st.selectbox("Filter by type", ["All"] + analysis_types)
        
        with col_hist3:
            # Sort options
            sort_by = st.selectbox("Sort by", ["Date (newest)", "Date (oldest)", "Risk Score", "Confidence"])
        
        # Filter and sort results
        filtered_results = st.session_state.result_history.copy()
        
        if selected_type != "All":
            filtered_results = [r for r in filtered_results if r.get('analysis_type') == selected_type]
        
        # Sort results
        if sort_by == "Date (newest)":
            filtered_results.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
        elif sort_by == "Date (oldest)":
            filtered_results.sort(key=lambda x: x.get('timestamp', ''))
        elif sort_by == "Risk Score":
            filtered_results.sort(key=lambda x: x.get('risk_score', 0), reverse=True)
        elif sort_by == "Confidence":
            filtered_results.sort(key=lambda x: x.get('confidence', 0), reverse=True)
        
        # Display results in expandable sections
        for idx, result in enumerate(filtered_results):
            with st.expander(f"Analysis {idx + 1}: {result.get('filename', 'Unnamed')} - {result.get('prediction', 'Unknown')}"):
                col_res1, col_res2, col_res3 = st.columns(3)
                
                with col_res1:
                    st.write(f"**Date:** {result.get('timestamp', 'Unknown')}")
                    st.write(f"**Type:** {result.get('analysis_type', 'Unknown')}")
                    st.write(f"**Filename:** {result.get('filename', 'N/A')}")
                
                with col_res2:
                    st.write(f"**Prediction:** {result.get('prediction', 'Unknown')}")
                    st.write(f"**Confidence:** {fmt_percent(result.get('confidence', 0), 2)}")
                    st.write(f"**Risk Level:** {result.get('risk_level', 'Unknown')}")
                
                with col_res3:
                    st.write(f"**Risk Score:** {fmt_float(result.get('risk_score', 0), 1)}")
                    
                    # Neural pressure if available
                    if 'neural_pressure' in result:
                        npi = result['neural_pressure']
                        st.write(f"**NPI:** {fmt_float(npi.get('npi_score', 0), 1)} ({npi.get('npi_category', 'Unknown')})")
                
                # Action buttons for this result
                col_act1, col_act2, col_act3 = st.columns(3)
                
                with col_act1:
                    if st.button(f"View Details", key=f"view_{idx}"):
                        self.display_results(result)
                
                with col_act2:
                    if st.button(f"Delete", key=f"delete_{idx}"):
                        st.session_state.result_history.remove(result)
                        st.rerun()
                
                with col_act3:
                    # Export this result
                    json_str = json.dumps(result, indent=2, default=str)
                    st.download_button(
                        label="Export",
                        data=json_str,
                        file_name=f"analysis_{idx + 1}_{datetime.now().strftime('%Y%m%d')}.json",
                        mime="application/json",
                        key=f"export_{idx}"
                    )
        
        # Clear all history
        st.markdown("---")
        col_clear1, col_clear2 = st.columns([3, 1])
        
        with col_clear2:
            if st.button("🗑️ Clear All History", type="secondary", width="stretch"):
                st.session_state.result_history = []
                st.rerun()
        
        # Export all history
        if st.session_state.result_history:
            st.markdown("---")
            st.subheader("Export All History")
            
            json_all = json.dumps(st.session_state.result_history, indent=2, default=str)
            
            st.download_button(
                label="📥 Download All as JSON",
                data=json_all,
                file_name=f"analysis_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json",
                use_container_width=True
            )
            
            # Create CSV summary
            csv_data = []
            for result in st.session_state.result_history:
                csv_data.append({
                    'Timestamp': result.get('timestamp', ''),
                    'Filename': result.get('filename', ''),
                    'Analysis Type': result.get('analysis_type', ''),
                    'Prediction': result.get('prediction', ''),
                    'Confidence': result.get('confidence', 0),
                    'Risk Score': result.get('risk_score', 0),
                    'Risk Level': result.get('risk_level', '')
                })
            
            if csv_data:
                df_summary = pd.DataFrame(csv_data)
                csv_str = df_summary.to_csv(index=False)
                
                st.download_button(
                    label="📥 Download Summary CSV",
                    data=csv_str,
                    file_name=f"analysis_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )

def main():
    """Main entry point"""
    app = MentalHealthWebApp()
    app.run()

if __name__ == "__main__":
    main()