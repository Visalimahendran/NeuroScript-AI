import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import cv2
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import torch
import networkx as nx
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

class AdvancedVisualization:
    """Advanced visualization techniques for mental health assessment"""
    
    def __init__(self, save_dir='advanced_visualizations'):
        import os
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
        # Color schemes
        self.risk_colors = {
            'Normal': '#2E8B57',  # SeaGreen
            'Mild': '#FFA500',    # Orange
            'Severe': '#DC143C'   # Crimson
        }
        
        self.feature_colors = px.colors.qualitative.Set3
    
    def create_comprehensive_dashboard(self, analysis_results):
        """Create comprehensive interactive dashboard"""
        fig = make_subplots(
            rows=3, cols=3,
            subplot_titles=(
                'Risk Assessment Gauge',
                'Feature Importance Radar',
                'Temporal Pattern Analysis',
                'Spatial Distribution',
                'Confusion Matrix',
                'ROC Curves',
                'Feature Correlations',
                'Cluster Analysis',
                'Trend Analysis'
            ),
            specs=[
                [{'type': 'indicator'}, {'type': 'scatterpolar'}, {'type': 'scatter'}],
                [{'type': 'heatmap'}, {'type': 'heatmap'}, {'type': 'scatter'}],
                [{'type': 'heatmap'}, {'type': 'scatter'}, {'type': 'scatter'}]
            ],
            vertical_spacing=0.08,
            horizontal_spacing=0.08
        )
        
        # 1. Risk Assessment Gauge
        if 'risk_score' in analysis_results:
            fig.add_trace(
                go.Indicator(
                    mode="gauge+number+delta",
                    value=analysis_results['risk_score'],
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
                            'value': analysis_results['risk_score']
                        }
                    }
                ),
                row=1, col=1
            )
        
        # 2. Feature Importance Radar
        if 'feature_importance' in analysis_results:
            features = list(analysis_results['feature_importance'].keys())[:8]
            values = list(analysis_results['feature_importance'].values())[:8]
            
            fig.add_trace(
                go.Scatterpolar(
                    r=values,
                    theta=features,
                    fill='toself',
                    line_color='blue',
                    name='Feature Importance'
                ),
                row=1, col=2
            )
        
        # 3. Temporal Pattern Analysis
        if 'temporal_patterns' in analysis_results:
            temporal_data = analysis_results['temporal_patterns']
            fig.add_trace(
                go.Scatter(
                    x=temporal_data.get('timestamps', []),
                    y=temporal_data.get('values', []),
                    mode='lines+markers',
                    name='Temporal Pattern'
                ),
                row=1, col=3
            )
        
        # 4. Spatial Distribution Heatmap
        if 'spatial_distribution' in analysis_results:
            spatial_data = analysis_results['spatial_distribution']
            fig.add_trace(
                go.Heatmap(
                    z=spatial_data,
                    colorscale='Viridis',
                    name='Spatial Distribution'
                ),
                row=2, col=1
            )
        
        # 5. Confusion Matrix
        if 'confusion_matrix' in analysis_results:
            cm = analysis_results['confusion_matrix']
            fig.add_trace(
                go.Heatmap(
                    z=cm,
                    x=['Normal', 'Mild', 'Severe'],
                    y=['Normal', 'Mild', 'Severe'],
                    colorscale='Blues',
                    text=cm,
                    texttemplate='%{text}',
                    name='Confusion Matrix'
                ),
                row=2, col=2
            )
        
        # 6. ROC Curves
        if 'roc_curves' in analysis_results:
            roc_data = analysis_results['roc_curves']
            for i, (fpr, tpr) in enumerate(roc_data):
                fig.add_trace(
                    go.Scatter(
                        x=fpr, y=tpr,
                        mode='lines',
                        name=f'Class {i}',
                        showlegend=False
                    ),
                    row=2, col=3
                )
        
        # 7. Feature Correlations
        if 'feature_correlations' in analysis_results:
            corr_matrix = analysis_results['feature_correlations']
            fig.add_trace(
                go.Heatmap(
                    z=corr_matrix,
                    colorscale='RdBu',
                    zmid=0,
                    name='Feature Correlations'
                ),
                row=3, col=1
            )
        
        # 8. Cluster Analysis
        if 'cluster_data' in analysis_results:
            cluster_data = analysis_results['cluster_data']
            fig.add_trace(
                go.Scatter(
                    x=cluster_data['x'],
                    y=cluster_data['y'],
                    mode='markers',
                    marker=dict(
                        color=cluster_data['labels'],
                        colorscale='Viridis'
                    ),
                    name='Clusters'
                ),
                row=3, col=2
            )
        
        # 9. Trend Analysis
        if 'trend_data' in analysis_results:
            trend_data = analysis_results['trend_data']
            fig.add_trace(
                go.Scatter(
                    x=trend_data['x'],
                    y=trend_data['y'],
                    mode='lines+markers',
                    line=dict(width=3),
                    name='Trend'
                ),
                row=3, col=3
            )
        
        fig.update_layout(
            height=1200,
            width=1400,
            title_text="Comprehensive Mental Health Assessment Dashboard",
            showlegend=True,
            template="plotly_white"
        )
        
        # Save dashboard
        fig.write_html(f"{self.save_dir}/comprehensive_dashboard.html")
        
        return fig
    
    def visualize_handwriting_analysis(self, image, features, predictions):
        """Create detailed handwriting analysis visualization"""
        fig = make_subplots(
            rows=2, cols=3,
            subplot_titles=(
                'Original Handwriting',
                'Feature Extraction',
                'Stroke Analysis',
                'Spatial Distribution',
                'Frequency Analysis',
                'Prediction Results'
            ),
            specs=[
                [{'type': 'image'}, {'type': 'heatmap'}, {'type': 'scatter'}],
                [{'type': 'scatter'}, {'type': 'bar'}, {'type': 'indicator'}]
            ]
        )
        
        # 1. Original Image
        fig.add_trace(
            go.Image(z=image, colormodel='gray'),
            row=1, col=1
        )
        
        # 2. Feature Heatmap
        if 'feature_map' in features:
            fig.add_trace(
                go.Heatmap(
                    z=features['feature_map'],
                    colorscale='Viridis',
                    showscale=False
                ),
                row=1, col=2
            )
        
        # 3. Stroke Analysis
        if 'stroke_data' in features:
            stroke_data = features['stroke_data']
            fig.add_trace(
                go.Scatter(
                    x=stroke_data['x'],
                    y=stroke_data['y'],
                    mode='lines+markers',
                    line=dict(width=2),
                    marker=dict(size=6)
                ),
                row=1, col=3
            )
        
        # 4. Spatial Distribution
        if 'spatial_features' in features:
            spatial = features['spatial_features']
            fig.add_trace(
                go.Scatter(
                    x=spatial['x_coords'],
                    y=spatial['y_coords'],
                    mode='markers',
                    marker=dict(
                        size=spatial['sizes'],
                        color=spatial['intensities'],
                        colorscale='Viridis',
                        showscale=True
                    )
                ),
                row=2, col=1
            )
        
        # 5. Feature Importance Bar Chart
        if 'feature_importance' in features:
            imp_features = features['feature_importance']
            fig.add_trace(
                go.Bar(
                    x=list(imp_features.keys())[:10],
                    y=list(imp_features.values())[:10],
                    marker_color=self.feature_colors
                ),
                row=2, col=2
            )
        
        # 6. Prediction Indicator
        if 'prediction' in predictions:
            pred = predictions['prediction']
            confidence = predictions.get('confidence', 0.5)
            
            color_map = {
                'Normal': 'green',
                'Mild': 'orange',
                'Severe': 'red'
            }
            
            fig.add_trace(
                go.Indicator(
                    mode="gauge+number",
                    value=confidence * 100,
                    title={'text': f"Prediction: {pred}"},
                    gauge={
                        'axis': {'range': [0, 100]},
                        'bar': {'color': color_map.get(pred, 'gray')},
                        'steps': [
                            {'range': [0, 30], 'color': "lightgray"},
                            {'range': [30, 70], 'color': "gray"},
                            {'range': [70, 100], 'color': "darkgray"}
                        ]
                    }
                ),
                row=2, col=3
            )
        
        fig.update_layout(
            height=800,
            width=1200,
            title_text="Handwriting Analysis Report",
            showlegend=False,
            template="plotly_white"
        )
        
        return fig
    
    def plot_feature_evolution(self, feature_history, time_points=None):
        """Plot evolution of features over time"""
        if time_points is None:
            time_points = range(len(feature_history))
        
        fig = go.Figure()
        
        # Get feature names from first entry
        if len(feature_history) > 0:
            feature_names = list(feature_history[0].keys())
            
            for feature in feature_names[:6]:  # Plot top 6 features
                values = [entry.get(feature, 0) for entry in feature_history]
                
                fig.add_trace(go.Scatter(
                    x=time_points,
                    y=values,
                    mode='lines+markers',
                    name=feature,
                    line=dict(width=2)
                ))
        
        fig.update_layout(
            title="Feature Evolution Over Time",
            xaxis_title="Time Point",
            yaxis_title="Feature Value",
            hovermode='x unified',
            height=500,
            template="plotly_white"
        )
        
        return fig
    
    def create_3d_feature_visualization(self, features, labels):
        """Create 3D visualization of feature space"""
        # Reduce to 3D using PCA
        from sklearn.decomposition import PCA
        
        if len(features.shape) > 2:
            features = features.reshape(features.shape[0], -1)
        
        pca = PCA(n_components=3)
        features_3d = pca.fit_transform(features)
        
        fig = go.Figure()
        
        # Create scatter plot
        scatter = go.Scatter3d(
            x=features_3d[:, 0],
            y=features_3d[:, 1],
            z=features_3d[:, 2],
            mode='markers',
            marker=dict(
                size=8,
                color=labels,
                colorscale='Viridis',
                opacity=0.8
            ),
            text=[f"Sample {i}" for i in range(len(labels))]
        )
        
        fig.add_trace(scatter)
        
        fig.update_layout(
            title="3D Feature Space Visualization",
            scene=dict(
                xaxis_title=f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)",
                yaxis_title=f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)",
                zaxis_title=f"PC3 ({pca.explained_variance_ratio_[2]*100:.1f}%)"
            ),
            height=700,
            template="plotly_white"
        )
        
        return fig
    
    def plot_attention_weights(self, attention_weights, sequence_data=None):
        """Visualize attention weights for sequence models"""
        fig = go.Figure()
        
        if attention_weights.ndim == 3:
            # Average over heads if multi-head attention
            attention_weights = np.mean(attention_weights, axis=0)
        
        # Create heatmap
        heatmap = go.Heatmap(
            z=attention_weights,
            colorscale='Viridis',
            colorbar=dict(title="Attention Weight")
        )
        
        fig.add_trace(heatmap)
        
        # Add sequence labels if available
        if sequence_data is not None:
            fig.update_xaxes(
                ticktext=sequence_data.get('tokens', []),
                tickvals=list(range(len(attention_weights[0])))
            )
            fig.update_yaxes(
                ticktext=sequence_data.get('tokens', []),
                tickvals=list(range(len(attention_weights)))
            )
        
        fig.update_layout(
            title="Attention Weights Visualization",
            xaxis_title="Key Sequence",
            yaxis_title="Query Sequence",
            height=600,
            width=800,
            template="plotly_white"
        )
        
        return fig
    
    def create_statistical_summary(self, data_dict, title="Statistical Summary"):
        """Create statistical summary visualization"""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Distribution',
                'Box Plot',
                'Violin Plot',
                'Q-Q Plot'
            )
        )
        
        # Extract data
        if isinstance(data_dict, dict):
            data = list(data_dict.values())[0]
        else:
            data = data_dict
        
        # 1. Histogram
        fig.add_trace(
            go.Histogram(
                x=data,
                nbinsx=30,
                name='Distribution',
                marker_color='skyblue'
            ),
            row=1, col=1
        )
        
        # 2. Box Plot
        fig.add_trace(
            go.Box(
                y=data,
                name='Box Plot',
                marker_color='lightgreen'
            ),
            row=1, col=2
        )
        
        # 3. Violin Plot
        fig.add_trace(
            go.Violin(
                y=data,
                name='Violin Plot',
                marker_color='lightcoral'
            ),
            row=2, col=1
        )
        
        # 4. Q-Q Plot
        from scipy import stats as sp_stats
        qq = sp_stats.probplot(data, dist="norm")
        x = qq[0][0]
        y = qq[0][1]
        
        fig.add_trace(
            go.Scatter(
                x=x, y=y,
                mode='markers',
                name='Q-Q Plot',
                marker_color='orange'
            ),
            row=2, col=2
        )
        
        # Add theoretical line for Q-Q plot
        fig.add_trace(
            go.Scatter(
                x=[x[0], x[-1]],
                y=[y[0], y[-1]],
                mode='lines',
                name='Theoretical',
                line=dict(color='red', dash='dash'),
                showlegend=False
            ),
            row=2, col=2
        )
        
        fig.update_layout(
            height=800,
            width=1000,
            title_text=title,
            showlegend=True,
            template="plotly_white"
        )
        
        return fig
    
    def visualize_model_architecture(self, model, input_size=(1, 128, 128)):
        """Visualize neural network architecture"""
        try:
            import torchviz
            from torchviz import make_dot
            
            # Create dummy input
            dummy_input = torch.randn(1, *input_size)
            
            # Forward pass
            output = model(dummy_input)
            
            # Create visualization
            dot = make_dot(output, params=dict(model.named_parameters()))
            
            # Save as image
            dot.render(f"{self.save_dir}/model_architecture", format="png")
            
            # Also create a text summary
            model_summary = str(model)
            with open(f"{self.save_dir}/model_summary.txt", "w") as f:
                f.write(model_summary)
            
            return dot
            
        except ImportError:
            print("torchviz not installed. Install with: pip install torchviz")
            return None
    
    def plot_correlation_matrix(self, features_df, method='pearson'):
        """Plot correlation matrix of features"""
        # Calculate correlation matrix
        corr_matrix = features_df.corr(method=method)
        
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.index,
            colorscale='RdBu',
            zmin=-1,
            zmax=1,
            colorbar=dict(title="Correlation")
        ))
        
        fig.update_layout(
            title=f"Feature Correlation Matrix ({method.capitalize()})",
            height=800,
            width=900,
            xaxis_showgrid=False,
            yaxis_showgrid=False,
            template="plotly_white"
        )
        
        # Highlight high correlations
        high_corr = np.where(np.abs(corr_matrix.values) > 0.8)
        annotations = []
        for i, j in zip(*high_corr):
            if i != j:  # Skip diagonal
                annotations.append(dict(
                    x=corr_matrix.columns[j],
                    y=corr_matrix.index[i],
                    text=f"{corr_matrix.values[i, j]:.2f}",
                    showarrow=False,
                    font=dict(color="white" if abs(corr_matrix.values[i, j]) > 0.5 else "black")
                ))
        
        fig.update_layout(annotations=annotations)
        
        return fig
    
    def create_interactive_timeline(self, timeline_data):
        """Create interactive timeline visualization"""
        fig = go.Figure()
        
        for event in timeline_data:
            fig.add_trace(go.Scatter(
                x=[event['timestamp']],
                y=[event['value']],
                mode='markers+text',
                name=event.get('category', 'Event'),
                text=event.get('description', ''),
                textposition="top center",
                marker=dict(
                    size=event.get('size', 10),
                    color=event.get('color', 'blue'),
                    symbol=event.get('symbol', 'circle')
                )
            ))
        
        # Add trend line if available
        if 'trend' in timeline_data[0]:
            trend_x = [e['timestamp'] for e in timeline_data]
            trend_y = [e['trend'] for e in timeline_data]
            
            fig.add_trace(go.Scatter(
                x=trend_x,
                y=trend_y,
                mode='lines',
                name='Trend',
                line=dict(color='red', width=2, dash='dash')
            ))
        
        fig.update_layout(
            title="Mental Health Timeline",
            xaxis_title="Time",
            yaxis_title="Assessment Value",
            hovermode='closest',
            height=600,
            template="plotly_white"
        )
        
        return fig
    
    def plot_feature_importance_waterfall(self, feature_importance):
        """Create waterfall plot of feature importance"""
        # Sort features by importance
        sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
        features = [f[0] for f in sorted_features[:15]]  # Top 15 features
        importance = [f[1] for f in sorted_features[:15]]
        
        fig = go.Figure(go.Waterfall(
            name="Feature Importance",
            orientation="v",
            measure=["relative"] * len(features),
            x=features,
            y=importance,
            text=[f"{val:.3f}" for val in importance],
            textposition="outside",
            connector={"line": {"color": "rgb(63, 63, 63)"}},
        ))
        
        fig.update_layout(
            title="Feature Importance Waterfall",
            showlegend=False,
            height=600,
            template="plotly_white"
        )
        
        return fig

class RealTimeVisualization:
    """Real-time visualization for live analysis"""
    
    def __init__(self):
        self.figures = {}
        self.data_history = {}
        
    def update_live_plot(self, figure_id, new_data):
        """Update live plot with new data"""
        if figure_id not in self.data_history:
            self.data_history[figure_id] = []
        
        self.data_history[figure_id].append(new_data)
        
        # Keep only last 100 points
        if len(self.data_history[figure_id]) > 100:
            self.data_history[figure_id] = self.data_history[figure_id][-100:]
        
        return self.data_history[figure_id]
    
    def create_live_dashboard(self, metrics_dict):
        """Create live updating dashboard"""
        import dash
        from dash import dcc, html
        from dash.dependencies import Input, Output
        import plotly.graph_objs as go
        
        app = dash.Dash(__name__)
        
        app.layout = html.Div([
            html.H1("Live Mental Health Assessment Dashboard"),
            
            html.Div([
                dcc.Graph(id='live-risk-gauge'),
                dcc.Graph(id='live-features-plot'),
                dcc.Graph(id='live-temporal-plot')
            ], style={'columnCount': 3}),
            
            dcc.Interval(
                id='interval-component',
                interval=1000,  # Update every second
                n_intervals=0
            )
        ])
        
        @app.callback(
            [Output('live-risk-gauge', 'figure'),
             Output('live-features-plot', 'figure'),
             Output('live-temporal-plot', 'figure')],
            [Input('interval-component', 'n_intervals')]
        )
        def update_graphs(n):
            # Create risk gauge
            risk_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=metrics_dict.get('risk_score', 50),
                title={'text': "Risk Score"},
                gauge={'axis': {'range': [0, 100]}}
            ))
            
            # Create features plot
            features_plot = go.Figure()
            for feature, values in metrics_dict.get('features', {}).items():
                features_plot.add_trace(go.Scatter(
                    y=values,
                    mode='lines',
                    name=feature
                ))
            
            # Create temporal plot
            temporal_plot = go.Figure(go.Scatter(
                y=metrics_dict.get('temporal_data', []),
                mode='lines+markers',
                name='Temporal Pattern'
            ))
            
            return risk_gauge, features_plot, temporal_plot
        
        return app