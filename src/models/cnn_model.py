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

class VisualizationEngine:
    """Visualization engine for mental health assessment results"""
    
    def __init__(self, save_dir='visualizations'):
        import os
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
        # Set style
        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set_palette("husl")
    
    def plot_feature_importance(self, feature_names, importance_scores, top_n=20):
        """Plot feature importance"""
        plt.figure(figsize=(12, 8))
        
        # Sort features by importance
        indices = np.argsort(importance_scores)[::-1][:top_n]
        sorted_names = [feature_names[i] for i in indices]
        sorted_scores = importance_scores[indices]
        
        # Create bar plot
        bars = plt.barh(range(len(sorted_names)), sorted_scores, align='center')
        plt.yticks(range(len(sorted_names)), sorted_names)
        plt.xlabel('Importance Score')
        plt.title(f'Top {top_n} Most Important Features')
        
        # Add value labels
        for i, (bar, score) in enumerate(zip(bars, sorted_scores)):
            plt.text(score, i, f'{score:.3f}', va='center')
        
        plt.tight_layout()
        plt.savefig(f'{self.save_dir}/feature_importance.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        # Create interactive plot
        fig = go.Figure(go.Bar(
            x=sorted_scores,
            y=sorted_names,
            orientation='h',
            marker=dict(color=sorted_scores, colorscale='Viridis')
        ))
        
        fig.update_layout(
            title=f'Top {top_n} Feature Importance',
            xaxis_title='Importance Score',
            yaxis_title='Features',
            height=600
        )
        
        fig.write_html(f'{self.save_dir}/feature_importance_interactive.html')
        
        return fig
    
    def plot_confusion_matrix(self, cm, class_names=['Normal', 'Mild', 'Severe']):
        """Plot confusion matrix"""
        plt.figure(figsize=(8, 6))
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=class_names,
                   yticklabels=class_names)
        
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        
        plt.tight_layout()
        plt.savefig(f'{self.save_dir}/confusion_matrix.png', dpi=150)
        plt.show()
        
        # Interactive heatmap
        fig = go.Figure(data=go.Heatmap(
            z=cm,
            x=class_names,
            y=class_names,
            colorscale='Blues',
            text=cm,
            texttemplate='%{text}',
            textfont={"size": 16}
        ))
        
        fig.update_layout(
            title='Confusion Matrix',
            xaxis_title='Predicted',
            yaxis_title='Actual',
            width=600,
            height=500
        )
        
        fig.write_html(f'{self.save_dir}/confusion_matrix_interactive.html')
        
        return fig
    
    def plot_roc_curves(self, fpr, tpr, roc_auc, class_names):
        """Plot ROC curves for multi-class classification"""
        plt.figure(figsize=(10, 8))
        
        # Plot each class
        for i, class_name in enumerate(class_names):
            plt.plot(fpr[i], tpr[i], 
                    label=f'{class_name} (AUC = {roc_auc[i]:.2f})',
                    linewidth=2)
        
        # Plot diagonal
        plt.plot([0, 1], [0, 1], 'k--', linewidth=2)
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Multi-class ROC Curves')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{self.save_dir}/roc_curves.png', dpi=150)
        plt.show()
        
        # Interactive ROC curve
        fig = go.Figure()
        
        for i, class_name in enumerate(class_names):
            fig.add_trace(go.Scatter(
                x=fpr[i], y=tpr[i],
                mode='lines',
                name=f'{class_name} (AUC = {roc_auc[i]:.2f})',
                line=dict(width=2)
            ))
        
        fig.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1],
            mode='lines',
            name='Random',
            line=dict(dash='dash', color='black')
        ))
        
        fig.update_layout(
            title='ROC Curves',
            xaxis_title='False Positive Rate',
            yaxis_title='True Positive Rate',
            width=800,
            height=600
        )
        
        fig.write_html(f'{self.save_dir}/roc_curves_interactive.html')
        
        return fig
    
    def plot_training_history(self, history):
        """Plot training history"""
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # Loss plot
        axes[0].plot(history['train_loss'], label='Train Loss', linewidth=2)
        axes[0].plot(history['val_loss'], label='Val Loss', linewidth=2)
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Training and Validation Loss')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Accuracy plot
        axes[1].plot(history['train_acc'], label='Train Accuracy', linewidth=2)
        axes[1].plot(history['val_acc'], label='Val Accuracy', linewidth=2)
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy (%)')
        axes[1].set_title('Training and Validation Accuracy')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{self.save_dir}/training_history.png', dpi=150)
        plt.show()
        
        return fig
    
    def visualize_handwriting_features(self, image, features, prediction):
        """Visualize handwriting features on image"""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Original image
        axes[0, 0].imshow(image, cmap='gray')
        axes[0, 0].set_title('Original Handwriting')
        axes[0, 0].axis('off')
        
        # Edge detection
        edges = cv2.Canny(image, 50, 150)
        axes[0, 1].imshow(edges, cmap='gray')
        axes[0, 1].set_title('Edge Detection')
        axes[0, 1].axis('off')
        
        # Skeleton
        from src.preprocessing import HandwritingPreprocessor
        preprocessor = HandwritingPreprocessor()
        skeleton = preprocessor.zhang_suen_skeletonization(image)
        axes[0, 2].imshow(skeleton, cmap='gray')
        axes[0, 2].set_title('Skeletonization')
        axes[0, 2].axis('off')
        
        # Contours
        contours, _ = cv2.findContours(
            (image > 127).astype(np.uint8), 
            cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        contour_img = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        cv2.drawContours(contour_img, contours, -1, (0, 255, 0), 2)
        axes[1, 0].imshow(contour_img)
        axes[1, 0].set_title(f'Contours ({len(contours)} strokes)')
        axes[1, 0].axis('off')
        
        # Feature bar chart
        if features:
            feature_names = list(features.keys())[:8]
            feature_values = list(features.values())[:8]
            axes[1, 1].barh(feature_names, feature_values)
            axes[1, 1].set_title('Top Features')
            axes[1, 1].set_xlabel('Value')
        
        # Prediction gauge
        axes[1, 2].axis('off')
        axes[1, 2].text(0.5, 0.6, f'Prediction: {prediction}', 
                       ha='center', va='center', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(f'{self.save_dir}/handwriting_analysis.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        return fig
    
    def plot_tsne_visualization(self, features, labels, title='t-SNE Visualization'):
        """Plot t-SNE visualization of features"""
        # Reduce dimensionality
        tsne = TSNE(n_components=2, random_state=42, perplexity=30)
        features_2d = tsne.fit_transform(features)
        
        plt.figure(figsize=(10, 8))
        
        # Scatter plot with different colors for each class
        scatter = plt.scatter(features_2d[:, 0], features_2d[:, 1], 
                            c=labels, cmap='viridis', alpha=0.7, s=50)
        
        plt.colorbar(scatter, label='Class')
        plt.title(title)
        plt.xlabel('t-SNE Component 1')
        plt.ylabel('t-SNE Component 2')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{self.save_dir}/tsne_visualization.png', dpi=150)
        plt.show()
        
        # Interactive plot
        fig = px.scatter(
            x=features_2d[:, 0], y=features_2d[:, 1],
            color=labels,
            title=title,
            labels={'color': 'Class'},
            opacity=0.7
        )
        
        fig.update_layout(
            width=800,
            height=600
        )
        
        fig.write_html(f'{self.save_dir}/tsne_interactive.html')
        
        return fig
    
    def create_dashboard(self, results_dict):
        """Create comprehensive dashboard"""
        fig = make_subplots(
            rows=2, cols=3,
            subplot_titles=('Risk Score Distribution', 'Feature Importance',
                          'Confusion Matrix', 'ROC Curve', 'Training History',
                          't-SNE Visualization'),
            specs=[[{'type': 'histogram'}, {'type': 'bar'}, {'type': 'heatmap'}],
                  [{'type': 'scatter'}, {'type': 'xy'}, {'type': 'scatter'}]]
        )
        
        # Add traces for each subplot
        # 1. Risk Score Distribution
        if 'risk_scores' in results_dict:
            fig.add_trace(
                go.Histogram(x=results_dict['risk_scores'], name='Risk Scores'),
                row=1, col=1
            )
        
        # 2. Feature Importance
        if 'feature_importance' in results_dict:
            fig.add_trace(
                go.Bar(x=results_dict['feature_names'][:10],
                      y=results_dict['feature_importance'][:10],
                      name='Feature Importance'),
                row=1, col=2
            )
        
        # 3. Confusion Matrix
        if 'confusion_matrix' in results_dict:
            fig.add_trace(
                go.Heatmap(z=results_dict['confusion_matrix'],
                          x=['Normal', 'Mild', 'Severe'],
                          y=['Normal', 'Mild', 'Severe'],
                          colorscale='Blues'),
                row=1, col=3
            )
        
        # 4. ROC Curve
        if 'roc_data' in results_dict:
            for i, (fpr, tpr) in enumerate(results_dict['roc_data']):
                fig.add_trace(
                    go.Scatter(x=fpr, y=tpr, mode='lines',
                              name=f'Class {i}'),
                    row=2, col=1
                )
        
        # 5. Training History
        if 'training_history' in results_dict:
            history = results_dict['training_history']
            fig.add_trace(
                go.Scatter(y=history['train_loss'], mode='lines',
                          name='Train Loss'),
                row=2, col=2
            )
            fig.add_trace(
                go.Scatter(y=history['val_loss'], mode='lines',
                          name='Val Loss'),
                row=2, col=2
            )
        
        # 6. t-SNE Visualization
        if 'tsne_features' in results_dict:
            fig.add_trace(
                go.Scatter(x=results_dict['tsne_features'][:, 0],
                          y=results_dict['tsne_features'][:, 1],
                          mode='markers',
                          marker=dict(color=results_dict['tsne_labels'],
                                     colorscale='Viridis')),
                row=2, col=3
            )
        
        fig.update_layout(height=800, width=1200,
                         title_text="Mental Health Assessment Dashboard",
                         showlegend=True)
        
        fig.write_html(f'{self.save_dir}/comprehensive_dashboard.html')
        
        return fig
    
    def plot_stress_timeline(self, timestamps, stress_scores, events=None):
        """Plot stress score timeline"""
        plt.figure(figsize=(12, 6))
        
        # Convert timestamps if needed
        if isinstance(timestamps[0], str):
            import pandas as pd
            timestamps = pd.to_datetime(timestamps)
        
        plt.plot(timestamps, stress_scores, 'b-', linewidth=2, label='Stress Score')
        plt.fill_between(timestamps, stress_scores, alpha=0.3)
        
        # Add threshold lines
        plt.axhline(y=30, color='g', linestyle='--', alpha=0.5, label='Normal Threshold')
        plt.axhline(y=70, color='r', linestyle='--', alpha=0.5, label='Severe Threshold')
        
        # Add events if provided
        if events:
            for event_time, event_text in events:
                plt.axvline(x=event_time, color='orange', linestyle=':', alpha=0.7)
                plt.text(event_time, max(stress_scores) * 0.9, event_text,
                        rotation=90, fontsize=8)
        
        plt.xlabel('Time')
        plt.ylabel('Stress Score')
        plt.title('Stress Score Timeline')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        plt.savefig(f'{self.save_dir}/stress_timeline.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        # Interactive timeline
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=timestamps, y=stress_scores,
            mode='lines+markers',
            name='Stress Score',
            fill='tozeroy'
        ))
        
        fig.add_hline(y=30, line_dash="dot", line_color="green",
                     annotation_text="Normal Threshold")
        fig.add_hline(y=70, line_dash="dot", line_color="red",
                     annotation_text="Severe Threshold")
        
        fig.update_layout(
            title='Stress Score Timeline',
            xaxis_title='Time',
            yaxis_title='Stress Score',
            hovermode='x unified',
            height=500
        )
        
        fig.write_html(f'{self.save_dir}/stress_timeline_interactive.html')
        
        return fig

class RealTimeVisualizer:
    """Real-time visualization for webcam input"""
    
    def __init__(self):
        self.colors = {
            'Normal': (0, 255, 0),    # Green
            'Mild': (0, 165, 255),    # Orange
            'Severe': (0, 0, 255)     # Red
        }
    
    def draw_analysis_overlay(self, frame, analysis_result):
        """Draw analysis overlay on frame"""
        overlay = frame.copy()
        
        # Semi-transparent background for text
        cv2.rectangle(overlay, (10, 10), (400, 200), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.7, frame, 0.3, 0)
        
        # Get prediction color
        prediction = analysis_result['prediction']
        color = self.colors.get(prediction, (255, 255, 255))
        
        # Draw text
        y_offset = 40
        line_height = 30
        
        texts = [
            f"Prediction: {prediction}",
            f"Confidence: {analysis_result['confidence']:.1%}",
            f"Risk Score: {analysis_result['risk_score']:.1f}",
            f"Risk Level: {analysis_result['risk_level']}"
        ]
        
        for i, text in enumerate(texts):
            cv2.putText(frame, text, (20, y_offset + i * line_height),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        # Draw risk gauge
        self.draw_risk_gauge(frame, analysis_result['risk_score'])
        
        return frame
    
    def draw_risk_gauge(self, frame, risk_score):
        """Draw risk gauge visualization"""
        height, width = frame.shape[:2]
        
        # Gauge position and size
        gauge_x = width - 200
        gauge_y = 50
        gauge_width = 150
        gauge_height = 20
        
        # Draw gauge background
        cv2.rectangle(frame, 
                     (gauge_x, gauge_y),
                     (gauge_x + gauge_width, gauge_y + gauge_height),
                     (100, 100, 100), -1)
        
        # Draw colored segments
        segment_width = gauge_width // 3
        
        # Green segment (0-33)
        cv2.rectangle(frame,
                     (gauge_x, gauge_y),
                     (gauge_x + segment_width, gauge_y + gauge_height),
                     (0, 255, 0), -1)
        
        # Orange segment (34-66)
        cv2.rectangle(frame,
                     (gauge_x + segment_width, gauge_y),
                     (gauge_x + 2 * segment_width, gauge_y + gauge_height),
                     (0, 165, 255), -1)
        
        # Red segment (67-100)
        cv2.rectangle(frame,
                     (gauge_x + 2 * segment_width, gauge_y),
                     (gauge_x + gauge_width, gauge_y + gauge_height),
                     (0, 0, 255), -1)
        
        # Draw needle
        needle_x = gauge_x + int((risk_score / 100) * gauge_width)
        cv2.line(frame,
                (needle_x, gauge_y - 10),
                (needle_x, gauge_y + gauge_height + 10),
                (255, 255, 255), 3)
        
        # Add labels
        cv2.putText(frame, "Low", (gauge_x, gauge_y - 15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, "Risk", (gauge_x, gauge_y + gauge_height + 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Add current score
        cv2.putText(frame, f"{risk_score:.0f}", 
                   (needle_x - 10, gauge_y - 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    def create_live_plot(self, history_data):
        """Create live updating plot for real-time visualization"""
        from matplotlib.animation import FuncAnimation
        
        fig, ax = plt.subplots(figsize=(10, 4))
        
        # Initialize empty plot
        line, = ax.plot([], [], 'b-', linewidth=2)
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 100)
        ax.set_xlabel('Time (samples)')
        ax.set_ylabel('Risk Score')
        ax.set_title('Real-time Risk Assessment')
        ax.grid(True, alpha=0.3)
        
        # Add threshold lines
        ax.axhline(y=30, color='g', linestyle='--', alpha=0.5)
        ax.axhline(y=70, color='r', linestyle='--', alpha=0.5)
        
        def update(frame_num):
            """Update plot with new data"""
            if len(history_data) > 0:
                x_data = range(len(history_data))
                y_data = history_data
                
                line.set_data(x_data, y_data)
                
                # Adjust x-axis limits
                if len(history_data) > 100:
                    ax.set_xlim(len(history_data) - 100, len(history_data))
                else:
                    ax.set_xlim(0, max(100, len(history_data)))
            
            return line,
        
        anim = FuncAnimation(fig, update, interval=1000, blit=True)
        
        return anim, fig