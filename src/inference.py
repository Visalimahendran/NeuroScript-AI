import os
from pyexpat import model
import torch
import numpy as np
import cv2
from PIL import Image
import torch.nn.functional as F
import json
from models.hybrid_model import HybridMentalHealthModel
class MentalHealthInference:
    """
    Inference pipeline for mental health assessment from handwriting
    """
    
    def __init__(self, model_path):
        self.device = torch.device("cpu")
        self.model = self.load_model(model_path)
        self.model.eval()
        
        # Label mapping
        self.label_map = {
            0: {'class': 'Normal', 'color': 'green', 'risk': 'Low'},
            1: {'class': 'Mild', 'color': 'orange', 'risk': 'Medium'},
            2: {'class': 'Severe', 'color': 'red', 'risk': 'High'}
        }
        
        # Feature importance (can be learned from model)
        self.feature_importance = {
            'stroke_tremor': 0.25,
            'pressure_variability': 0.20,
            'slant_inconsistency': 0.15,
            'size_variation': 0.15,
            'spacing_irregularity': 0.10,
            'speed_fluctuation': 0.10,
            'curvature_abnormality': 0.05
        }
    
    def load_model(self, model_path):
        device = torch.device("cpu")
        """Load trained model"""
        checkpoint = torch.load(model_path, map_location=device)
        model = HybridMentalHealthModel()
        
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint)
        model.to(device)
        model.eval()

        return model
    
    def preprocess_image(self, image):
        """
        Preprocess input image for model inference
        Supports various input types: file path, numpy array, PIL Image
        """
        # Convert to numpy array
        if isinstance(image, str):
            # Load from file
            img = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
        elif isinstance(image, Image.Image):
            # Convert PIL to numpy
            img = np.array(image.convert('L'))
        elif isinstance(image, np.ndarray):
            if len(image.shape) == 3:
                # Convert to grayscale
                img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                img = image
        else:
            raise ValueError("Unsupported image type")
        
        # Resize to model input size
        img = cv2.resize(img, (128, 128))
        
        # Normalize
        img = img.astype(np.float32) / 255.0
        
        # Add batch and channel dimensions
        img = np.expand_dims(img, axis=0)  # Add channel dimension
        img = np.expand_dims(img, axis=0)  # Add batch dimension
        
        # Convert to tensor
        tensor = torch.FloatTensor(img).to(self.device)
        
        return tensor
    
    def predict(self, image):
        """
        Main prediction function
        Returns: prediction, confidence, and detailed analysis
        """
        # Preprocess
        input_tensor = self.preprocess_image(image)
        
        # Model inference
        with torch.no_grad():
            if hasattr(self.model, 'forward_with_features'):
                output, features = self.model.forward_with_features(input_tensor)
            else:
                output = self.model(input_tensor)
            
            # Get probabilities
            probabilities = F.softmax(output, dim=1)
            confidence, predicted_class = torch.max(probabilities, 1)
            
            predicted_class = predicted_class.item()
            confidence = confidence.item()
        
        # Generate detailed analysis
        analysis = self.analyze_features(input_tensor, predicted_class, confidence)
        
        # Calculate risk score (0-100)
        risk_score = self.calculate_risk_score(predicted_class, confidence, analysis)
        
        return {
            'prediction': self.label_map[predicted_class]['class'],
            'confidence': float(confidence),
            'risk_level': self.label_map[predicted_class]['risk'],
            'risk_score': float(risk_score),
            'recommendation': self.generate_recommendation(predicted_class, risk_score),
            'detailed_analysis': analysis
        }
    
    def analyze_features(self, image_tensor, predicted_class, confidence):
        """Analyze handwriting features for detailed report"""
        
        # Extract features using intermediate layers (if available)
        features = self.extract_intermediate_features(image_tensor)
        
        # Simulate feature extraction (in practice, use actual feature extractor)
        analysis = {
            'stroke_characteristics': {
                'tremor_index': np.random.uniform(0.1, 0.9) if predicted_class > 0 else np.random.uniform(0, 0.3),
                'pressure_consistency': np.random.uniform(0.6, 1.0) if predicted_class == 0 else np.random.uniform(0, 0.6),
                'slant_stability': np.random.uniform(0.7, 1.0) if predicted_class == 0 else np.random.uniform(0.3, 0.7),
            },
            'spatial_properties': {
                'size_variation': np.random.uniform(0.1, 0.4) if predicted_class == 0 else np.random.uniform(0.5, 0.9),
                'spacing_regularity': np.random.uniform(0.7, 1.0) if predicted_class == 0 else np.random.uniform(0.3, 0.7),
                'alignment_deviation': np.random.uniform(0.1, 0.3) if predicted_class == 0 else np.random.uniform(0.4, 0.8),
            },
            'temporal_properties': {
                'speed_consistency': np.random.uniform(0.7, 1.0) if predicted_class == 0 else np.random.uniform(0.3, 0.7),
                'pause_pattern': 'Regular' if predicted_class == 0 else 'Irregular',
                'stroke_fluency': 'Smooth' if predicted_class == 0 else 'Jerky',
            }
        }
        
        # Calculate abnormality scores for each feature
        for category in analysis:
            if isinstance(analysis[category], dict):
                abnormality_score = 0
                for feature, value in analysis[category].items():
                    if isinstance(value, (int, float)):
                        abnormality_score += (1 - value) * self.feature_importance.get(feature, 0.1)
                analysis[category]['abnormality_score'] = abnormality_score
        
        return analysis
    
    def extract_intermediate_features(self, image_tensor):
        """Extract features from intermediate model layers"""
        features = {}
        
        # Hook to get intermediate activations
        activations = {}
        
        def get_activation(name):
            def hook(model, input, output):
                activations[name] = output.detach()
            return hook
        
        # Register hooks to some layers
        hooks = []
        layer_names = ['conv_layers.3', 'conv_layers.7', 'global_pool']
        
        for name, layer in self.model.named_modules():
            if any(layer_name in name for layer_name in layer_names):
                hook = layer.register_forward_hook(get_activation(name))
                hooks.append(hook)
        
        # Forward pass to capture activations
        with torch.no_grad():
            _ = self.model(image_tensor)
        
        # Remove hooks
        for hook in hooks:
            hook.remove()
        
        # Process activations
        for name, activation in activations.items():
            # Calculate statistics
            features[f'{name}_mean'] = activation.mean().item()
            features[f'{name}_std'] = activation.std().item()
            features[f'{name}_max'] = activation.max().item()
        
        return features
    
    def calculate_risk_score(self, predicted_class, confidence, analysis):
        """Calculate comprehensive risk score (0-100)"""

        # Base risk from class
        base_score = predicted_class * 33.33  # 0, 33.33, 66.66

        # Confidence adjustment
        confidence_adjustment = (confidence - 0.5) * 20  # ±10

        # Feature abnormality aggregation
        feature_abnormality = 0.0
        for category, values in analysis.items():
            if isinstance(values, dict) and 'abnormality_score' in values:
                feature_abnormality += values['abnormality_score']

        # Feature impact
        feature_adjustment = feature_abnormality * 30

        # Final score
        risk_score = base_score + confidence_adjustment + feature_adjustment

        return max(0, min(100, risk_score))

    
    def generate_recommendation(self, predicted_class, risk_score):
        """Generate personalized recommendations"""
        
        recommendations = {
            'immediate': [],
            'short_term': [],
            'long_term': []
        }
        
        if predicted_class == 0:  # Normal
            recommendations['immediate'].append("No immediate action needed")
            recommendations['short_term'].append("Maintain healthy writing habits")
            recommendations['long_term'].append("Regular mental health check-ups")
            
        elif predicted_class == 1:  # Mild
            recommendations['immediate'].append("Consider stress management techniques")
            recommendations['short_term'].append("Practice mindfulness exercises")
            recommendations['short_term'].append("Consult with a general practitioner")
            recommendations['long_term'].append("Develop regular relaxation routine")
            
        else:  # Severe
            recommendations['immediate'].append("Consult mental health professional")
            recommendations['immediate'].append("Engage in relaxation exercises")
            recommendations['short_term'].append("Regular therapy sessions recommended")
            recommendations['short_term'].append("Consider lifestyle adjustments")
            recommendations['long_term'].append("Develop comprehensive wellness plan")
        
        # Add risk-based recommendations
        if risk_score > 70:
            recommendations['immediate'].append("Urgent professional evaluation advised")
        
        return recommendations
    
    def batch_predict(self, image_paths):
        """Predict for multiple images"""
        results = []
        
        for image_path in image_paths:
            try:
                result = self.predict(image_path)
                result['file_path'] = image_path
                results.append(result)
            except Exception as e:
                print(f"Error processing {image_path}: {e}")
                results.append({
                    'file_path': image_path,
                    'error': str(e)
                })
        
        return results
    
    

    def save_report(self, prediction_result, output_path):
        """Save detailed prediction report"""

        # ✅ Ensure directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        report = {
            'timestamp': str(np.datetime64('now')),
            'assessment': prediction_result,
            'model_info': {
                'type': str(type(self.model).__name__),
                'device': str(self.device)
            }
        }
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)

        print(f"Report saved to {output_path}")
    

class RealTimeWebcamInference:
    """Real-time inference using webcam"""
    
    def __init__(self, model_path):
        self.inference_engine = MentalHealthInference(model_path)
        self.cap = cv2.VideoCapture(0)
        
    def run(self):
        """Run real-time webcam inference"""
        print("Starting real-time handwriting analysis...")
        print("Press 's' to analyze current frame, 'q' to quit")
        
        analysis_results = []
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            # Display instructions
            cv2.putText(frame, "Press 's' to analyze, 'q' to quit", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Show current frame
            cv2.imshow('Handwriting Analysis', frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('s'):
                # Analyze current frame
                result = self.inference_engine.predict(frame)
                
                # Display results
                self.display_results(frame, result)
                analysis_results.append(result)
                
            elif key == ord('q'):
                break
        
        self.cap.release()
        cv2.destroyAllWindows()
        
        return analysis_results
    
    def display_results(self, frame, result):
        """Display prediction results on frame"""
        
        # Create overlay
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 50), (400, 250), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.7, frame, 0.3, 0)
        
        # Display prediction
        color_map = {'Normal': (0, 255, 0), 'Mild': (0, 165, 255), 'Severe': (0, 0, 255)}
        color = color_map.get(result['prediction'], (255, 255, 255))
        
        cv2.putText(frame, f"Prediction: {result['prediction']}", 
                   (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.putText(frame, f"Confidence: {result['confidence']:.2%}", 
                   (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Risk Score: {result['risk_score']:.1f}/100", 
                   (20, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Risk Level: {result['risk_level']}", 
                   (20, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        cv2.imshow('Analysis Results', frame)
        cv2.waitKey(3000)  # Show for 3 seconds