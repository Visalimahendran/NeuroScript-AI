import numpy as np
import time
import threading
import queue
from collections import deque
import cv2

class RealTimeAnalysisPipeline:
    """Real-time pipeline for handwriting analysis"""
    
    def __init__(self, model_path='saved_models/best_model.pth'):
        from src.neural_pressure import NeuralPressureEstimator
        from src.neuro_motor_features import NeuroMotorFeatureExtractor
        from src.inference import MentalHealthInference
        
        # Initialize components
        self.pressure_estimator = NeuralPressureEstimator()
        self.feature_extractor = NeuroMotorFeatureExtractor()
        self.mental_health_inference = MentalHealthInference(model_path)
        
        # Real-time buffers
        self.feature_buffer = deque(maxlen=30)  # Last 30 seconds of features
        self.npi_buffer = deque(maxlen=30)      # Last 30 NPI values
        self.mh_buffer = deque(maxlen=30)       # Last 30 mental health scores
        
        # Current analysis state
        self.current_analysis = None
        self.last_update_time = 0
        self.update_interval = 1.0  # Update every second
        
        # Threading
        self.analysis_queue = queue.Queue()
        self.running = False
        self.analysis_thread = None
        
        # Visualization data
        self.visualization_data = {
            'npi_history': [],
            'mh_history': [],
            'feature_history': [],
            'timestamps': []
        }
    
    def start_analysis(self):
        """Start real-time analysis thread"""
        self.running = True
        self.analysis_thread = threading.Thread(target=self._analysis_loop)
        self.analysis_thread.start()
        print("Real-time analysis started")
    
    def _analysis_loop(self):
        """Main analysis loop"""
        while self.running:
            try:
                # Get data from queue with timeout
                stroke_data = self.analysis_queue.get(timeout=0.1)
                
                # Perform analysis
                analysis_result = self._analyze_stroke_data(stroke_data)
                
                # Store result
                self.current_analysis = analysis_result
                
                # Update visualization data
                self._update_visualization_data(analysis_result)
                
                # Clear queue to prevent backlog
                while not self.analysis_queue.empty():
                    try:
                        self.analysis_queue.get_nowait()
                    except queue.Empty:
                        break
                
            except queue.Empty:
                # No new data, continue
                continue
            except Exception as e:
                print(f"Analysis error: {e}")
                continue
    
    def _analyze_stroke_data(self, stroke_data):
        """Analyze stroke data and produce comprehensive results"""
        
        # Extract neuro-motor features
        neuromotor_features = self.feature_extractor.extract_all_features(stroke_data)
        
        # Estimate Neural Pressure Index
        npi_result = self.pressure_estimator.estimate_neural_pressure(neuromotor_features)
        
        # Prepare image for mental health model
        if 'frames' in stroke_data and len(stroke_data['frames']) > 0:
            # Use last frame for mental health analysis
            last_frame = stroke_data['frames'][-1]
            
            # Convert to grayscale and resize
            gray_frame = cv2.cvtColor(last_frame, cv2.COLOR_BGR2GRAY)
            resized_frame = cv2.resize(gray_frame, (128, 128))
            
            # Get mental health prediction
            mh_result = self.mental_health_inference.predict(resized_frame)
        else:
            mh_result = {
                'prediction': 'Unknown',
                'confidence': 0.5,
                'risk_score': 50.0,
                'risk_level': 'Unknown'
            }
        
        # Combine results
        combined_result = {
            'timestamp': time.time(),
            'neuromotor_features': neuromotor_features,
            'neural_pressure': npi_result,
            'mental_health': mh_result,
            'combined_risk': self._calculate_combined_risk(npi_result, mh_result)
        }
        
        # Update buffers
        self.feature_buffer.append(neuromotor_features)
        self.npi_buffer.append(npi_result['npi_score'])
        self.mh_buffer.append(mh_result['risk_score'])
        
        return combined_result
    
    def _calculate_combined_risk(self, npi_result, mh_result):
        """Calculate combined risk score from NPI and MH assessment"""
        
        npi_score = npi_result['npi_score']
        mh_score = mh_result.get('risk_score', 50)
        
        # Weighted combination (60% NPI, 40% MH)
        combined_score = 0.6 * npi_score + 0.4 * mh_score
        
        # Categorize
        if combined_score < 30:
            risk_level = 'Low'
            recommendation = 'No immediate concerns. Maintain healthy habits.'
        elif combined_score < 60:
            risk_level = 'Moderate'
            recommendation = 'Monitor patterns. Consider stress management techniques.'
        else:
            risk_level = 'High'
            recommendation = 'Seek professional evaluation. Practice relaxation exercises.'
        
        return {
            'score': float(combined_score),
            'level': risk_level,
            'recommendation': recommendation,
            'npi_contribution': float(0.6 * npi_score),
            'mh_contribution': float(0.4 * mh_score)
        }
    
    def _update_visualization_data(self, analysis_result):
        """Update visualization data buffers"""
        current_time = time.time()
        
        self.visualization_data['timestamps'].append(current_time)
        self.visualization_data['npi_history'].append(
            analysis_result['neural_pressure']['npi_score']
        )
        self.visualization_data['mh_history'].append(
            analysis_result['mental_health']['risk_score']
        )
        
        # Keep only last 60 points
        max_points = 60
        for key in self.visualization_data:
            if len(self.visualization_data[key]) > max_points:
                self.visualization_data[key] = self.visualization_data[key][-max_points:]
    
    def add_stroke_data(self, stroke_data):
        """Add new stroke data for analysis"""
        if self.running:
            self.analysis_queue.put(stroke_data)
            return True
        return False
    
    def get_current_analysis(self):
        """Get current analysis results"""
        return self.current_analysis
    
    def get_visualization_data(self):
        """Get data for visualization"""
        return self.visualization_data.copy()
    
    def get_trend_analysis(self):
        """Get trend analysis over time"""
        if len(self.npi_buffer) < 5:
            return None
        
        npi_values = list(self.npi_buffer)
        mh_values = list(self.mh_buffer)
        
        # Calculate trends
        npi_trend = self._calculate_trend(npi_values)
        mh_trend = self._calculate_trend(mh_values)
        
        # Calculate correlation
        correlation = None
        if len(npi_values) == len(mh_values) and len(npi_values) >= 10:
            correlation = np.corrcoef(npi_values, mh_values)[0, 1]
        
        return {
            'npi_trend': npi_trend,
            'mh_trend': mh_trend,
            'correlation': correlation,
            'npi_mean': float(np.mean(npi_values)),
            'mh_mean': float(np.mean(mh_values)),
            'data_points': len(npi_values)
        }
    
    def _calculate_trend(self, values):
        """Calculate trend from time series"""
        if len(values) < 2:
            return {'direction': 'stable', 'slope': 0}
        
        x = np.arange(len(values))
        slope, _ = np.polyfit(x, values, 1)
        
        if slope > 0.1:
            direction = 'increasing'
        elif slope < -0.1:
            direction = 'decreasing'
        else:
            direction = 'stable'
        
        return {
            'direction': direction,
            'slope': float(slope),
            'magnitude': abs(float(slope)) * len(values)
        }
    
    def stop_analysis(self):
        """Stop real-time analysis"""
        self.running = False
        if self.analysis_thread:
            self.analysis_thread.join()
        print("Real-time analysis stopped")
    
    def generate_report(self, duration_minutes=5):
        """Generate comprehensive analysis report"""
        if not self.current_analysis:
            return None
        
        report = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'duration_minutes': duration_minutes,
            'summary': self.current_analysis['combined_risk'],
            'neural_pressure': self.current_analysis['neural_pressure'],
            'mental_health': self.current_analysis['mental_health'],
            'neuromotor_summary': self._summarize_neuromotor_features(
                self.current_analysis['neuromotor_features']
            ),
            'trends': self.get_trend_analysis(),
            'recommendations': self._generate_recommendations()
        }
        
        return report
    
    def _summarize_neuromotor_features(self, features):
        """Summarize neuromotor features for report"""
        summary = {}
        
        # Key feature categories
        categories = {
            'tremor': ['tremor_mean', 'tremor_power', 'jitter_index'],
            'pressure': ['pressure_variability', 'pressure_consistency'],
            'curvature': ['curvature_instability', 'slant_instability'],
            'temporal': ['velocity_cv', 'acceleration_cv', 'pause_ratio'],
            'symmetry': ['stroke_consistency', 'directional_symmetry']
        }
        
        for category, feature_list in categories.items():
            category_scores = []
            for feature in feature_list:
                if feature in features:
                    category_scores.append(features[feature])
            
            if category_scores:
                summary[category] = {
                    'mean': float(np.mean(category_scores)),
                    'min': float(np.min(category_scores)),
                    'max': float(np.max(category_scores))
                }
        
        return summary
    
    def _generate_recommendations(self):
        """Generate personalized recommendations"""
        if not self.current_analysis:
            return []
        
        recommendations = []
        npi_score = self.current_analysis['neural_pressure']['npi_score']
        mh_score = self.current_analysis['mental_health']['risk_score']
        
        # NPI-based recommendations
        if npi_score > 70:
            recommendations.append(
                "High neural pressure detected. Practice deep breathing exercises "
                "for 5 minutes, 3 times daily."
            )
        elif npi_score > 50:
            recommendations.append(
                "Moderate neural pressure. Consider mindfulness meditation "
                "for 10 minutes daily."
            )
        
        # MH-based recommendations
        if mh_score > 70:
            recommendations.append(
                "High mental health risk indicated. Consider consulting a "
                "mental health professional for evaluation."
            )
        elif mh_score > 50:
            recommendations.append(
                "Elevated mental health risk. Maintain regular sleep patterns "
                "and engage in physical activity."
            )
        
        # Feature-specific recommendations
        features = self.current_analysis['neuromotor_features']
        
        if 'tremor_mean' in features and features['tremor_mean'] > 0.1:
            recommendations.append(
                "Noticeable tremor detected. Reduce caffeine intake and "
                "ensure adequate rest."
            )
        
        if 'pressure_variability' in features and features['pressure_variability'] > 0.2:
            recommendations.append(
                "Pressure inconsistencies suggest stress. Try progressive "
                "muscle relaxation techniques."
            )
        
        if 'pause_ratio' in features and features['pause_ratio'] > 0.3:
            recommendations.append(
                "Frequent hesitations observed. Practice focused writing "
                "exercises to improve flow."
            )
        
        return recommendations