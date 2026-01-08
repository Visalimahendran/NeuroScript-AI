import numpy as np
from scipy import signal, stats
import warnings
warnings.filterwarnings('ignore')

class NeuralPressureEstimator:
    """Neural Pressure Index (NPI) estimation from handwriting features"""
    
    def __init__(self):
        # Feature weights for NPI calculation
        self.feature_weights = {
            # Tremor-related features
            'tremor_mean': 0.15,
            'tremor_power': 0.10,
            'jitter_index': 0.10,
            
            # Pressure-related features
            'pressure_variability': 0.15,
            'pressure_autocorrelation_lag1': -0.05,  # Negative weight (higher correlation = lower stress)
            
            # Temporal features
            'velocity_cv': 0.10,
            'acceleration_cv': 0.10,
            'pause_ratio': 0.10,
            
            # Curvature features
            'curvature_instability': 0.10,
            'slant_instability': 0.05,
            
            # Symmetry features
            'stroke_consistency': -0.10,  # Negative weight (higher consistency = lower stress)
            'directional_symmetry': -0.10  # Negative weight
        }
        
        # Normalization ranges for features
        self.normalization_ranges = {
            'tremor_mean': (0, 0.5),
            'tremor_power': (0, 10),
            'jitter_index': (0, 5),
            'pressure_variability': (0, 0.5),
            'pressure_autocorrelation_lag1': (-1, 1),
            'velocity_cv': (0, 2),
            'acceleration_cv': (0, 3),
            'pause_ratio': (0, 1),
            'curvature_instability': (0, 0.5),
            'slant_instability': (0, 5),
            'stroke_consistency': (0, 1),
            'directional_symmetry': (0, 1)
        }
        
    def estimate_neural_pressure(self, neuromotor_features):
        """
        Estimate Neural Pressure Index (NPI)
        
        NPI is a composite index (0-100) representing estimated neural pressure
        based on handwriting motor control patterns
        """
        
        # Initialize NPI components
        npi_components = {}
        
        # Process each feature
        for feature, weight in self.feature_weights.items():
            if feature in neuromotor_features:
                value = neuromotor_features[feature]
                
                # Normalize feature value
                if feature in self.normalization_ranges:
                    min_val, max_val = self.normalization_ranges[feature]
                    normalized = (value - min_val) / (max_val - min_val + 1e-10)
                    normalized = np.clip(normalized, 0, 1)
                else:
                    normalized = np.clip(value, 0, 1)
                
                # Apply weight direction
                if weight < 0:  # Inverse relationship
                    npi_components[feature] = (1 - normalized) * abs(weight)
                else:
                    npi_components[feature] = normalized * weight
        
        # Calculate total NPI
        total_weight = sum(abs(w) for w in self.feature_weights.values())
        
        if total_weight > 0:
            npi_score = sum(npi_components.values()) / total_weight * 100
        else:
            npi_score = 50  # Default middle value
        
        # Apply nonlinear transformation for better discrimination
        npi_score = self._apply_nonlinear_transform(npi_score)
        
        # Calculate confidence based on available features
        confidence = self._calculate_confidence(neuromotor_features)
        
        # Categorize NPI level
        npi_category = self._categorize_npi(npi_score)
        
        # Calculate contributing factors
        contributing_factors = self._identify_contributing_factors(npi_components)
        
        return {
            'npi_score': float(npi_score),
            'npi_category': npi_category,
            'npi_confidence': float(confidence),
            'npi_components': npi_components,
            'contributing_factors': contributing_factors,
            'interpretation': self._generate_interpretation(npi_score, contributing_factors)
        }
    
    def _apply_nonlinear_transform(self, npi_score):
        """Apply nonlinear transformation to enhance discrimination"""
        # Sigmoid-like transformation to emphasize extreme values
        transformed = 100 * (1 / (1 + np.exp(-0.1 * (npi_score - 50))))
        return transformed
    
    def _calculate_confidence(self, features):
        """Calculate confidence in NPI estimation"""
        available_features = sum(1 for f in self.feature_weights.keys() if f in features)
        total_features = len(self.feature_weights)
        
        # Base confidence based on feature coverage
        coverage_confidence = available_features / total_features
        
        # Additional confidence based on feature quality
        quality_indicators = []
        
        if 'velocity_cv' in features and features['velocity_cv'] > 0:
            quality_indicators.append(0.8)  # Good temporal data
        
        if 'pressure_variability' in features:
            quality_indicators.append(0.7)  # Pressure data available
        
        if 'tremor_mean' in features and features['tremor_mean'] > 0:
            quality_indicators.append(0.9)  # Tremor data available
        
        if quality_indicators:
            quality_confidence = np.mean(quality_indicators)
        else:
            quality_confidence = 0.5
        
        # Combined confidence
        confidence = 0.7 * coverage_confidence + 0.3 * quality_confidence
        
        return min(confidence, 1.0)
    
    def _categorize_npi(self, npi_score):
        """Categorize NPI into levels"""
        if npi_score < 25:
            return 'Very Low'
        elif npi_score < 40:
            return 'Low'
        elif npi_score < 60:
            return 'Moderate'
        elif npi_score < 75:
            return 'High'
        else:
            return 'Very High'
    
    def _identify_contributing_factors(self, npi_components):
        """Identify main contributing factors to NPI"""
        if not npi_components:
            return []
        
        # Sort components by contribution
        sorted_components = sorted(npi_components.items(), 
                                  key=lambda x: x[1], 
                                  reverse=True)
        
        # Get top 3 contributing factors
        top_factors = []
        for feature, contribution in sorted_components[:3]:
            if contribution > 0.05:  # Only include significant contributions
                factor_name = self._get_factor_name(feature)
                top_factors.append({
                    'factor': factor_name,
                    'contribution': float(contribution * 100),  # Percentage
                    'feature': feature
                })
        
        return top_factors
    
    def _get_factor_name(self, feature):
        """Get human-readable factor name"""
        factor_names = {
            'tremor_mean': 'Hand Tremor',
            'tremor_power': 'Tremor Intensity',
            'jitter_index': 'Movement Jitter',
            'pressure_variability': 'Pressure Inconsistency',
            'pressure_autocorrelation_lag1': 'Pressure Rhythm',
            'velocity_cv': 'Speed Variability',
            'acceleration_cv': 'Acceleration Irregularity',
            'pause_ratio': 'Hesitation Patterns',
            'curvature_instability': 'Curvature Instability',
            'slant_instability': 'Slant Irregularity',
            'stroke_consistency': 'Stroke Consistency',
            'directional_symmetry': 'Directional Symmetry'
        }
        
        return factor_names.get(feature, feature.replace('_', ' ').title())
    
    def _generate_interpretation(self, npi_score, contributing_factors):
        """Generate interpretation text for NPI"""
        
        interpretations = {
            'Very Low': "Minimal neural pressure detected. Handwriting shows smooth, "
                       "controlled movements with good motor coordination.",
            'Low': "Low neural pressure. Writing patterns show good control with "
                  "minor inconsistencies.",
            'Moderate': "Moderate neural pressure. Some signs of motor control "
                       "challenges are present.",
            'High': "High neural pressure detected. Significant motor control "
                   "difficulties observed with notable tremor or instability.",
            'Very High': "Very high neural pressure. Severe motor control issues "
                        "indicating high stress or neurological load."
        }
        
        base_interpretation = interpretations.get(
            self._categorize_npi(npi_score),
            "Neural pressure assessment completed."
        )
        
        # Add factor-specific details
        if contributing_factors:
            factor_details = []
            for factor in contributing_factors[:2]:  # Top 2 factors
                factor_details.append(
                    f"{factor['factor']} ({factor['contribution']:.1f}% contribution)"
                )
            
            if factor_details:
                base_interpretation += f" Main factors: {', '.join(factor_details)}."
        
        return base_interpretation
    
    def calculate_dynamic_npi(self, time_series_features):
        """
        Calculate dynamic NPI over time for real-time analysis
        
        Args:
            time_series_features: List of feature dictionaries over time
        
        Returns:
            Dynamic NPI analysis with trends
        """
        if not time_series_features:
            return None
        
        npis = []
        categories = []
        
        for features in time_series_features:
            npi_result = self.estimate_neural_pressure(features)
            npis.append(npi_result['npi_score'])
            categories.append(npi_result['npi_category'])
        
        npis = np.array(npis)
        
        # Calculate trends
        if len(npis) > 5:
            # Linear trend
            x = np.arange(len(npis))
            slope, intercept = np.polyfit(x, npis, 1)
            
            # Moving average
            window_size = min(5, len(npis))
            moving_avg = np.convolve(npis, np.ones(window_size)/window_size, mode='valid')
            
            # Trend classification
            if slope > 0.5:
                trend = 'Increasing'
            elif slope < -0.5:
                trend = 'Decreasing'
            else:
                trend = 'Stable'
            
            # Variability
            npi_std = np.std(npis)
            npi_cv = npi_std / (np.mean(npis) + 1e-10)
            
        else:
            slope = 0
            trend = 'Insufficient Data'
            moving_avg = npis
            npi_std = 0
            npi_cv = 0
        
        return {
            'npi_time_series': npis.tolist(),
            'npi_categories': categories,
            'trend_slope': float(slope),
            'trend_direction': trend,
            'npi_mean': float(np.mean(npis)),
            'npi_std': float(npi_std),
            'npi_cv': float(npi_cv),
            'moving_average': moving_avg.tolist(),
            'npi_range': [float(np.min(npis)), float(np.max(npis))]
        }
    
    def correlate_with_mental_health(self, npi_results, mental_health_scores):
        """
        Correlate NPI with mental health assessment scores
        
        Args:
            npi_results: List of NPI scores
            mental_health_scores: List of corresponding mental health scores
        
        Returns:
            Correlation analysis
        """
        if len(npi_results) != len(mental_health_scores) or len(npi_results) < 3:
            return None
        
        npi_array = np.array(npi_results)
        mh_array = np.array(mental_health_scores)
        
        # Calculate correlations
        pearson_corr, pearson_p = stats.pearsonr(npi_array, mh_array)
        spearman_corr, spearman_p = stats.spearmanr(npi_array, mh_array)
        
        # Linear regression
        slope, intercept = np.polyfit(npi_array, mh_array, 1)
        predicted = slope * npi_array + intercept
        r_squared = 1 - np.sum((mh_array - predicted)**2) / np.sum((mh_array - np.mean(mh_array))**2)
        
        return {
            'pearson_correlation': float(pearson_corr),
            'pearson_p_value': float(pearson_p),
            'spearman_correlation': float(spearman_corr),
            'spearman_p_value': float(spearman_p),
            'r_squared': float(r_squared),
            'regression_slope': float(slope),
            'regression_intercept': float(intercept),
            'significant': pearson_p < 0.05
        }