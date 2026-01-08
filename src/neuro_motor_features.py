import numpy as np
from scipy import signal, stats
import cv2
from scipy.spatial import distance
from scipy.interpolate import interp1d

class NeuroMotorFeatureExtractor:
    """Extract neuro-motor features from handwriting"""
    
    def __init__(self):
        self.feature_cache = {}
        
    def extract_all_features(self, stroke_data):
        """Extract comprehensive neuro-motor features"""
        features = {}
        
        # 1. Temporal features
        features.update(self.extract_temporal_features(stroke_data))
        
        # 2. Pressure features
        features.update(self.extract_pressure_features(stroke_data))
        
        # 3. Tremor and jitter features
        features.update(self.extract_tremor_features(stroke_data))
        
        # 4. Slant and curvature features
        features.update(self.extract_curvature_features(stroke_data))
        
        # 5. Pause and hesitation features
        features.update(self.extract_pause_features(stroke_data))
        
        # 6. Acceleration and jerk features
        features.update(self.extract_acceleration_features(stroke_data))
        
        # 7. Symmetry and regularity features
        features.update(self.extract_symmetry_features(stroke_data))
        
        # 8. Composite neuro-motor index
        features.update(self.calculate_neuromotor_index(features))
        
        return features
    
    def extract_temporal_features(self, stroke_data):
        """Extract temporal features"""
        features = {}
        
        if 'timestamps' in stroke_data and len(stroke_data['timestamps']) > 1:
            timestamps = np.array(stroke_data['timestamps'])
            
            # Duration features
            total_duration = timestamps[-1] - timestamps[0]
            features['total_duration'] = total_duration
            
            # Velocity features
            if 'points' in stroke_data and len(stroke_data['points']) > 1:
                points = np.array(stroke_data['points'])
                time_diffs = np.diff(timestamps)
                
                # Calculate velocities
                displacements = np.linalg.norm(np.diff(points, axis=0), axis=1)
                velocities = displacements / (time_diffs + 1e-10)
                
                features['mean_velocity'] = np.mean(velocities)
                features['velocity_std'] = np.std(velocities)
                features['velocity_cv'] = features['velocity_std'] / (features['mean_velocity'] + 1e-10)
                features['max_velocity'] = np.max(velocities)
                features['min_velocity'] = np.min(velocities)
                
                # Velocity profile smoothness
                velocity_profile = interp1d(timestamps[:-1], velocities, kind='cubic')
                t_smooth = np.linspace(timestamps[0], timestamps[-2], 100)
                v_smooth = velocity_profile(t_smooth)
                
                # Velocity profile entropy
                hist, _ = np.histogram(v_smooth, bins=20, density=True)
                hist = hist[hist > 0]
                features['velocity_profile_entropy'] = -np.sum(hist * np.log(hist))
        
        return features
    
    def extract_pressure_features(self, stroke_data):
        """Extract pressure-related features"""
        features = {}
        
        if 'pressures' in stroke_data and len(stroke_data['pressures']) > 0:
            pressures = np.array(stroke_data['pressures'])
            
            # Basic pressure statistics
            features['mean_pressure'] = np.mean(pressures)
            features['pressure_std'] = np.std(pressures)
            features['pressure_cv'] = features['pressure_std'] / (features['mean_pressure'] + 1e-10)
            features['max_pressure'] = np.max(pressures)
            features['min_pressure'] = np.min(pressures)
            features['pressure_range'] = features['max_pressure'] - features['min_pressure']
            
            # Pressure variability
            pressure_diff = np.abs(np.diff(pressures))
            features['pressure_variability'] = np.mean(pressure_diff)
            
            # Pressure consistency
            features['pressure_consistency'] = 1 - features['pressure_cv']
            
            # Pressure histogram features
            hist, bin_edges = np.histogram(pressures, bins=10, density=True)
            features['pressure_histogram_skew'] = stats.skew(pressures)
            features['pressure_histogram_kurtosis'] = stats.kurtosis(pressures)
            
            # Pressure entropy
            hist = hist[hist > 0]
            features['pressure_entropy'] = -np.sum(hist * np.log(hist))
            
            # Pressure autocorrelation (for rhythmic patterns)
            if len(pressures) > 10:
                autocorr = np.correlate(pressures - np.mean(pressures), 
                                       pressures - np.mean(pressures), mode='full')
                autocorr = autocorr[len(autocorr)//2:]
                autocorr = autocorr / autocorr[0]
                features['pressure_autocorrelation_lag1'] = autocorr[1] if len(autocorr) > 1 else 0
        
        return features
    
    def extract_tremor_features(self, stroke_data):
        """Extract tremor and micro-jitter features"""
        features = {}
        
        if 'points' in stroke_data and len(stroke_data['points']) > 10:
            points = np.array(stroke_data['points'])
            
            # Calculate position differences
            position_diff = np.diff(points, axis=0)
            
            # High-frequency jitter (tremor)
            if len(position_diff) > 5:
                # Apply bandpass filter for tremor frequencies (4-12 Hz)
                if 'timestamps' in stroke_data:
                    timestamps = np.array(stroke_data['timestamps'])
                    sampling_rate = 1 / np.mean(np.diff(timestamps[:-1]))
                    
                    if sampling_rate > 10:  # Minimum sampling rate for tremor analysis
                        # Filter for tremor frequencies
                        b, a = signal.butter(4, [4/(sampling_rate/2), 12/(sampling_rate/2)], 
                                            btype='bandpass')
                        
                        # Filter x and y components separately
                        x_filtered = signal.filtfilt(b, a, position_diff[:, 0])
                        y_filtered = signal.filtfilt(b, a, position_diff[:, 1])
                        
                        # Calculate tremor magnitude
                        tremor_magnitude = np.sqrt(x_filtered**2 + y_filtered**2)
                        
                        features['tremor_mean'] = np.mean(tremor_magnitude)
                        features['tremor_std'] = np.std(tremor_magnitude)
                        features['tremor_power'] = np.sum(tremor_magnitude**2)
                        
                        # Tremor frequency analysis
                        if len(tremor_magnitude) > 20:
                            fft_result = np.fft.fft(tremor_magnitude)
                            frequencies = np.fft.fftfreq(len(tremor_magnitude), 
                                                       1/sampling_rate)
                            
                            # Find dominant tremor frequency
                            positive_freq = frequencies[:len(frequencies)//2]
                            positive_fft = np.abs(fft_result[:len(fft_result)//2])
                            
                            # Focus on tremor frequency band
                            tremor_mask = (positive_freq >= 4) & (positive_freq <= 12)
                            if np.any(tremor_mask):
                                tremor_freqs = positive_freq[tremor_mask]
                                tremor_power = positive_fft[tremor_mask]
                                
                                if len(tremor_freqs) > 0:
                                    dominant_idx = np.argmax(tremor_power)
                                    features['dominant_tremor_freq'] = tremor_freqs[dominant_idx]
                                    features['tremor_freq_power'] = tremor_power[dominant_idx]
            
            # Calculate jitter (instantaneous velocity changes)
            if len(points) > 2:
                velocities = np.linalg.norm(np.diff(points, axis=0), axis=1)
                acceleration = np.diff(velocities)
                jerk = np.diff(acceleration)
                
                features['mean_jerk'] = np.mean(np.abs(jerk)) if len(jerk) > 0 else 0
                features['jerk_std'] = np.std(jerk) if len(jerk) > 0 else 0
                
                # Jitter index (normalized jerk)
                if features['mean_velocity'] > 0:
                    features['jitter_index'] = features['mean_jerk'] / features['mean_velocity']
                else:
                    features['jitter_index'] = 0
        
        return features
    
    def extract_curvature_features(self, stroke_data):
        """Extract curvature and slant instability features"""
        features = {}
        
        if 'points' in stroke_data and len(stroke_data['points']) > 10:
            points = np.array(stroke_data['points'])
            
            # Calculate curvature
            dx = np.gradient(points[:, 0])
            dy = np.gradient(points[:, 1])
            d2x = np.gradient(dx)
            d2y = np.gradient(dy)
            
            curvature = np.abs(d2x * dy - dx * d2y) / (dx**2 + dy**2 + 1e-10)**1.5
            
            features['mean_curvature'] = np.mean(curvature)
            features['curvature_std'] = np.std(curvature)
            features['max_curvature'] = np.max(curvature)
            features['curvature_entropy'] = stats.entropy(np.histogram(curvature, bins=20)[0])
            
            # Slant analysis
            if len(points) > 2:
                segments = points[1:] - points[:-1]
                angles = np.arctan2(segments[:, 1], segments[:, 0])
                angles_deg = np.degrees(angles)
                
                features['slant_mean'] = np.mean(angles_deg)
                features['slant_std'] = np.std(angles_deg)
                features['slant_instability'] = features['slant_std'] / (np.abs(features['slant_mean']) + 1e-10)
                
                # Slant consistency (how much slant varies)
                slant_diff = np.abs(np.diff(angles_deg))
                features['slant_consistency'] = 1 - (np.mean(slant_diff) / 180)
            
            # Curvature instability (changes in curvature)
            curvature_diff = np.abs(np.diff(curvature))
            features['curvature_instability'] = np.mean(curvature_diff)
            
            # Number of inflection points
            curvature_sign = np.sign(curvature)
            inflection_points = np.sum(np.abs(np.diff(curvature_sign)) > 0)
            features['inflection_points'] = inflection_points
            features['inflection_density'] = inflection_points / len(points)
        
        return features
    
    def extract_pause_features(self, stroke_data):
        """Extract pause and hesitation patterns"""
        features = {}
        
        if 'timestamps' in stroke_data and len(stroke_data['timestamps']) > 2:
            timestamps = np.array(stroke_data['timestamps'])
            
            if 'points' in stroke_data and len(stroke_data['points']) > 1:
                points = np.array(stroke_data['points'])
                
                # Calculate movement distances between timestamps
                time_intervals = np.diff(timestamps)
                movements = np.linalg.norm(np.diff(points, axis=0), axis=1)
                
                # Calculate instantaneous speeds
                speeds = movements / (time_intervals + 1e-10)
                
                # Identify pauses (speed below threshold)
                pause_threshold = np.percentile(speeds, 10)  # Bottom 10% as pause
                pause_mask = speeds < pause_threshold
                
                features['pause_count'] = np.sum(pause_mask)
                features['pause_ratio'] = np.sum(pause_mask) / len(speeds)
                features['total_pause_duration'] = np.sum(time_intervals[pause_mask])
                features['mean_pause_duration'] = np.mean(time_intervals[pause_mask]) if features['pause_count'] > 0 else 0
                
                # Hesitation patterns (frequent short pauses)
                if features['pause_count'] > 1:
                    # Find consecutive pauses
                    pause_groups = []
                    current_group = []
                    
                    for i, is_pause in enumerate(pause_mask):
                        if is_pause:
                            current_group.append(i)
                        elif current_group:
                            pause_groups.append(current_group)
                            current_group = []
                    
                    if current_group:
                        pause_groups.append(current_group)
                    
                    # Analyze pause patterns
                    group_durations = []
                    for group in pause_groups:
                        if len(group) > 0:
                            duration = timestamps[group[-1]+1] - timestamps[group[0]]
                            group_durations.append(duration)
                    
                    if group_durations:
                        features['hesitation_pattern_std'] = np.std(group_durations)
                        features['hesitation_frequency'] = len(group_durations) / timestamps[-1]
                
                # Speed variability as hesitation indicator
                speed_cv = np.std(speeds) / (np.mean(speeds) + 1e-10)
                features['hesitation_index'] = speed_cv
        
        return features
    
    def extract_acceleration_features(self, stroke_data):
        """Extract acceleration and jerk patterns"""
        features = {}
        
        if 'points' in stroke_data and len(stroke_data['points']) > 3:
            points = np.array(stroke_data['points'])
            
            if 'timestamps' in stroke_data and len(stroke_data['timestamps']) > 3:
                timestamps = np.array(stroke_data['timestamps'])
                
                # Calculate velocities
                time_intervals = np.diff(timestamps)
                displacements = np.linalg.norm(np.diff(points, axis=0), axis=1)
                velocities = displacements / (time_intervals + 1e-10)
                
                # Calculate accelerations
                if len(velocities) > 1:
                    time_intervals_acc = time_intervals[:-1] + time_intervals[1:]
                    time_intervals_acc = time_intervals_acc / 2
                    accelerations = np.diff(velocities) / (time_intervals_acc + 1e-10)
                    
                    features['mean_acceleration'] = np.mean(np.abs(accelerations))
                    features['acceleration_std'] = np.std(accelerations)
                    features['max_acceleration'] = np.max(np.abs(accelerations))
                    features['acceleration_cv'] = features['acceleration_std'] / (features['mean_acceleration'] + 1e-10)
                    
                    # Jerk (rate of change of acceleration)
                    if len(accelerations) > 1:
                        jerk = np.diff(accelerations) / (time_intervals[:-2] + 1e-10)
                        features['mean_jerk'] = np.mean(np.abs(jerk))
                        features['jerk_std'] = np.std(jerk)
                        features['jerk_power'] = np.sum(jerk**2)
                        
                        # Smoothness metric
                        features['movement_smoothness'] = 1 / (1 + features['jerk_power'])
                    
                    # Acceleration profile regularity
                    acceleration_profile = interp1d(timestamps[1:-1], accelerations, kind='cubic')
                    t_smooth = np.linspace(timestamps[1], timestamps[-2], 100)
                    a_smooth = acceleration_profile(t_smooth)
                    
                    # Calculate regularity using autocorrelation
                    if len(a_smooth) > 10:
                        autocorr = np.correlate(a_smooth - np.mean(a_smooth), 
                                               a_smooth - np.mean(a_smooth), mode='full')
                        autocorr = autocorr[len(autocorr)//2:]
                        autocorr = autocorr / autocorr[0]
                        
                        # Regularity index (how periodic the acceleration is)
                        regularity = np.mean(np.abs(autocorr[1:6]))  # First 5 lags
                        features['acceleration_regularity'] = regularity
        
        return features
    
    def extract_symmetry_features(self, stroke_data):
        """Extract symmetry and regularity features"""
        features = {}
        
        if 'strokes' in stroke_data and len(stroke_data['strokes']) > 1:
            strokes = stroke_data['strokes']
            
            # Analyze stroke-to-stroke consistency
            stroke_lengths = []
            stroke_durations = []
            stroke_speeds = []
            
            for stroke in strokes:
                if 'points' in stroke and len(stroke['points']) > 1:
                    points = np.array(stroke['points'])
                    
                    # Stroke length
                    length = np.sum(np.linalg.norm(np.diff(points, axis=0), axis=1))
                    stroke_lengths.append(length)
                    
                    # Stroke duration
                    if 'timestamps' in stroke and len(stroke['timestamps']) > 1:
                        duration = stroke['timestamps'][-1] - stroke['timestamps'][0]
                        stroke_durations.append(duration)
                        
                        # Average speed
                        if duration > 0:
                            stroke_speeds.append(length / duration)
            
            if stroke_lengths:
                features['stroke_length_mean'] = np.mean(stroke_lengths)
                features['stroke_length_std'] = np.std(stroke_lengths)
                features['stroke_length_cv'] = features['stroke_length_std'] / (features['stroke_length_mean'] + 1e-10)
            
            if stroke_durations:
                features['stroke_duration_mean'] = np.mean(stroke_durations)
                features['stroke_duration_std'] = np.std(stroke_durations)
                features['stroke_duration_cv'] = features['stroke_duration_std'] / (features['stroke_duration_mean'] + 1e-10)
            
            if stroke_speeds:
                features['stroke_speed_mean'] = np.mean(stroke_speeds)
                features['stroke_speed_std'] = np.std(stroke_speeds)
                features['stroke_speed_cv'] = features['stroke_speed_std'] / (features['stroke_speed_mean'] + 1e-10)
                
                # Consistency index (inverse of CV)
                features['stroke_consistency'] = 1 - min(features['stroke_speed_cv'], 1)
            
            # Direction symmetry analysis
            if 'points' in stroke_data and len(stroke_data['points']) > 10:
                points = np.array(stroke_data['points'])
                
                # Calculate principal directions
                segments = points[1:] - points[:-1]
                if len(segments) > 0:
                    angles = np.arctan2(segments[:, 1], segments[:, 0])
                    
                    # Convert to 0-180 degree range for symmetry analysis
                    angles_deg = np.degrees(angles) % 180
                    
                    # Calculate angular symmetry
                    hist, bins = np.histogram(angles_deg, bins=18, range=(0, 180))
                    hist_norm = hist / np.sum(hist)
                    
                    # Symmetry index (how evenly distributed are directions)
                    symmetry_index = 1 - np.std(hist_norm) / np.mean(hist_norm)
                    features['directional_symmetry'] = symmetry_index
        
        return features
    
    def calculate_neuromotor_index(self, features):
        """Calculate composite neuro-motor index"""
        neuromotor_index = {}
        
        # Weights for different feature categories
        weights = {
            'tremor': 0.25,
            'pressure': 0.20,
            'curvature': 0.15,
            'pause': 0.15,
            'acceleration': 0.10,
            'symmetry': 0.10,
            'temporal': 0.05
        }
        
        # Calculate sub-indices
        sub_indices = {}
        
        # Tremor index
        tremor_score = 0
        if 'tremor_mean' in features:
            tremor_score = min(features['tremor_mean'] * 10, 1)
        sub_indices['tremor_index'] = tremor_score
        
        # Pressure instability index
        pressure_score = 0
        if 'pressure_cv' in features:
            pressure_score = min(features['pressure_cv'] * 2, 1)
        sub_indices['pressure_instability'] = pressure_score
        
        # Curvature instability index
        curvature_score = 0
        if 'curvature_instability' in features:
            curvature_score = min(features['curvature_instability'] * 5, 1)
        sub_indices['curvature_instability'] = curvature_score
        
        # Pause index
        pause_score = 0
        if 'pause_ratio' in features:
            pause_score = min(features['pause_ratio'] * 3, 1)
        sub_indices['pause_index'] = pause_score
        
        # Acceleration irregularity
        acceleration_score = 0
        if 'acceleration_cv' in features:
            acceleration_score = min(features['acceleration_cv'] * 2, 1)
        sub_indices['acceleration_irregularity'] = acceleration_score
        
        # Symmetry index (inverted)
        symmetry_score = 0
        if 'stroke_consistency' in features:
            symmetry_score = 1 - min(features['stroke_consistency'], 1)
        sub_indices['asymmetry_index'] = symmetry_score
        
        # Temporal variability
        temporal_score = 0
        if 'velocity_cv' in features:
            temporal_score = min(features['velocity_cv'] * 2, 1)
        sub_indices['temporal_variability'] = temporal_score
        
        # Calculate weighted neuro-motor index (0-100 scale)
        neuromotor_score = 0
        total_weight = 0
        
        for sub_index, score in sub_indices.items():
            category = sub_index.split('_')[0]
            if category in weights:
                neuromotor_score += score * weights[category]
                total_weight += weights[category]
        
        if total_weight > 0:
            neuromotor_score = (neuromotor_score / total_weight) * 100
        
        neuromotor_index['neuromotor_index'] = neuromotor_score
        neuromotor_index['neuromotor_sub_indices'] = sub_indices
        
        # Risk classification based on neuromotor index
        if neuromotor_score < 30:
            neuromotor_index['neuromotor_risk'] = 'Low'
        elif neuromotor_score < 60:
            neuromotor_index['neuromotor_risk'] = 'Moderate'
        else:
            neuromotor_index['neuromotor_risk'] = 'High'
        
        return neuromotor_index