import numpy as np
import cv2
from scipy import signal, stats, ndimage, interpolate
import networkx as nx
from sklearn.decomposition import PCA
from scipy.spatial import distance, ConvexHull
import pywt
import warnings
warnings.filterwarnings('ignore')

class HandwritingFeatureExtractor:
    """Complete feature extraction for handwriting analysis"""
    
    def __init__(self):
        self.feature_categories = [
            'spatial',
            'temporal',
            'spectral',
            'statistical',
            'graph',
            'fractal',
            'entropy'
        ]
    
    def extract_all_features(self, image_data, temporal_data=None):
        """Extract comprehensive set of features"""
        features = {}
        
        # 1. Basic image features
        features.update(self.extract_basic_features(image_data))
        
        # 2. Spatial features
        features.update(self.extract_spatial_features(image_data))
        
        # 3. Statistical features
        features.update(self.extract_statistical_features(image_data))
        
        # 4. Graph-based features
        features.update(self.extract_graph_features(image_data))
        
        # 5. Fractal features
        features.update(self.extract_fractal_features(image_data))
        
        # 6. Entropy features
        features.update(self.extract_entropy_features(image_data))
        
        # 7. Spectral features
        features.update(self.extract_spectral_features(image_data))
        
        # 8. Temporal features (if available)
        if temporal_data is not None:
            features.update(self.extract_temporal_features(temporal_data))
        
        # 9. Combined features
        features.update(self.extract_combined_features(features))
        
        return features
    
    def extract_basic_features(self, image):
        """Extract basic image features"""
        features = {}
        
        # Convert to binary if needed
        if image.max() > 1:
            _, binary = cv2.threshold(image, 127, 1, cv2.THRESH_BINARY)
        else:
            binary = (image > 0).astype(np.uint8)
        
        # Basic statistics
        features['ink_density'] = np.sum(binary) / binary.size
        features['aspect_ratio'] = image.shape[1] / image.shape[0]
        
        # Image moments
        moments = cv2.moments(binary)
        if moments['m00'] != 0:
            features['centroid_x'] = moments['m10'] / moments['m00']
            features['centroid_y'] = moments['m01'] / moments['m00']
            
            # Central moments
            features['mu20'] = moments['mu20'] / moments['m00']
            features['mu02'] = moments['mu02'] / moments['m00']
            features['mu11'] = moments['mu11'] / moments['m00']
        
        # Hu moments (invariant)
        hu_moments = cv2.HuMoments(moments).flatten()
        for i in range(7):
            if hu_moments[i] != 0:
                features[f'hu_{i}'] = -np.sign(hu_moments[i]) * np.log10(np.abs(hu_moments[i]))
            else:
                features[f'hu_{i}'] = 0
        
        # Zernike moments
        try:
            zernike_moments = self.calculate_zernike_moments(binary, degree=8)
            for i, moment in enumerate(zernike_moments[:10]):
                features[f'zernike_{i}'] = np.abs(moment)
        except:
            pass
        
        return features
    
    def extract_spatial_features(self, image):
        """Extract spatial layout features"""
        features = {}
        
        if image.max() > 1:
            _, binary = cv2.threshold(image, 127, 1, cv2.THRESH_BINARY)
        else:
            binary = (image > 0).astype(np.uint8)
        
        # Find contours
        contours, _ = cv2.findContours(
            binary.astype(np.uint8), 
            cv2.RETR_EXTERNAL, 
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        if len(contours) > 0:
            # Stroke features
            stroke_lengths = []
            stroke_areas = []
            stroke_perimeters = []
            stroke_circularities = []
            
            for contour in contours:
                # Area
                area = cv2.contourArea(contour)
                stroke_areas.append(area)
                
                # Perimeter
                perimeter = cv2.arcLength(contour, True)
                stroke_perimeters.append(perimeter)
                
                # Length (approximate)
                if len(contour) > 1:
                    points = contour.squeeze()
                    if len(points.shape) == 2:
                        length = np.sum(np.linalg.norm(points[1:] - points[:-1], axis=1))
                        stroke_lengths.append(length)
                
                # Circularity
                if perimeter > 0:
                    circularity = (4 * np.pi * area) / (perimeter ** 2)
                    stroke_circularities.append(circularity)
            
            # Aggregate stroke features
            if stroke_lengths:
                features['num_strokes'] = len(contours)
                features['stroke_length_mean'] = np.mean(stroke_lengths)
                features['stroke_length_std'] = np.std(stroke_lengths)
                features['stroke_length_cv'] = features['stroke_length_std'] / features['stroke_length_mean']
            
            if stroke_areas:
                features['stroke_area_mean'] = np.mean(stroke_areas)
                features['stroke_area_std'] = np.std(stroke_areas)
                features['total_ink_area'] = np.sum(stroke_areas)
            
            if stroke_perimeters:
                features['stroke_perimeter_mean'] = np.mean(stroke_perimeters)
                features['stroke_perimeter_std'] = np.std(stroke_perimeters)
            
            if stroke_circularities:
                features['circularity_mean'] = np.mean(stroke_circularities)
                features['circularity_std'] = np.std(stroke_circularities)
        
        # Slant analysis
        slant_features = self.analyze_slant(binary)
        features.update(slant_features)
        
        # Curvature analysis
        curvature_features = self.analyze_curvature(binary)
        features.update(curvature_features)
        
        # Line spacing and alignment
        spacing_features = self.analyze_spacing(binary)
        features.update(spacing_features)
        
        # Margin analysis
        margin_features = self.analyze_margins(binary)
        features.update(margin_features)
        
        return features
    
    def analyze_slant(self, binary_image):
        """Analyze handwriting slant"""
        features = {}
        
        # Use Hough transform to detect line angles
        edges = cv2.Canny((binary_image * 255).astype(np.uint8), 50, 150)
        lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=50)
        
        if lines is not None:
            angles = []
            for rho, theta in lines[:, 0]:
                angle = np.degrees(theta)
                # Filter for reasonable slant angles
                if 0 < angle < 180:
                    # Convert to writing slant (-90 to 90)
                    if angle > 90:
                        angle = angle - 180
                    angles.append(angle)
            
            if angles:
                features['slant_mean'] = np.mean(angles)
                features['slant_std'] = np.std(angles)
                features['slant_abs_mean'] = np.mean(np.abs(angles))
                features['slant_range'] = np.max(angles) - np.min(angles)
                
                # Slant direction
                right_slant = np.sum(np.array(angles) > 0)
                left_slant = np.sum(np.array(angles) < 0)
                features['slant_direction_ratio'] = right_slant / max(1, left_slant)
        
        return features
    
    def analyze_curvature(self, binary_image):
        """Analyze stroke curvature"""
        features = {}
        
        # Skeletonize for curvature analysis
        skeleton = self.zhang_suen_skeletonization(binary_image)
        
        # Find contours of skeleton
        contours, _ = cv2.findContours(
            skeleton.astype(np.uint8), 
            cv2.RETR_EXTERNAL, 
            cv2.CHAIN_APPROX_NONE
        )
        
        if contours and len(contours[0]) > 10:
            contour = contours[0][:, 0, :].astype(np.float32)
            
            # Smooth contour
            contour = cv2.GaussianBlur(contour, (5, 5), 0)
            
            # Calculate curvature using derivative
            dx = np.gradient(contour[:, 0])
            dy = np.gradient(contour[:, 1])
            d2x = np.gradient(dx)
            d2y = np.gradient(dy)
            
            curvature = np.abs(d2x * dy - dx * d2y) / (dx**2 + dy**2 + 1e-10)**1.5
            
            # Curvature features
            features['curvature_mean'] = np.mean(curvature)
            features['curvature_std'] = np.std(curvature)
            features['curvature_max'] = np.max(curvature)
            features['curvature_min'] = np.min(curvature)
            features['curvature_range'] = features['curvature_max'] - features['curvature_min']
            
            # Number of inflection points
            inflection_points = np.where(np.diff(np.sign(curvature)))[0]
            features['num_inflection_points'] = len(inflection_points)
            
            # Curvature entropy
            hist, _ = np.histogram(curvature, bins=20, density=True)
            hist = hist[hist > 0]
            features['curvature_entropy'] = -np.sum(hist * np.log(hist))
        
        return features
    
    def analyze_spacing(self, binary_image):
        """Analyze spacing between lines and words"""
        features = {}
        
        # Projection profiles
        horizontal_profile = np.sum(binary_image, axis=1)
        vertical_profile = np.sum(binary_image, axis=0)
        
        # Find peaks in projection profiles
        from scipy.signal import find_peaks
        
        # Horizontal spacing (line spacing)
        h_peaks, h_properties = find_peaks(horizontal_profile, 
                                          height=np.mean(horizontal_profile),
                                          distance=10)
        
        if len(h_peaks) > 1:
            line_spacing = np.diff(h_peaks)
            features['line_spacing_mean'] = np.mean(line_spacing)
            features['line_spacing_std'] = np.std(line_spacing)
            features['line_spacing_cv'] = features['line_spacing_std'] / features['line_spacing_mean']
            features['num_lines'] = len(h_peaks)
        
        # Vertical spacing (word spacing)
        v_peaks, v_properties = find_peaks(vertical_profile,
                                          height=np.mean(vertical_profile),
                                          distance=5)
        
        if len(v_peaks) > 1:
            word_spacing = np.diff(v_peaks)
            features['word_spacing_mean'] = np.mean(word_spacing)
            features['word_spacing_std'] = np.std(word_spacing)
            features['num_words'] = len(v_peaks)
        
        # Baseline analysis
        if 'num_lines' in features and features['num_lines'] > 0:
            baseline_angles = []
            for peak in h_peaks:
                # Extract line region
                line_region = binary_image[max(0, peak-5):min(binary_image.shape[0], peak+5), :]
                if np.sum(line_region) > 0:
                    # Find baseline using Hough transform on this region
                    line_edges = cv2.Canny((line_region * 255).astype(np.uint8), 50, 150)
                    line_lines = cv2.HoughLines(line_edges, 1, np.pi/180, threshold=20)
                    if line_lines is not None:
                        for rho, theta in line_lines[:, 0]:
                            angle = np.degrees(theta)
                            if 0 < angle < 180:
                                baseline_angles.append(angle)
            
            if baseline_angles:
                features['baseline_angle_mean'] = np.mean(baseline_angles)
                features['baseline_angle_std'] = np.std(baseline_angles)
        
        return features
    
    def analyze_margins(self, binary_image):
        """Analyze page margins and writing positioning"""
        features = {}
        
        h, w = binary_image.shape
        
        # Find bounding box of writing
        rows = np.any(binary_image, axis=1)
        cols = np.any(binary_image, axis=0)
        
        if np.any(rows) and np.any(cols):
            y_min, y_max = np.where(rows)[0][[0, -1]]
            x_min, x_max = np.where(cols)[0][[0, -1]]
            
            # Calculate margins
            features['left_margin'] = x_min / w
            features['right_margin'] = (w - x_max) / w
            features['top_margin'] = y_min / h
            features['bottom_margin'] = (h - y_max) / h
            
            # Writing position
            features['writing_center_x'] = (x_min + x_max) / (2 * w)
            features['writing_center_y'] = (y_min + y_max) / (2 * h)
            
            # Writing area ratio
            writing_area = (x_max - x_min) * (y_max - y_min)
            total_area = w * h
            features['writing_area_ratio'] = writing_area / total_area
        
        return features
    
    def extract_statistical_features(self, image):
        """Extract statistical features"""
        features = {}
        
        if image.max() > 1:
            _, binary = cv2.threshold(image, 127, 1, cv2.THRESH_BINARY)
        else:
            binary = (image > 0).astype(np.uint8)
        
        # Intensity statistics (if grayscale)
        if len(image.shape) == 2 and image.max() > 1:
            features['intensity_mean'] = np.mean(image)
            features['intensity_std'] = np.std(image)
            features['intensity_skew'] = stats.skew(image.flatten())
            features['intensity_kurtosis'] = stats.kurtosis(image.flatten())
        
        # Binary image statistics
        ink_pixels = np.sum(binary)
        total_pixels = binary.size
        
        features['ink_density'] = ink_pixels / total_pixels
        
        # Spatial distribution statistics
        if ink_pixels > 0:
            # Center of mass
            y_indices, x_indices = np.where(binary > 0)
            features['com_x'] = np.mean(x_indices) / binary.shape[1]
            features['com_y'] = np.mean(y_indices) / binary.shape[0]
            
            # Spatial variance
            features['spatial_var_x'] = np.var(x_indices) / (binary.shape[1] ** 2)
            features['spatial_var_y'] = np.var(y_indices) / (binary.shape[0] ** 2)
            
            # Radial distribution
            distances = np.sqrt((x_indices - features['com_x'] * binary.shape[1]) ** 2 +
                              (y_indices - features['com_y'] * binary.shape[0]) ** 2)
            features['mean_radial_distance'] = np.mean(distances)
            features['radial_distance_std'] = np.std(distances)
        
        # Run length statistics
        run_length_features = self.calculate_run_length_features(binary)
        features.update(run_length_features)
        
        return features
    
    def calculate_run_length_features(self, binary_image):
        """Calculate run-length encoding features"""
        features = {}
        
        # Horizontal runs
        horizontal_runs = []
        for row in binary_image:
            runs = np.diff(np.where(np.concatenate(([0], row, [0])))[0])[::2]
            horizontal_runs.extend(runs)
        
        # Vertical runs
        vertical_runs = []
        for col in binary_image.T:
            runs = np.diff(np.where(np.concatenate(([0], col, [0])))[0])[::2]
            vertical_runs.extend(runs)
        
        if horizontal_runs:
            features['horiz_run_mean'] = np.mean(horizontal_runs)
            features['horiz_run_std'] = np.std(horizontal_runs)
            features['horiz_run_max'] = np.max(horizontal_runs)
            features['horiz_run_entropy'] = stats.entropy(np.histogram(horizontal_runs, bins=20)[0])
        
        if vertical_runs:
            features['vert_run_mean'] = np.mean(vertical_runs)
            features['vert_run_std'] = np.std(vertical_runs)
            features['vert_run_max'] = np.max(vertical_runs)
            features['vert_run_entropy'] = stats.entropy(np.histogram(vertical_runs, bins=20)[0])
        
        return features
    
    def extract_graph_features(self, image):
        """Extract graph-based features from handwriting skeleton"""
        features = {}
        
        if image.max() > 1:
            _, binary = cv2.threshold(image, 127, 1, cv2.THRESH_BINARY)
        else:
            binary = (image > 0).astype(np.uint8)
        
        # Skeletonize
        skeleton = self.zhang_suen_skeletonization(binary)
        
        # Create graph from skeleton
        G = self.skeleton_to_graph(skeleton)
        
        if G.number_of_nodes() > 0:
            # Basic graph features
            features['graph_num_nodes'] = G.number_of_nodes()
            features['graph_num_edges'] = G.number_of_edges()
            features['graph_density'] = nx.density(G)
            
            # Degree statistics
            degrees = [d for n, d in G.degree()]
            features['graph_degree_mean'] = np.mean(degrees)
            features['graph_degree_std'] = np.std(degrees)
            features['graph_max_degree'] = np.max(degrees)
            
            # Path length statistics
            if nx.is_connected(G):
                features['graph_avg_path_length'] = nx.average_shortest_path_length(G)
                features['graph_diameter'] = nx.diameter(G)
            else:
                # For disconnected graphs, analyze largest component
                largest_cc = max(nx.connected_components(G), key=len)
                if len(largest_cc) > 1:
                    subgraph = G.subgraph(largest_cc)
                    features['graph_avg_path_length'] = nx.average_shortest_path_length(subgraph)
                    features['graph_diameter'] = nx.diameter(subgraph)
            
            # Clustering coefficient
            features['graph_avg_clustering'] = nx.average_clustering(G)
            
            # Centrality measures
            if G.number_of_nodes() > 1:
                betweenness = nx.betweenness_centrality(G)
                features['graph_betweenness_mean'] = np.mean(list(betweenness.values()))
                features['graph_betweenness_std'] = np.std(list(betweenness.values()))
        
        return features
    
    def skeleton_to_graph(self, skeleton):
        """Convert skeleton to graph representation"""
        G = nx.Graph()
        
        # Find all skeleton points
        points = np.column_stack(np.where(skeleton > 0))
        
        if len(points) == 0:
            return G
        
        # Create nodes
        for i, (y, x) in enumerate(points):
            G.add_node(i, pos=(x, y))
        
        # Connect neighboring points (8-connectivity)
        for i, (y1, x1) in enumerate(points):
            for j, (y2, x2) in enumerate(points[i+1:], i+1):
                if abs(x1 - x2) <= 1 and abs(y1 - y2) <= 1:
                    distance = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
                    G.add_edge(i, j, weight=distance)
        
        return G
    
    def extract_fractal_features(self, image):
        """Extract fractal dimension features"""
        features = {}
        
        if image.max() > 1:
            _, binary = cv2.threshold(image, 127, 1, cv2.THRESH_BINARY)
        else:
            binary = (image > 0).astype(np.uint8)
        
        # Box-counting dimension
        features['fractal_dimension'] = self.calculate_box_counting_dimension(binary)
        
        # Hurst exponent (for time series if available)
        # Convert 2D image to 1D signal for Hurst calculation
        if binary.shape[0] > 1:
            row_sums = np.sum(binary, axis=1)
            if np.any(row_sums > 0):
                try:
                    features['hurst_exponent'] = self.calculate_hurst_exponent(row_sums)
                except:
                    features['hurst_exponent'] = 0.5
        
        # Lacunarity (measure of gappiness)
        features['lacunarity'] = self.calculate_lacunarity(binary)
        
        return features
    
    def calculate_box_counting_dimension(self, binary_image, box_sizes=None):
        """Calculate fractal dimension using box-counting method"""
        if box_sizes is None:
            box_sizes = [2, 4, 8, 16, 32, 64]
        
        counts = []
        
        for box_size in box_sizes:
            if box_size < min(binary_image.shape):
                # Count boxes containing ink
                h, w = binary_image.shape
                count = 0
                
                for i in range(0, h, box_size):
                    for j in range(0, w, box_size):
                        box = binary_image[i:min(i+box_size, h), j:min(j+box_size, w)]
                        if np.any(box > 0):
                            count += 1
                
                counts.append(count)
        
        if len(counts) >= 2:
            # Linear fit in log-log space
            log_sizes = np.log([box_size for box_size in box_sizes[:len(counts)]])
            log_counts = np.log(counts)
            
            # Calculate slope (negative of fractal dimension)
            slope, intercept = np.polyfit(log_sizes, log_counts, 1)
            return -slope
        
        return 0.0
    
    def calculate_hurst_exponent(self, time_series):
        """Calculate Hurst exponent using R/S analysis"""
        n = len(time_series)
        if n < 10:
            return 0.5
        
        # Calculate R/S for different lags
        lags = np.arange(2, n//2)
        rs_values = []
        
        for lag in lags:
            # Split into chunks
            chunks = [time_series[i:i+lag] for i in range(0, n, lag) if len(time_series[i:i+lag]) == lag]
            
            if len(chunks) > 1:
                chunk_rs = []
                for chunk in chunks:
                    # Calculate mean and cumulative deviations
                    mean_chunk = np.mean(chunk)
                    deviations = chunk - mean_chunk
                    cumulative = np.cumsum(deviations)
                    
                    # Range and standard deviation
                    R = np.max(cumulative) - np.min(cumulative)
                    S = np.std(chunk)
                    
                    if S > 0:
                        chunk_rs.append(R / S)
                
                if chunk_rs:
                    rs_values.append(np.mean(chunk_rs))
        
        if len(rs_values) >= 2:
            log_lags = np.log(lags[:len(rs_values)])
            log_rs = np.log(rs_values)
            
            # Fit line
            slope, _ = np.polyfit(log_lags, log_rs, 1)
            return slope
        
        return 0.5
    
    def calculate_lacunarity(self, binary_image, box_sizes=None):
        """Calculate lacunarity (measure of heterogeneity)"""
        if box_sizes is None:
            box_sizes = [2, 4, 8, 16, 32]
        
        lacunarities = []
        
        for box_size in box_sizes:
            if box_size < min(binary_image.shape):
                h, w = binary_image.shape
                masses = []
                
                for i in range(0, h, box_size):
                    for j in range(0, w, box_size):
                        box = binary_image[i:min(i+box_size, h), j:min(j+box_size, w)]
                        mass = np.sum(box)
                        masses.append(mass)
                
                if masses:
                    mean_mass = np.mean(masses)
                    std_mass = np.std(masses)
                    if mean_mass > 0:
                        lacunarity = (std_mass / mean_mass) ** 2
                        lacunarities.append(lacunarity)
        
        return np.mean(lacunarities) if lacunarities else 0.0
    
    def extract_entropy_features(self, image):
        """Extract entropy-based features"""
        features = {}
        
        if image.max() > 1:
            _, binary = cv2.threshold(image, 127, 1, cv2.THRESH_BINARY)
        else:
            binary = (image > 0).astype(np.uint8)
        
        # Shannon entropy
        hist, _ = np.histogram(binary.flatten(), bins=2, density=True)
        hist = hist[hist > 0]
        features['shannon_entropy'] = -np.sum(hist * np.log2(hist))
        
        # Spatial entropy (local entropy)
        from skimage.filters.rank import entropy
        from skimage.morphology import disk
        
        if len(image.shape) == 2 and image.max() > 1:
            local_entropy = entropy(image.astype(np.uint8), disk(5))
            features['local_entropy_mean'] = np.mean(local_entropy)
            features['local_entropy_std'] = np.std(local_entropy)
        
        # Approximate entropy (for complexity)
        if binary.shape[0] > 10:
            row_sums = np.sum(binary, axis=1)
            features['approx_entropy'] = self.calculate_approximate_entropy(row_sums, m=2, r=0.2)
        
        # Sample entropy
        features['sample_entropy'] = self.calculate_sample_entropy(binary.flatten())
        
        return features
    
    def calculate_approximate_entropy(self, time_series, m=2, r=0.2):
        """Calculate approximate entropy"""
        n = len(time_series)
        if n < m + 1:
            return 0
        
        # Normalize time series
        time_series = (time_series - np.mean(time_series)) / np.std(time_series)
        
        def _phi(m):
            patterns = []
            for i in range(n - m + 1):
                pattern = tuple(time_series[i:i+m])
                patterns.append(pattern)
            
            counts = {}
            for pattern in patterns:
                counts[pattern] = counts.get(pattern, 0) + 1
            
            # Calculate probability
            total = len(patterns)
            probabilities = [count / total for count in counts.values()]
            
            # Calculate phi
            phi = 0
            for p in probabilities:
                if p > 0:
                    phi += p * np.log(p)
            
            return phi
        
        phi_m = _phi(m)
        phi_m1 = _phi(m + 1)
        
        return phi_m - phi_m1
    
    def calculate_sample_entropy(self, time_series, m=2, r=0.2):
        """Calculate sample entropy"""
        n = len(time_series)
        if n < m + 1:
            return 0
        
        # Normalize
        std = np.std(time_series)
        if std == 0:
            return 0
        time_series = (time_series - np.mean(time_series)) / std
        
        def _count_matches(m):
            count = 0
            patterns = []
            
            for i in range(n - m + 1):
                patterns.append(time_series[i:i+m])
            
            for i in range(len(patterns)):
                for j in range(i+1, len(patterns)):
                    if np.max(np.abs(patterns[i] - patterns[j])) <= r:
                        count += 1
            
            return count
        
        A = _count_matches(m + 1)
        B = _count_matches(m)
        
        if B == 0:
            return 0
        
        return -np.log(A / B)
    
    def extract_spectral_features(self, image):
        """Extract spectral (frequency domain) features"""
        features = {}
        
        if len(image.shape) == 2:
            # 2D Fourier Transform
            f_transform = np.fft.fft2(image)
            f_shift = np.fft.fftshift(f_transform)
            magnitude_spectrum = np.abs(f_shift)
            
            # Spectral features
            features['spectral_energy'] = np.sum(magnitude_spectrum ** 2)
            features['spectral_centroid'] = np.sum(magnitude_spectrum * np.arange(magnitude_spectrum.size)) / features['spectral_energy']
            features['spectral_spread'] = np.sqrt(np.sum(magnitude_spectrum * (np.arange(magnitude_spectrum.size) - features['spectral_centroid']) ** 2) / features['spectral_energy'])
            
            # Wavelet features
            try:
                wavelet_features = self.extract_wavelet_features(image)
                features.update(wavelet_features)
            except:
                pass
        
        return features
    
    def extract_wavelet_features(self, image):
        """Extract wavelet transform features"""
        features = {}
        
        # Perform 2D wavelet decomposition
        coeffs = pywt.wavedec2(image, 'db1', level=3)
        
        # Extract coefficients from each level
        for i, coeff in enumerate(coeffs):
            if i == 0:
                # Approximation coefficients
                features[f'wavelet_approx_mean'] = np.mean(coeff)
                features[f'wavelet_approx_std'] = np.std(coeff)
                features[f'wavelet_approx_energy'] = np.sum(coeff ** 2)
            else:
                # Detail coefficients (horizontal, vertical, diagonal)
                h_coeff, v_coeff, d_coeff = coeff
                
                features[f'wavelet_level{i}_h_mean'] = np.mean(h_coeff)
                features[f'wavelet_level{i}_h_std'] = np.std(h_coeff)
                features[f'wavelet_level{i}_h_energy'] = np.sum(h_coeff ** 2)
                
                features[f'wavelet_level{i}_v_mean'] = np.mean(v_coeff)
                features[f'wavelet_level{i}_v_std'] = np.std(v_coeff)
                features[f'wavelet_level{i}_v_energy'] = np.sum(v_coeff ** 2)
                
                features[f'wavelet_level{i}_d_mean'] = np.mean(d_coeff)
                features[f'wavelet_level{i}_d_std'] = np.std(d_coeff)
                features[f'wavelet_level{i}_d_energy'] = np.sum(d_coeff ** 2)
        
        return features
    
    def extract_temporal_features(self, temporal_data):
        """Extract temporal features from stroke data"""
        features = {}
        
        if 'timestamps' in temporal_data:
            timestamps = temporal_data['timestamps']
            if len(timestamps) > 1:
                # Timing features
                features['total_duration'] = timestamps[-1] - timestamps[0]
                features['mean_interval'] = np.mean(np.diff(timestamps))
                features['interval_std'] = np.std(np.diff(timestamps))
                features['interval_cv'] = features['interval_std'] / features['mean_interval']
                
                # Pause analysis
                intervals = np.diff(timestamps)
                long_pauses = np.sum(intervals > 0.5)  # pauses > 0.5 seconds
                features['long_pause_count'] = long_pauses
                features['long_pause_ratio'] = long_pauses / len(intervals)
        
        if 'velocity' in temporal_data:
            velocity = temporal_data['velocity']
            if len(velocity) > 0:
                # Velocity features
                speed = np.linalg.norm(velocity, axis=1) if velocity.ndim > 1 else np.abs(velocity)
                features['mean_speed'] = np.mean(speed)
                features['speed_std'] = np.std(speed)
                features['speed_cv'] = features['speed_std'] / features['mean_speed']
                features['max_speed'] = np.max(speed)
                features['min_speed'] = np.min(speed)
                features['speed_range'] = features['max_speed'] - features['min_speed']
                
                # Speed entropy
                hist, _ = np.histogram(speed, bins=20, density=True)
                hist = hist[hist > 0]
                features['speed_entropy'] = -np.sum(hist * np.log(hist))
        
        if 'acceleration' in temporal_data:
            acceleration = temporal_data['acceleration']
            if len(acceleration) > 0:
                # Acceleration features
                accel_mag = np.linalg.norm(acceleration, axis=1) if acceleration.ndim > 1 else np.abs(acceleration)
                features['mean_acceleration'] = np.mean(accel_mag)
                features['acceleration_std'] = np.std(accel_mag)
                features['max_acceleration'] = np.max(accel_mag)
                features['jerkiness'] = np.mean(np.abs(np.diff(accel_mag)))
        
        if 'pressures' in temporal_data:
            pressures = temporal_data['pressures']
            if len(pressures) > 0:
                # Pressure features
                features['mean_pressure'] = np.mean(pressures)
                features['pressure_std'] = np.std(pressures)
                features['pressure_cv'] = features['pressure_std'] / features['mean_pressure']
                features['max_pressure'] = np.max(pressures)
                features['min_pressure'] = np.min(pressures)
                features['pressure_range'] = features['max_pressure'] - features['min_pressure']
                
                # Pressure variability
                pressure_diff = np.abs(np.diff(pressures))
                features['pressure_variability'] = np.mean(pressure_diff)
                
                # Pressure entropy
                hist, _ = np.histogram(pressures, bins=20, density=True)
                hist = hist[hist > 0]
                features['pressure_entropy'] = -np.sum(hist * np.log(hist))
        
        # Rhythm and regularity features
        if 'timestamps' in temporal_data and len(timestamps) > 2:
            # Autocorrelation of intervals
            intervals = np.diff(timestamps)
            if len(intervals) > 1:
                autocorr = np.correlate(intervals - np.mean(intervals), 
                                       intervals - np.mean(intervals), mode='full')
                autocorr = autocorr[len(autocorr)//2:]
                autocorr = autocorr / autocorr[0]  # Normalize
                
                features['rhythm_regularity'] = np.mean(np.abs(autocorr[1:6]))  # First 5 lags
        
        return features
    
    def extract_combined_features(self, extracted_features):
        """Create combined/composite features"""
        features = {}
        
        # Stress indicators (composite features)
        if all(k in extracted_features for k in ['pressure_variability', 'speed_cv', 'curvature_std']):
            features['tremor_index'] = (
                extracted_features.get('pressure_variability', 0) * 0.4 +
                extracted_features.get('speed_cv', 0) * 0.3 +
                extracted_features.get('curvature_std', 0) * 0.3
            )
        
        if all(k in extracted_features for k in ['slant_std', 'baseline_angle_std']):
            features['slant_consistency'] = 1 / (1 + extracted_features.get('slant_std', 0) + 
                                                extracted_features.get('baseline_angle_std', 0))
        
        if all(k in extracted_features for k in ['stroke_length_cv', 'word_spacing_std']):
            features['size_consistency'] = 1 / (1 + extracted_features.get('stroke_length_cv', 0) + 
                                               extracted_features.get('word_spacing_std', 0))
        
        # Overall handwriting quality score
        quality_indicators = [
            'tremor_index',
            'slant_consistency',
            'size_consistency',
            'graph_avg_clustering',
            'shannon_entropy'
        ]
        
        quality_score = 0
        weight_sum = 0
        
        for indicator in quality_indicators:
            if indicator in features:
                weight = 1.0
                if indicator == 'tremor_index':
                    # Lower tremor is better
                    quality_score += (1 - min(features[indicator], 1)) * weight
                else:
                    quality_score += min(features[indicator], 1) * weight
                weight_sum += weight
        
        if weight_sum > 0:
            features['handwriting_quality'] = quality_score / weight_sum
        
        return features
    
    def calculate_zernike_moments(self, image, degree):
        """Calculate Zernike moments"""
        # This is a simplified version - for production use a proper implementation
        h, w = image.shape
        y, x = np.mgrid[-1:1:complex(0, h), -1:1:complex(0, w)]
        
        # Convert to polar coordinates
        r = np.sqrt(x**2 + y**2)
        theta = np.arctan2(y, x)
        
        # Mask valid region
        mask = r <= 1
        r = r[mask]
        theta = theta[mask]
        image_vals = image.flatten()[mask.flatten()]
        
        # Calculate Zernike moments
        moments = []
        
        for n in range(degree + 1):
            for m in range(-n, n + 1, 2):
                if abs(m) <= n:
                    # Zernike polynomial
                    R_nm = self.zernike_radial_poly(r, n, abs(m))
                    
                    if m >= 0:
                        V = R_nm * np.cos(m * theta)
                    else:
                        V = R_nm * np.sin(abs(m) * theta)
                    
                    # Calculate moment
                    moment = np.sum(image_vals * V) * (n + 1) / np.pi
                    moments.append(moment)
        
        return np.array(moments)
    
    def zernike_radial_poly(self, r, n, m):
        """Calculate Zernike radial polynomial"""
        R = np.zeros_like(r)
        
        for k in range((n - m) // 2 + 1):
            coeff = ((-1) ** k * np.math.factorial(n - k) /
                    (np.math.factorial(k) * np.math.factorial((n + m) // 2 - k) *
                     np.math.factorial((n - m) // 2 - k)))
            R += coeff * r ** (n - 2 * k)
        
        return R
    
    def zhang_suen_skeletonization(self, image):
        """Zhang-Suen thinning algorithm"""
        # Implementation from earlier
        img = image.copy().astype(bool)
        skeleton = np.zeros_like(img)
        
        changing1 = changing2 = True
        while changing1 or changing2:
            # Step 1
            changing1 = []
            rows, cols = img.shape
            for i in range(1, rows - 1):
                for j in range(1, cols - 1):
                    if img[i, j]:
                        neighbor_coords = [
                            (i-1, j), (i-1, j+1), (i, j+1), (i+1, j+1),
                            (i+1, j), (i+1, j-1), (i, j-1), (i-1, j-1)
                        ]
                        neighbors = [img[r, c] for r, c in neighbor_coords]
                        
                        B_P1 = sum(neighbors)
                        if not (2 <= B_P1 <= 6):
                            continue
                        
                        A_P1 = 0
                        for k in range(8):
                            if neighbors[k] and not neighbors[(k + 1) % 8]:
                                A_P1 += 1
                        if A_P1 != 1:
                            continue
                        
                        if neighbors[0] * neighbors[2] * neighbors[4]:
                            continue
                        
                        if neighbors[2] * neighbors[4] * neighbors[6]:
                            continue
                        
                        changing1.append((i, j))
            
            for i, j in changing1:
                img[i, j] = 0
            
            # Step 2
            changing2 = []
            for i in range(1, rows - 1):
                for j in range(1, cols - 1):
                    if img[i, j]:
                        neighbor_coords = [
                            (i-1, j), (i-1, j+1), (i, j+1), (i+1, j+1),
                            (i+1, j), (i+1, j-1), (i, j-1), (i-1, j-1)
                        ]
                        neighbors = [img[r, c] for r, c in neighbor_coords]
                        
                        B_P1 = sum(neighbors)
                        if not (2 <= B_P1 <= 6):
                            continue
                        
                        A_P1 = 0
                        for k in range(8):
                            if neighbors[k] and not neighbors[(k + 1) % 8]:
                                A_P1 += 1
                        if A_P1 != 1:
                            continue
                        
                        if neighbors[0] * neighbors[2] * neighbors[6]:
                            continue
                        
                        if neighbors[0] * neighbors[4] * neighbors[6]:
                            continue
                        
                        changing2.append((i, j))
            
            for i, j in changing2:
                img[i, j] = 0
            
            changing1 = len(changing1) > 0
            changing2 = len(changing2) > 0
        
        return img.astype(np.uint8)