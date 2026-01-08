import cv2
import numpy as np
from scipy import ndimage, signal
import matplotlib.pyplot as plt
from skimage import filters, morphology, measure
import warnings
warnings.filterwarnings('ignore')

class HandwritingPreprocessor:
    """Complete preprocessing pipeline for handwriting analysis"""
    
    def __init__(self, target_size=(512, 512)):
        self.target_size = target_size
        self.preprocessing_pipeline = [
            'grayscale',
            'denoise',
            'binarize',
            'normalize',
            'deskew',
            'remove_borders',
            'skeletonize'
        ]
    
    def preprocess_image(self, image, pipeline=None):
        """Main preprocessing function with configurable pipeline"""
        if pipeline is None:
            pipeline = self.preprocessing_pipeline
        
        results = {'original': image.copy()}
        processed = image.copy()
        
        for step in pipeline:
            if step == 'grayscale':
                processed = self.convert_grayscale(processed)
                results['grayscale'] = processed.copy()
                
            elif step == 'denoise':
                processed = self.denoise_image(processed)
                results['denoised'] = processed.copy()
                
            elif step == 'binarize':
                processed = self.binarize_image(processed)
                results['binary'] = processed.copy()
                
            elif step == 'normalize':
                processed = self.normalize_image(processed)
                results['normalized'] = processed.copy()
                
            elif step == 'deskew':
                processed = self.deskew_image(processed)
                results['deskewed'] = processed.copy()
                
            elif step == 'remove_borders':
                processed = self.remove_borders(processed)
                results['border_removed'] = processed.copy()
                
            elif step == 'skeletonize':
                processed = self.skeletonize_image(processed)
                results['skeleton'] = processed.copy()
                
            elif step == 'enhance_contrast':
                processed = self.enhance_contrast(processed)
                results['contrast_enhanced'] = processed.copy()
                
            elif step == 'remove_small_objects':
                processed = self.remove_small_objects(processed)
                results['cleaned'] = processed.copy()
        
        results['final'] = processed.copy()
        return results
    
    def convert_grayscale(self, image):
        """Convert image to grayscale"""
        if len(image.shape) == 3:
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return image
    
    def denoise_image(self, image):
        """Advanced denoising with multiple methods"""
        if len(image.shape) == 3:
            # For color images
            denoised = cv2.fastNlMeansDenoisingColored(
                image, None, h=10, hColor=10, templateWindowSize=7, searchWindowSize=21
            )
        else:
            # For grayscale images
            # Try multiple denoising methods
            denoised_nlm = cv2.fastNlMeansDenoising(
                image, None, h=10, templateWindowSize=7, searchWindowSize=21
            )
            
            # Median filter for salt-and-pepper noise
            denoised_median = cv2.medianBlur(image, 3)
            
            # Gaussian filter
            denoised_gaussian = cv2.GaussianBlur(image, (3, 3), 0)
            
            # Combine results
            denoised = cv2.addWeighted(denoised_nlm, 0.5, denoised_median, 0.5, 0)
        
        return denoised
    
    def binarize_image(self, image, method='adaptive'):
        """Multiple binarization methods"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        if method == 'otsu':
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
        elif method == 'adaptive':
            # Adaptive Gaussian thresholding
            binary = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY, 11, 2
            )
            
        elif method == 'sauvola':
            # Sauvola binarization
            window_size = 25
            thresh_sauvola = filters.threshold_sauvola(gray, window_size=window_size)
            binary = (gray > thresh_sauvola).astype(np.uint8) * 255
            
        elif method == 'multi':
            # Combine multiple methods
            _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            adaptive = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY, 11, 2
            )
            
            # Combine Otsu and Adaptive
            binary = cv2.bitwise_and(otsu, adaptive)
            
        else:
            raise ValueError(f"Unknown binarization method: {method}")
        
        return binary
    
    def normalize_image(self, image):
        """Normalize image size and intensity"""
        # Resize to target size
        normalized = cv2.resize(image, self.target_size, interpolation=cv2.INTER_AREA)
        
        # Normalize intensity to [0, 255]
        if normalized.dtype != np.uint8:
            normalized = cv2.normalize(normalized, None, 0, 255, cv2.NORM_MINMAX)
            normalized = normalized.astype(np.uint8)
        
        # Enhance contrast using CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        if len(normalized.shape) == 2:
            normalized = clahe.apply(normalized)
        
        return normalized
    
    def deskew_image(self, image):
        """Deskew handwriting using Hough transform"""
        # Ensure binary image
        if image.max() > 1:
            _, binary = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
        else:
            binary = image
        
        # Detect lines using Hough transform
        edges = cv2.Canny(binary, 50, 150, apertureSize=3)
        lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=100)
        
        if lines is not None:
            angles = []
            for rho, theta in lines[:, 0]:
                angle = np.degrees(theta)
                # Filter reasonable angles for handwriting
                if 0 < angle < 180:
                    # Convert to skew angle (-45 to 45 degrees)
                    if angle > 90:
                        angle = angle - 180
                    if abs(angle) < 45:  # Reasonable skew range
                        angles.append(angle)
            
            if angles:
                median_angle = np.median(angles)
                
                # Rotate image to correct skew
                (h, w) = binary.shape[:2]
                center = (w // 2, h // 2)
                M = cv2.getRotationMatrix2D(center, median_angle, 1.0)
                deskewed = cv2.warpAffine(
                    binary, M, (w, h),
                    flags=cv2.INTER_CUBIC,
                    borderMode=cv2.BORDER_REPLICATE
                )
                return deskewed
        
        return image
    
    def remove_borders(self, image, margin=10):
        """Remove borders and extract writing region"""
        if image.max() > 1:
            _, binary = cv2.threshold(image, 127, 1, cv2.THRESH_BINARY)
        else:
            binary = (image > 0).astype(np.uint8)
        
        # Find all non-zero points
        points = np.column_stack(np.where(binary > 0))
        
        if len(points) > 0:
            # Get bounding box
            y_min, x_min = points.min(axis=0)
            y_max, x_max = points.max(axis=0)
            
            # Add margin
            y_min = max(0, y_min - margin)
            x_min = max(0, x_min - margin)
            y_max = min(binary.shape[0], y_max + margin)
            x_max = min(binary.shape[1], x_max + margin)
            
            # Crop image
            cropped = image[y_min:y_max, x_min:x_max]
            
            # Resize back to original size if needed
            if cropped.shape[0] > 0 and cropped.shape[1] > 0:
                return cv2.resize(cropped, self.target_size, interpolation=cv2.INTER_AREA)
        
        return image
    
    def skeletonize_image(self, image, method='zhang'):
        """Skeletonize binary image"""
        if image.max() > 1:
            _, binary = cv2.threshold(image, 127, 1, cv2.THRESH_BINARY)
        else:
            binary = (image > 0).astype(np.uint8)
        
        if method == 'zhang':
            skeleton = self.zhang_suen_skeletonization(binary)
        elif method == 'medial_axis':
            from skimage.morphology import medial_axis
            skeleton = medial_axis(binary).astype(np.uint8)
        elif method == 'thinning':
            from skimage.morphology import thin
            skeleton = thin(binary).astype(np.uint8)
        else:
            raise ValueError(f"Unknown skeletonization method: {method}")
        
        return skeleton * 255
    
    def zhang_suen_skeletonization(self, image):
        """Zhang-Suen thinning algorithm"""
        # Convert to boolean
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
                        
                        # Condition 1: 2 <= B(P1) <= 6
                        B_P1 = sum(neighbors)
                        if not (2 <= B_P1 <= 6):
                            continue
                        
                        # Condition 2: A(P1) = 1
                        A_P1 = 0
                        for k in range(8):
                            if neighbors[k] and not neighbors[(k + 1) % 8]:
                                A_P1 += 1
                        if A_P1 != 1:
                            continue
                        
                        # Condition 3: P2 * P4 * P6 = 0
                        if neighbors[0] * neighbors[2] * neighbors[4]:
                            continue
                        
                        # Condition 4: P4 * P6 * P8 = 0
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
                        
                        # Condition 1: 2 <= B(P1) <= 6
                        B_P1 = sum(neighbors)
                        if not (2 <= B_P1 <= 6):
                            continue
                        
                        # Condition 2: A(P1) = 1
                        A_P1 = 0
                        for k in range(8):
                            if neighbors[k] and not neighbors[(k + 1) % 8]:
                                A_P1 += 1
                        if A_P1 != 1:
                            continue
                        
                        # Condition 3: P2 * P4 * P8 = 0
                        if neighbors[0] * neighbors[2] * neighbors[6]:
                            continue
                        
                        # Condition 4: P2 * P6 * P8 = 0
                        if neighbors[0] * neighbors[4] * neighbors[6]:
                            continue
                        
                        changing2.append((i, j))
            
            for i, j in changing2:
                img[i, j] = 0
            
            changing1 = len(changing1) > 0
            changing2 = len(changing2) > 0
        
        skeleton = img.astype(np.uint8)
        return skeleton
    
    def enhance_contrast(self, image):
        """Enhance contrast using histogram equalization"""
        if len(image.shape) == 2:
            # CLAHE for local contrast enhancement
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(image)
        else:
            # For color images, enhance each channel
            enhanced = np.zeros_like(image)
            for i in range(3):
                enhanced[:, :, i] = cv2.equalizeHist(image[:, :, i])
        
        return enhanced
    
    def remove_small_objects(self, image, min_size=50):
        """Remove small noise objects"""
        if image.max() > 1:
            _, binary = cv2.threshold(image, 127, 1, cv2.THRESH_BINARY)
        else:
            binary = (image > 0).astype(np.uint8)
        
        # Label connected components
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            binary, connectivity=8
        )
        
        # Create mask for components larger than min_size
        mask = np.zeros_like(binary)
        for i in range(1, num_labels):  # Skip background
            if stats[i, cv2.CC_STAT_AREA] >= min_size:
                mask[labels == i] = 1
        
        return mask * 255
    
    def extract_writing_regions(self, image):
        """Extract individual lines or words"""
        if image.max() > 1:
            _, binary = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY_INV)
        else:
            binary = (1 - image) * 255  # Invert for contour detection
        
        # Find contours
        contours, _ = cv2.findContours(
            binary.astype(np.uint8), 
            cv2.RETR_EXTERNAL, 
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        regions = []
        for contour in contours:
            if cv2.contourArea(contour) > 100:  # Minimum area threshold
                x, y, w, h = cv2.boundingRect(contour)
                region = image[y:y+h, x:x+w]
                regions.append({
                    'image': region,
                    'bbox': (x, y, w, h),
                    'area': cv2.contourArea(contour)
                })
        
        # Sort regions by y-coordinate (top to bottom)
        regions.sort(key=lambda r: r['bbox'][1])
        
        return regions
    
    def create_preprocessing_report(self, image, steps=None):
        """Generate visualization of preprocessing steps"""
        if steps is None:
            steps = self.preprocessing_pipeline
        
        results = self.preprocess_image(image, steps)
        
        # Create figure
        n_steps = len(results)
        fig, axes = plt.subplots(1, n_steps, figsize=(4*n_steps, 4))
        
        if n_steps == 1:
            axes = [axes]
        
        for ax, (name, img) in zip(axes, results.items()):
            if len(img.shape) == 2:
                ax.imshow(img, cmap='gray')
            else:
                ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            ax.set_title(name)
            ax.axis('off')
        
        plt.tight_layout()
        return fig, results

class TemporalPreprocessor:
    """Preprocessing for temporal handwriting data"""
    
    def __init__(self, sampling_rate=30):
        self.sampling_rate = sampling_rate
    
    def preprocess_sequence(self, sequence_data):
        """Preprocess temporal sequence data"""
        processed = {}
        
        # Extract components
        if 'timestamps' in sequence_data:
            processed['timestamps'] = self.normalize_timestamps(
                sequence_data['timestamps']
            )
        
        if 'positions' in sequence_data:
            processed['positions'] = self.smooth_positions(
                sequence_data['positions']
            )
            
            # Calculate derivatives
            processed['velocity'] = self.calculate_velocity(
                processed['positions'], processed.get('timestamps')
            )
            processed['acceleration'] = self.calculate_acceleration(
                processed['velocity'], processed.get('timestamps')
            )
        
        if 'pressures' in sequence_data:
            processed['pressures'] = self.normalize_pressures(
                sequence_data['pressures']
            )
        
        # Extract features
        processed['features'] = self.extract_temporal_features(processed)
        
        return processed
    
    def normalize_timestamps(self, timestamps):
        """Normalize timestamps to start from 0"""
        if len(timestamps) > 0:
            return np.array(timestamps) - timestamps[0]
        return np.array(timestamps)
    
    def smooth_positions(self, positions, window_size=5):
        """Smooth position data using moving average"""
        if len(positions) == 0:
            return positions
        
        positions = np.array(positions)
        smoothed = np.zeros_like(positions)
        
        for i in range(len(positions)):
            start = max(0, i - window_size // 2)
            end = min(len(positions), i + window_size // 2 + 1)
            smoothed[i] = np.mean(positions[start:end], axis=0)
        
        return smoothed
    
    def normalize_pressures(self, pressures):
        """Normalize pressure values to [0, 1]"""
        pressures = np.array(pressures)
        if len(pressures) > 0:
            min_p = pressures.min()
            max_p = pressures.max()
            if max_p > min_p:
                return (pressures - min_p) / (max_p - min_p)
        return pressures
    
    def calculate_velocity(self, positions, timestamps=None):
        """Calculate velocity from position data"""
        if len(positions) < 2:
            return np.zeros_like(positions)
        
        velocities = np.zeros_like(positions)
        
        if timestamps is not None and len(timestamps) == len(positions):
            # Use timestamps for accurate velocity
            dt = np.diff(timestamps)
            dt = np.concatenate([dt, [dt[-1]]])  # Extend last value
            
            for i in range(len(positions)):
                if i == 0:
                    velocities[i] = (positions[1] - positions[0]) / dt[0]
                elif i == len(positions) - 1:
                    velocities[i] = (positions[-1] - positions[-2]) / dt[-1]
                else:
                    velocities[i] = (positions[i+1] - positions[i-1]) / (dt[i-1] + dt[i])
        else:
            # Assume uniform sampling
            for i in range(1, len(positions)):
                velocities[i] = positions[i] - positions[i-1]
        
        return velocities
    
    def calculate_acceleration(self, velocities, timestamps=None):
        """Calculate acceleration from velocity data"""
        if len(velocities) < 2:
            return np.zeros_like(velocities)
        
        accelerations = np.zeros_like(velocities)
        
        if timestamps is not None and len(timestamps) == len(velocities):
            # Use timestamps for accurate acceleration
            dt = np.diff(timestamps)
            dt = np.concatenate([dt, [dt[-1]]])  # Extend last value
            
            for i in range(len(velocities)):
                if i == 0:
                    accelerations[i] = (velocities[1] - velocities[0]) / dt[0]
                elif i == len(velocities) - 1:
                    accelerations[i] = (velocities[-1] - velocities[-2]) / dt[-1]
                else:
                    accelerations[i] = (velocities[i+1] - velocities[i-1]) / (dt[i-1] + dt[i])
        else:
            # Assume uniform sampling
            for i in range(1, len(velocities)):
                accelerations[i] = velocities[i] - velocities[i-1]
        
        return accelerations
    
    def extract_temporal_features(self, processed_data):
        """Extract temporal features from processed data"""
        features = {}
        
        # Time-based features
        if 'timestamps' in processed_data:
            timestamps = processed_data['timestamps']
            if len(timestamps) > 1:
                features['total_duration'] = timestamps[-1] - timestamps[0]
                features['mean_sampling_rate'] = len(timestamps) / features['total_duration']
        
        # Motion-based features
        if 'velocity' in processed_data:
            velocity = processed_data['velocity']
            if len(velocity) > 0:
                speed = np.linalg.norm(velocity, axis=1)
                features['mean_speed'] = np.mean(speed)
                features['speed_std'] = np.std(speed)
                features['max_speed'] = np.max(speed)
        
        if 'acceleration' in processed_data:
            acceleration = processed_data['acceleration']
            if len(acceleration) > 0:
                accel_magnitude = np.linalg.norm(acceleration, axis=1)
                features['mean_acceleration'] = np.mean(accel_magnitude)
                features['acceleration_std'] = np.std(accel_magnitude)
        
        # Pressure-based features
        if 'pressures' in processed_data:
            pressures = processed_data['pressures']
            if len(pressures) > 0:
                features['mean_pressure'] = np.mean(pressures)
                features['pressure_std'] = np.std(pressures)
                features['pressure_variability'] = np.std(pressures) / np.mean(pressures)
        
        return features
    
    def segment_strokes(self, positions, pressures=None, velocity_threshold=0.1):
        """Segment continuous writing into individual strokes"""
        if len(positions) < 2:
            return []
        
        # Calculate speed
        if 'velocity' in locals():
            speed = np.linalg.norm(positions[1:] - positions[:-1], axis=1)
        else:
            # Approximate speed from positions
            speed = np.linalg.norm(positions[1:] - positions[:-1], axis=1)
        
        # Find pause points (low speed)
        pause_points = np.where(speed < velocity_threshold)[0]
        
        strokes = []
        start_idx = 0
        
        for pause in pause_points:
            if pause - start_idx > 1:  # Minimum stroke length
                stroke = {
                    'positions': positions[start_idx:pause+1],
                    'indices': (start_idx, pause)
                }
                
                if pressures is not None:
                    stroke['pressures'] = pressures[start_idx:pause+1]
                
                strokes.append(stroke)
            
            start_idx = pause + 1
        
        # Add last stroke
        if start_idx < len(positions) - 1:
            stroke = {
                'positions': positions[start_idx:],
                'indices': (start_idx, len(positions) - 1)
            }
            
            if pressures is not None:
                stroke['pressures'] = pressures[start_idx:]
            
            strokes.append(stroke)
        
        return strokes