import cv2
import numpy as np
import mediapipe as mp
import time
from collections import deque
import threading
import queue

class RealTimeHandwritingCapture:
    """Real-time handwriting acquisition with temporal tracking"""
    
    def __init__(self, camera_id=0, fps=30, buffer_size=60):
        self.camera_id = camera_id
        self.fps = fps
        self.buffer_size = buffer_size
        self.cap = None
        self.running = False
        
        # MediaPipe for hand tracking
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        
        # Temporal buffers
        self.frame_buffer = deque(maxlen=buffer_size)
        self.timestamp_buffer = deque(maxlen=buffer_size)
        self.hand_landmarks_buffer = deque(maxlen=buffer_size)
        self.pressure_buffer = deque(maxlen=buffer_size)
        
        # Writing surface calibration
        self.writing_surface = None
        self.surface_calibrated = False
        
        # Threading
        self.frame_queue = queue.Queue(maxsize=10)
        self.processing_thread = None
        
    def start_capture(self):
        """Start real-time capture"""
        self.cap = cv2.VideoCapture(self.camera_id)
        if not self.cap.isOpened():
            raise Exception("Cannot open webcam")
        
        self.cap.set(cv2.CAP_PROP_FPS, self.fps)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        self.running = True
        self.processing_thread = threading.Thread(target=self._capture_loop)
        self.processing_thread.start()
        
        print("Real-time capture started...")
        print("Press 'c' to calibrate writing surface")
        print("Press 's' to start/stop analysis")
        print("Press 'q' to quit")
        
    def _capture_loop(self):
        """Main capture loop"""
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            # Mirror for natural writing
            frame = cv2.flip(frame, 1)
            
            # Process frame
            timestamp = time.time()
            processed_frame, hand_landmarks = self._process_frame(frame)
            
            # Store in buffers
            self.frame_buffer.append(processed_frame.copy())
            self.timestamp_buffer.append(timestamp)
            self.hand_landmarks_buffer.append(hand_landmarks)
            
            # Estimate pressure
            pressure = self._estimate_pressure(hand_landmarks)
            self.pressure_buffer.append(pressure)
            
            # Add to queue for display
            if not self.frame_queue.full():
                self.frame_queue.put(processed_frame)
            
            # Small delay to maintain FPS
            time.sleep(1/self.fps)
    
    def _process_frame(self, frame):
        """Process individual frame"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb_frame.flags.writeable = False
        results = self.hands.process(rgb_frame)
        
        # Convert back to BGR for display
        rgb_frame.flags.writeable = True
        processed_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
        
        hand_landmarks = None
        
        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            
            # Draw hand landmarks
            self.mp_drawing.draw_landmarks(
                processed_frame, 
                hand_landmarks, 
                self.mp_hands.HAND_CONNECTIONS,
                self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                self.mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2)
            )
            
            # Draw writing region if calibrated
            if self.surface_calibrated and self.writing_surface:
                x1, y1, x2, y2 = self.writing_surface
                cv2.rectangle(processed_frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
                cv2.putText(processed_frame, "Writing Surface", (x1, y1-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        
        return processed_frame, hand_landmarks
    
    def _estimate_pressure(self, hand_landmarks):
        """Estimate writing pressure from hand landmarks"""
        if hand_landmarks is None:
            return 0.5  # Default medium pressure
        
        try:
            # Get key landmarks
            index_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]
            index_mcp = hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_MCP]
            wrist = hand_landmarks.landmark[self.mp_hands.HandLandmark.WRIST]
            
            # Calculate distances
            tip_to_mcp = np.sqrt((index_tip.x - index_mcp.x)**2 + 
                                (index_tip.y - index_mcp.y)**2)
            tip_to_wrist = np.sqrt((index_tip.x - wrist.x)**2 + 
                                  (index_tip.y - wrist.y)**2)
            
            # Normalize pressure (0-1 range)
            pressure = (tip_to_mcp + tip_to_wrist) / 2
            pressure = np.clip(pressure * 2, 0, 1)  # Scale to reasonable range
            
            return float(pressure)
        except:
            return 0.5
    
    def calibrate_writing_surface(self, top_left, bottom_right):
        """Calibrate the writing surface area"""
        self.writing_surface = (top_left[0], top_left[1], 
                               bottom_right[0], bottom_right[1])
        self.surface_calibrated = True
        print(f"Writing surface calibrated: {self.writing_surface}")
    
    def get_stroke_data(self):
        """Extract stroke data from buffer"""
        if len(self.frame_buffer) < 2:
            return None
        
        # Convert frames to grayscale
        gray_frames = []
        for frame in self.frame_buffer:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray_frames.append(gray)
        
        # Calculate optical flow for stroke detection
        stroke_data = {
            'frames': list(self.frame_buffer),
            'timestamps': list(self.timestamp_buffer),
            'pressures': list(self.pressure_buffer),
            'landmarks': list(self.hand_landmarks_buffer),
            'optical_flow': self._calculate_optical_flow(gray_frames)
        }
        
        return stroke_data
    
    def _calculate_optical_flow(self, gray_frames):
        """Calculate optical flow between consecutive frames"""
        if len(gray_frames) < 2:
            return []
        
        flow_data = []
        prev_gray = gray_frames[0]
        
        for gray in gray_frames[1:]:
            flow = cv2.calcOpticalFlowFarneback(
                prev_gray, gray, None,
                0.5, 3, 15, 3, 5, 1.2, 0
            )
            magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            flow_data.append({
                'magnitude': magnitude,
                'angle': angle,
                'mean_magnitude': np.mean(magnitude)
            })
            prev_gray = gray
        
        return flow_data
    
    def stop_capture(self):
        """Stop real-time capture"""
        self.running = False
        if self.processing_thread:
            self.processing_thread.join()
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        print("Capture stopped")
    
    def get_latest_frame(self):
        """Get latest frame for display"""
        try:
            return self.frame_queue.get(timeout=0.1)
        except queue.Empty:
            return None

class DigitalWritingCapture:
    """Capture digital writing from tablet or touch screen"""
    
    def __init__(self):
        self.stroke_points = []
        self.timestamps = []
        self.pressures = []
        self.strokes = []
        self.current_stroke = []
        self.current_timestamps = []
        self.current_pressures = []
        
    def start_stroke(self, x, y, pressure=0.5, timestamp=None):
        """Start a new stroke"""
        self.current_stroke = [(x, y)]
        self.current_timestamps = [timestamp or time.time()]
        self.current_pressures = [pressure]
    
    def add_point(self, x, y, pressure=0.5, timestamp=None):
        """Add point to current stroke"""
        self.current_stroke.append((x, y))
        self.current_timestamps.append(timestamp or time.time())
        self.current_pressures.append(pressure)
    
    def end_stroke(self):
        """End current stroke"""
        if len(self.current_stroke) > 1:
            self.strokes.append({
                'points': self.current_stroke.copy(),
                'timestamps': self.current_timestamps.copy(),
                'pressures': self.current_pressures.copy()
            })
            self.stroke_points.extend(self.current_stroke)
            self.timestamps.extend(self.current_timestamps)
            self.pressures.extend(self.current_pressures)
        
        self.current_stroke = []
        self.current_timestamps = []
        self.current_pressures = []
    
    def get_stroke_data(self):
        """Get all stroke data"""
        return {
            'strokes': self.strokes,
            'points': self.stroke_points,
            'timestamps': self.timestamps,
            'pressures': self.pressures
        }
    
    def clear(self):
        """Clear all stroke data"""
        self.stroke_points = []
        self.timestamps = []
        self.pressures = []
        self.strokes = []
        self.current_stroke = []
        self.current_timestamps = []
        self.current_pressures = []