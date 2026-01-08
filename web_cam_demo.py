import os
import cv2
import numpy as np
from src.inference import RealTimeWebcamInference

def main():
    print("Starting real-time handwriting analysis...")
    print("Instructions:")
    print("1. Write or draw on paper")
    print("2. Hold it in front of the camera")
    print("3. Press 's' to analyze")
    print("4. Press 'q' to quit")
    
    # Initialize inference
    MODEL_PATH = os.path.join("saved_models", "best_model.pth")
    inference = RealTimeWebcamInference(MODEL_PATH)

    #inference = RealTimeWebcamInference('model/best_model.pth')
    
    # Run webcam analysis
    results = inference.run()
    
    # Display summary
    if results:
        print(f"\nAnalysis complete! Processed {len(results)} frames.")
        for i, result in enumerate(results):
            print(f"\nFrame {i+1}:")
            print(f"  Prediction: {result['prediction']}")
            print(f"  Confidence: {result['confidence']:.2%}")
            print(f"  Risk Score: {result['risk_score']:.1f}")

if __name__ == "__main__":
    main()