"""
Example: Complete emotion recognition workflow
"""

import cv2
import numpy as np
from pathlib import Path

from affective_intelligence.models import EmotionNet, EmotionNetConfig
from affective_intelligence.inference import EmotionPredictor
from affective_intelligence.datasets import FER2013Dataset, get_val_transforms


def example_1_single_image_inference():
    """Example 1: Single image emotion prediction."""
    print("\n=== Example 1: Single Image Inference ===")
    
    # Initialize predictor
    predictor = EmotionPredictor(
        model_path="./checkpoints/best_model.pt",
        device="cuda"
    )
    
    # Load image
    image = cv2.imread("path/to/face.jpg")
    
    # Predict both macro and micro
    result = predictor.predict_both(image)
    
    print(f"Macro-Expression: {result['macro']['emotion']} "
          f"({result['macro']['confidence']:.2%})")
    print(f"Micro-Expression: {result['micro']['emotion']} "
          f"({result['micro']['classification_confidence']:.2%})")
    
    if result['micro']['is_micro_expression']:
        print("⚠️  Detected involuntary emotional leak!")


def example_2_video_stream():
    """Example 2: Real-time video stream processing."""
    print("\n=== Example 2: Video Stream Processing ===")
    
    predictor = EmotionPredictor("./checkpoints/best_model.pt", device="cuda")
    cap = cv2.VideoCapture(0)
    
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Process every 5th frame for speed
        if frame_count % 5 == 0:
            result = predictor.predict_both(frame)
            
            # Draw macro-expression
            macro = result["macro"]["emotion"]
            macro_conf = result["macro"]["confidence"]
            cv2.putText(frame, f"Macro: {macro} ({macro_conf:.1%})",
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Highlight micro-expression if detected
            if result["micro"]["is_micro_expression"]:
                micro = result["micro"]["emotion"]
                cv2.putText(frame, f"⚠️  MICRO: {micro} (involuntary)",
                           (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        cv2.imshow("Emotion Recognition", frame)
        
        frame_count += 1
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()


def example_3_batch_inference():
    """Example 3: Batch inference on directory of images."""
    print("\n=== Example 3: Batch Inference ===")
    
    predictor = EmotionPredictor("./checkpoints/best_model.pt")
    
    results = []
    image_dir = Path("path/to/images")
    
    for img_path in image_dir.glob("*.jpg"):
        image = cv2.imread(str(img_path))
        if image is None:
            continue
        
        pred = predictor.predict_both(image)
        
        results.append({
            "image": img_path.name,
            "macro_emotion": pred["macro"]["emotion"],
            "macro_conf": f"{pred['macro']['confidence']:.2%}",
            "micro_emotion": pred["micro"]["emotion"],
            "micro_conf": f"{pred['micro']['classification_confidence']:.2%}",
            "is_micro_expr": pred["micro"]["is_micro_expression"],
        })
    
    # Print results
    for r in results:
        print(f"{r['image']}: "
              f"Macro={r['macro_emotion']} ({r['macro_conf']}) | "
              f"Micro={r['micro_emotion']} | "
              f"Involuntary={r['is_micro_expr']}")


def example_4_dataset_loading():
    """Example 4: Load and explore FER2013 dataset."""
    print("\n=== Example 4: Dataset Loading ===")
    
    # Load FER2013 dataset
    transform = get_val_transforms()
    dataset = FER2013Dataset(
        data_dir="path/to/FER2013",
        split="train",
        transform=transform
    )
    
    print(f"Dataset size: {len(dataset)}")
    print(f"Emotions: {dataset.EMOTIONS}")
    
    # Get a sample
    image, label = dataset[0]
    emotion_name = dataset.get_emotion_name(label)
    print(f"Sample shape: {image.shape}, Emotion: {emotion_name}")


def example_5_model_configuration():
    """Example 5: Custom model configuration."""
    print("\n=== Example 5: Model Configuration ===")
    
    # Create custom config
    config = EmotionNetConfig(
        num_macro_classes=7,
        num_micro_classes=5,
        pretrained=True,
        dropout_rate=0.3,
        use_attention=True,
        embedding_dim=512,
    )
    
    # Initialize model
    model = EmotionNet(config)
    
    # Get model summary
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Configuration: {config.__dict__}")


def example_6_emotion_prediction_api():
    """Example 6: Using FastAPI endpoints."""
    print("\n=== Example 6: FastAPI Integration ===")
    
    import requests
    
    # Start the server first:
    # python -m uvicorn app.main:app --reload
    
    url_base = "http://localhost:8000/api/v1/emotion"
    
    # Upload image and predict macro
    with open("path/to/face.jpg", "rb") as f:
        response = requests.post(
            f"{url_base}/predict/macro",
            files={"file": f}
        )
        result = response.json()
        print(f"Macro prediction: {result['emotion']} ({result['confidence']:.2%})")
    
    # Predict micro
    with open("path/to/face.jpg", "rb") as f:
        response = requests.post(
            f"{url_base}/predict/micro",
            files={"file": f},
            params={"confidence_threshold": 0.5}
        )
        result = response.json()
        print(f"Micro prediction: {result['emotion']} ({result['confidence']:.2%})")
    
    # Dual prediction
    with open("path/to/face.jpg", "rb") as f:
        response = requests.post(
            f"{url_base}/predict/dual",
            files={"file": f}
        )
        result = response.json()
        print(f"Macro: {result['macro_emotion']} | "
              f"Micro: {result['micro_emotion']} | "
              f"Is involuntary: {result['is_micro_expression']}")


def example_7_advanced_features():
    """Example 7: Advanced features and customization."""
    print("\n=== Example 7: Advanced Features ===")
    
    predictor = EmotionPredictor("./checkpoints/best_model.pt")
    
    # Load image
    image = cv2.imread("path/to/face.jpg")
    
    # Get embeddings (useful for clustering/similarity)
    import torch
    tensor = predictor.preprocess_image(image)
    with torch.no_grad():
        output = predictor.model(tensor, return_embeddings=True)
        embeddings = output["embeddings"]
    print(f"Embedding shape: {embeddings.shape}")
    print(f"Can use embeddings for: similarity search, clustering, face recognition")
    
    # Custom confidence threshold
    result_strict = predictor.predict_micro_emotion(image, confidence_threshold=0.7)
    result_lenient = predictor.predict_micro_emotion(image, confidence_threshold=0.3)
    
    print(f"Strict threshold (0.7): is_micro={result_strict['is_micro_expression']}")
    print(f"Lenient threshold (0.3): is_micro={result_lenient['is_micro_expression']}")
    
    # Get model info
    info = predictor.get_model_summary()
    print(f"Model info: {info}")


if __name__ == "__main__":
    print("Facial Emotion Recognition Examples")
    print("=" * 50)
    
    # Run examples (comment out as needed)
    # example_1_single_image_inference()
    # example_2_video_stream()
    # example_3_batch_inference()
    # example_4_dataset_loading()
    example_5_model_configuration()
    example_6_emotion_prediction_api()
    example_7_advanced_features()
    
    print("\n" + "=" * 50)
    print("Examples completed!")
