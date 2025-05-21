import argparse
import os
from ultralytics import YOLO

# Default training parameters
EPOCHS = 5
MOSAIC = 0.1
OPTIMIZER = 'AdamW'
MOMENTUM = 0.2
LR0 = 0.001
LRF = 0.0001
SINGLE_CLS = False
MODEL_NAME = "yolov8s.pt"
DATA_YAML = "yolo_params.yaml"

if __name__ == '__main__':
    # Argument parser
    parser = argparse.ArgumentParser(description="Train YOLOv8 model")
    parser.add_argument('--epochs', type=int, default=EPOCHS, help='Number of training epochs')
    parser.add_argument('--mosaic', type=float, default=MOSAIC, help='Mosaic augmentation factor')
    parser.add_argument('--optimizer', type=str, default=OPTIMIZER, help='Optimizer to use (SGD, Adam, AdamW)')
    parser.add_argument('--momentum', type=float, default=MOMENTUM, help='Momentum for optimizer')
    parser.add_argument('--lr0', type=float, default=LR0, help='Initial learning rate')
    parser.add_argument('--lrf', type=float, default=LRF, help='Final learning rate')
    parser.add_argument('--single_cls', type=bool, default=SINGLE_CLS, help='Train as single class')
    parser.add_argument('--model', type=str, default=MODEL_NAME, help='Model file (e.g., yolov8s.pt)')
    parser.add_argument('--data', type=str, default=DATA_YAML, help='Dataset YAML file')
    args = parser.parse_args()

    # Set working directory to current file's location
    this_dir = os.path.dirname(__file__)
    os.chdir(this_dir)

    # Load YOLO model
    model_path = os.path.join(this_dir, args.model)
    model = YOLO(model_path)

    # Train the model
    results = model.train(
        data=os.path.join(this_dir, args.data),
        epochs=args.epochs,
        device='cpu',  # Force CPU usage due to unsupported GPU
        single_cls=args.single_cls,
        mosaic=args.mosaic,
        optimizer=args.optimizer,
        momentum=args.momentum,
        lr0=args.lr0,
        lrf=args.lrf
    )

    print("✅ Training complete.")
