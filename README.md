# Space-Station-Object-Detection 

This project uses YOLO (You Only Look Once) for object detection on space station datasets. It predicts objects in images and saves both the annotated images and bounding box labels.

## Features

- Predicts objects in images using a trained YOLO model
- Saves annotated images and bounding box labels
- Easy configuration via `yolo_params.yaml`
- Organized output structure for predictions

## Project Structure

```
HackByte_Dataset/
│
├── ENV_SETUP/                # Environment setup scripts
├── predictions/              # Output images and labels after prediction
├── runs/                     # YOLO training runs and weights
├── yolo_params.yaml          # Configuration file for paths
├── predict.py                # Main prediction script
├── .gitignore                # Git ignore rules
└── README.md                 # Project documentation
```

## Setup

1. **Clone the repository:**
   ```sh
   git clone https://github.com/NANDINIIIGUPTAAA/Space-Station-Object-Detection.git
   cd Space-Station-Object-Detection
   ```

2. **Set up the environment:**
   - Use the scripts in `ENV_SETUP/` or install dependencies manually:
     ```sh
     pip install -r requirements.txt
     ```

3. **Prepare your data:**
   - Place your test images in the directory specified in `yolo_params.yaml` under the `test` field.

4. **Run predictions:**
   ```sh
   python predict.py
   ```

## Output

- Annotated images are saved in `predictions/images/`
- Bounding box labels are saved in `predictions/labels/`

## Notes

- The `data/` folder is excluded from version control via `.gitignore`.
- Make sure you have your trained YOLO weights in the appropriate `runs/detect/train*/weights/best.pt` path.

## License

This project is for educational and research purposes.

---
