Smoking Detection System using YOLO11

This repository contains a complete pipeline for building a custom smoking-detection model using the Ultralytics YOLO11 framework.
The project includes dataset configuration, training scripts, model export tools, and example code for running inference on video inputs.
The trained model can later be integrated into a mobile application for real-time smoking detection.

1. Project Structure
smoking-project/
│
├── android/                      # Mobile application (not yet integrated)
│
├── training/
│   ├── train_smoking.py          # Script for training the YOLO11 model
│   ├── detect_video.py           # Script for running detection on video input
│   ├── export_onnx.py            # Export trained model to ONNX format
│   ├── smoking.yaml              # Dataset configuration file
│   └── alldatav1_relabel/        # Local dataset directory (ignored in Git)
│
└── .gitignore


Note: The dataset and large model files are excluded from version control to keep the repository compact.

2. Requirements and Installation

Install required libraries:

pip install ultralytics
pip install opencv-python


Python 3.10+ is recommended.

3. Dataset Format

The project uses the standard YOLO dataset structure:

dataset/
├── images/
│   ├── train/
│   └── val/
└── labels/
    ├── train/
    └── val/


Each label file follows YOLO format:

<class_id> <x_center> <y_center> <width> <height>


Update smoking.yaml so that the train: and val: paths match your dataset structure.

4. Training the Model

To train the smoking detection model:

python training/train_smoking.py


The script performs:

Training using YOLO11n

20 epochs (configurable)

Image size 640

Evaluation on the validation set

Saving the best model to:

runs/detect/train*/weights/best.pt


Training was conducted on CPU in this project, but GPU is recommended for faster performance.

5. Running Inference on a Video

To run inference on a custom video file:

python training/detect_video.py


The script will:

Load the trained model

Process the input video

Display results in a window

Save the output video inside:

runs/predict/


The detection video retains the original frame size to avoid distortion or zooming.

6. Exporting the Model to ONNX

To prepare the trained model for mobile or cross-platform deployment:

python training/export_onnx.py


The exported model will be saved as:

runs/detect/train*/weights/best.onnx


This ONNX model can later be used with ONNX Runtime Mobile, NNAPI, CoreML conversion, or other inference runtimes.

7. Mobile App Integration (Planned)

The repository includes an Android project folder where future integration will occur.
The planned workflow includes:

Converting the ONNX model for mobile inference

Implementing real-time camera input

Overlaying model predictions on screen

Triggering application-level alerts for smoking detection events

This portion of the system is still under development.

8. Notes and Recommendations

Training on CPU is significantly slower; using a GPU is strongly recommended.

Increasing epochs beyond 20 may improve accuracy, depending on dataset quality.

Dataset balancing is important for improving model recall.

When running inference on video, ensure the correct codec and file path are used.

9. License

This project is intended for educational and research purposes.
You may modify or extend the code to suit your own applications.
