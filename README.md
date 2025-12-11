# GROUP_PROJECT_V2 – Smoking Detection with YOLO11

This repository contains a YOLO11-based system for detecting **smoking behaviour** in images and videos.  
The model is fine-tuned to recognise three classes:

- `cigarette`
- `cigar`
- `smoke`

It supports three modes:

- Single **image** detection
- **Video file** detection
- **Webcam / live camera** detection

---

## 1. Project Structure

```text
GROUP_PROJECT_V2/
├─ training/
│  ├─ train_smoking.py        # train YOLO11 on the smoking dataset
│  ├─ detect_image.py         # run detection on a single image
│  ├─ detect_video.py         # run detection on a video file
│  ├─ detect_webcam.py        # run detection from webcam (live)
│  ├─ export_onnx.py          # export the trained model to ONNX (optional)
│  └─ ...                     # internal Ultralytics / helper files
│
├─ weights/
│  └─ smoking_yolo11s_best.pt # fine-tuned YOLO11s model for smoking detection
│
├─ data.yaml                  # YOLO dataset configuration (paths, class names)
├─ .gitignore                 # ignore datasets, runs, outputs, etc.
└─ README.md                  # this documentation
Note
The dataset itself is not included in this repository to keep the size small.
Each user should place the dataset on their own machine and update data.yaml accordingly.

2. Requirements
Python 3.9+ (tested on 3.9)

Optional but recommended:

NVIDIA GPU with CUDA support (training & inference are much faster)

2.1. Install dependencies
From the root of the repo:

bash
Copy code
pip install -U ultralytics opencv-python
If you have an NVIDIA GPU with CUDA 12.1:

bash
Copy code
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
If you only use CPU, you can install the default CPU wheels:

bash
Copy code
pip install torch torchvision torchaudio
3. Getting Started
3.1. Clone the repository
bash
Copy code
git clone https://github.com/jecxk/GROUP_PROJECT_V2.git
cd GROUP_PROJECT_V2
3.2. Verify PyTorch & CUDA (optional)
bash
Copy code
python -c "import torch; print(torch.__version__); print('cuda?', torch.cuda.is_available()); print('gpus:', torch.cuda.device_count())"
If cuda? True, the scripts will use the GPU (device=0).
If not, you can switch to CPU by setting device="cpu" in the detection / training scripts.

4. Using the Pre-trained Model
The repository already includes a fine-tuned model:

text
Copy code
weights/smoking_yolo11s_best.pt
You can use it directly for inference without retraining.

All detection scripts are located in training/.
Run them from inside that folder:

bash
Copy code
cd training
4.1. Detect on an image
Open training/detect_image.py.

Update the input and output paths:

python
Copy code
INPUT_IMAGE_PATH = r"PATH_TO_YOUR_IMAGE.jpg"
OUTPUT_IMAGE_PATH = r"OUTPUT/output_image_smoking.jpg"
Run:

bash
Copy code
python detect_image.py
A window will pop up showing the image with bounding boxes.

The processed image is saved to OUTPUT_IMAGE_PATH.

4.2. Detect on a video file
Place a test video somewhere on your system.

Open training/detect_video.py and edit:

python
Copy code
INPUT_VIDEO_PATH = r"PATH_TO_YOUR_VIDEO.mp4"
OUTPUT_VIDEO_PATH = r"OUTPUT/output_video_smoking.mp4"
Run:

bash
Copy code
python detect_video.py
A window will show the annotated video.

Press q to stop early.

The processed video is saved to OUTPUT_VIDEO_PATH.

4.3. Detect from webcam (live camera)
Open training/detect_webcam.py.

Optionally enable saving the webcam stream:

python
Copy code
SAVE_VIDEO = False          # set True to record the webcam output
OUTPUT_VIDEO_PATH = r"OUTPUT/output_webcam_smoking.mp4"
Run:

bash
Copy code
python detect_webcam.py
The default webcam (device 0) will be opened.

Press q to exit.

If SAVE_VIDEO = True, the annotated stream is saved to OUTPUT_VIDEO_PATH.

5. Training the Model (Optional)
If you want to retrain or further fine-tune the model, you can use train_smoking.py.

5.1. Dataset format
The dataset follows the standard YOLO detection format:

text
Copy code
<dataset_root>/
├─ train/
│  ├─ images/
│  └─ labels/         # .txt files in YOLO format
└─ valid/
   ├─ images/
   └─ labels/
Each label .txt file contains lines:

text
Copy code
<class_id> <x_center> <y_center> <width> <height>
(all values normalised to 0–1).

Class IDs in this project:

0 → cigarette

1 → cigar

2 → smoke

5.2. Configure data.yaml
Example:

yaml
Copy code
path: C:/datasets/smoking/alldatav1_relabel   # change to your dataset root
train: train/images
val: valid/images

names:
  0: cigarette
  1: cigar
  2: smoke
Update the path: field to match the dataset location on your machine.

5.3. Run training
From inside training/:

bash
Copy code
python train_smoking.py
A simplified version of the training script:

python
Copy code
from ultralytics import YOLO

def main():
    model = YOLO("yolo11s.pt")  # base YOLO11s model
    model.train(
        data="data.yaml",
        epochs=60,
        imgsz=640,
        batch=8,
        device=0,   # use "cpu" if you don't have a GPU
        workers=2,
    )

if __name__ == "__main__":
    main()
Training outputs (metrics, plots, checkpoints) are stored in training/runs/ and are ignored by git.

6. Exporting to ONNX (for apps / deployment)
To integrate the model into a mobile app or another runtime, you can export it to ONNX.

Open training/export_onnx.py.

Set the paths:

python
Copy code
MODEL_PATH = r"training/runs/detect/train6/weights/best.pt"  # or another checkpoint
EXPORT_DIR = r"EXPORT"
MODEL_NAME = "smoking_yolo11s"
Run:

bash
Copy code
cd training
python export_onnx.py
This will create something like:

text
Copy code
EXPORT/smoking_yolo11s/smoking_yolo11s.onnx
You can also create labels.txt for external applications:

text
Copy code
cigarette
cigar
smoke
7. Troubleshooting
[ERROR] Không mở được video!

Check that INPUT_VIDEO_PATH points to an existing file.

Try moving the video into the repository (e.g. training/videos/test.mp4) and updating the path.

CUDA not detected (cuda? False)

Make sure the correct CUDA-enabled PyTorch build is installed (or switch device to "cpu").

CUDA out of memory during training

Lower the batch size in train_smoking.py (batch=4 or batch=2).

Close other GPU-heavy applications.

8. Contact
Repository owner: @jecxk

Model: YOLO11s, fine-tuned for 3-class smoking detection (cigarette, cigar, smoke).
