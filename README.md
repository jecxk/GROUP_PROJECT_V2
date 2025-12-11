# GROUP_PROJECT_V2 – Smoking Detection with YOLO11

This repository contains our upgraded **smoking detection** project using **Ultralytics YOLO11**.  
It can detect:

- `cigarette`
- `cigar`
- `smoke`

and run on:

- single **images**
- **video files**
- **webcam / live camera**

The project is trained and tested on an NVIDIA **GTX 1660 SUPER**, but it also works on CPU (slower).

---

## 1. Project Structure

```text
GROUP_PROJECT_V2/
├─ training/
│  ├─ train_smoking.py        # script to train YOLO11 on our dataset
│  ├─ detect_image.py         # detect smoking on a single image
│  ├─ detect_video.py         # detect smoking in a video file
│  ├─ detect_webcam.py        # detect smoking from webcam (live)
│  ├─ export_onnx.py          # export trained model to ONNX (optional, for mobile/app)
│  └─ ... (internal YOLO/Ultralytics files)
│
├─ weights/
│  └─ smoking_yolo11s_best.pt # fine-tuned YOLO11s model (smoking detector)
│
├─ data.yaml                  # YOLO dataset config (paths to train/val images & labels)
├─ README.md                  # this file
└─ .gitignore
Note:
The dataset itself is NOT included in this repo (to keep it small).
You need to place your dataset on your own machine and update data.yaml accordingly.

2. Requirements
Python 3.9+ (project was tested with Python 3.9)

Recommended:

NVIDIA GPU + CUDA 12.1 (for fast training & inference)

Libraries (installed via pip):

bash
Copy code
pip install -U ultralytics opencv-python
# If you have NVIDIA GPU + CUDA 12.1:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
# If you only have CPU:
# pip install torch torchvision torchaudio   # (default CPU wheels from PyPI)
You can also set up a virtual environment if you prefer.

3. Getting Started
3.1. Clone this repository
bash
Copy code
git clone https://github.com/jecxk/GROUP_PROJECT_V2.git
cd GROUP_PROJECT_V2
3.2. (Optional) Check Python & CUDA
bash
Copy code
python -c "import torch; print(torch.__version__); print('cuda?', torch.cuda.is_available()); print('gpus:', torch.cuda.device_count())"
If cuda? True appears, you are using GPU. Otherwise, the scripts will still work on CPU (slower).

4. Using the Pre-trained Model (Detection Only)
The repository already includes a trained model:

text
Copy code
weights/smoking_yolo11s_best.pt
You can use it directly without retraining.

All detection scripts are located in the training/ folder.
Run them from inside that folder.

bash
Copy code
cd training
4.1. Detect on a single image
Open training/detect_image.py

Edit these lines:

python
Copy code
INPUT_IMAGE_PATH = r"FULL_PATH_TO_YOUR_IMAGE.jpg"
OUTPUT_IMAGE_PATH = r"F:\BACH\GROUP_PROJECT\OUTPUT\output_image_smoking.jpg"
Run:

bash
Copy code
cd training
python detect_image.py
A window will pop up showing the image with bounding boxes.

The output image is saved to OUTPUT_IMAGE_PATH.

4.2. Detect on a video file
Place your test video somewhere on your machine.

Open training/detect_video.py and edit:

python
Copy code
INPUT_VIDEO_PATH = r"FULL_PATH_TO_YOUR_VIDEO.mp4"
OUTPUT_VIDEO_PATH = r"F:\BACH\GROUP_PROJECT\OUTPUT\output_video_smoking.mp4"
Run:

bash
Copy code
cd training
python detect_video.py
A window will show the annotated video.

Press q to stop early.

The processed video is saved to OUTPUT_VIDEO_PATH.

4.3. Detect from webcam (live)
Open training/detect_webcam.py and check:

python
Copy code
SAVE_VIDEO = False  # set True if you want to record webcam output
OUTPUT_VIDEO_PATH = r"F:\BACH\GROUP_PROJECT\OUTPUT\output_webcam_smoking.mp4"
Run:

bash
Copy code
cd training
python detect_webcam.py
The script opens the default webcam (source=0).

Press q to exit.

If SAVE_VIDEO = True, the recorded annotated stream is saved to OUTPUT_VIDEO_PATH.

5. Training the Model (Optional)
If you want to retrain / fine-tune the smoking detector, you can use train_smoking.py.

5.1. Dataset Format
We use the standard YOLO format:

text
Copy code
your_dataset_root/
├─ train/
│  ├─ images/
│  └─ labels/    # .txt files, YOLO format
└─ valid/
   ├─ images/
   └─ labels/
Each label .txt file contains lines like:

text
Copy code
0 x_center y_center width height
1 ...
2 ...
Where class IDs:

0 → cigarette

1 → cigar

2 → smoke

(all normalized to 0–1 as in YOLO).

5.2. Update data.yaml
Example data.yaml:

yaml
Copy code
path: C:\BACH\datasets\smoking\alldatav1_relabel   # root of dataset on your machine
train: train/images
val: valid/images

names:
  0: cigarette
  1: cigar
  2: smoke
Change the path: to match your local dataset location.

5.3. Run training
From inside training/:

bash
Copy code
cd training
python train_smoking.py
The script (simplified) does:

python
Copy code
from ultralytics import YOLO

def main():
    model = YOLO("yolo11s.pt")  # YOLO11s base model
    model.train(
        data="data.yaml",
        epochs=60,
        imgsz=640,
        batch=8,
        device=0,   # use "cpu" if no GPU
        workers=2,
    )
    model.export(format="onnx")  # optional, for app deployment

if __name__ == "__main__":
    main()
Training outputs (runs, logs, etc.) are stored in training/runs/ and are ignored by git.

6. Exporting to ONNX (for Apps / Mobile)
If you want to use the model in a mobile app or another framework, you can export the fine-tuned weights to ONNX.

Open training/export_onnx.py and edit:

python
Copy code
MODEL_PATH = r"F:\BACH\GROUP_PROJECT\training\runs\detect\train6\weights\best.pt"
EXPORT_DIR = r"F:\BACH\GROUP_PROJECT\EXPORT"
MODEL_NAME = "smoking_yolo11s"
Run:

bash
Copy code
cd training
python export_onnx.py
You will get something like:

text
Copy code
EXPORT/smoking_yolo11s/smoking_yolo11s.onnx
You can also create a simple labels.txt:

text
Copy code
cigarette
cigar
smoke
for use in your app.

7. Troubleshooting
[ERROR] Không mở được video!

Check that INPUT_VIDEO_PATH is correct.

Try copying the video into the repo, e.g. training/videos/test.mp4, and update the path.

CUDA / GPU not detected

Run:

bash
Copy code
python -c "import torch; print(torch.__version__); print('cuda?', torch.cuda.is_available())"
If cuda? False, either:

install the correct CUDA-enabled torch build, or

change device=0 → device='cpu' in the scripts.

CUDA out of memory while training

Reduce batch size in train_smoking.py (e.g. batch=4 or batch=2).

Make sure no other heavy GPU tasks are running.

8. Contact / Notes
Repository owner: jecxk

Model: YOLO11s fine-tuned for smoking detection (3 classes: cigarette, cigar, smoke).
