import cv2
from ultralytics import YOLO
from pathlib import Path

# ================= CẤU HÌNH =================
MODEL_PATH = r"F:\BACH\GROUP_PROJECT\training\runs\detect\train6\weights\best.pt"

SAVE_VIDEO = False  # nếu muốn lưu lại video webcam thì đặt True
OUTPUT_VIDEO_PATH = r"F:\BACH\GROUP_PROJECT\OUTPUT\output_webcam_smoking.mp4"

CONF_THRES = 0.25
DEVICE = 0  # GPU 0, muốn CPU thì dùng "cpu"
# ===========================================


def main():
    print("[INFO] Loading model...")
    model = YOLO(MODEL_PATH)

    print("[INFO] Opening WEBCAM (source=0)...")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Không mở được webcam!")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = None

    cv2.namedWindow("Smoking detection - WEBCAM", cv2.WINDOW_NORMAL)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[INFO] Webcam không còn frame.")
            break

        # nếu muốn lưu video webcam
        if SAVE_VIDEO and out is None:
            h, w = frame.shape[:2]
            Path(OUTPUT_VIDEO_PATH).parent.mkdir(parents=True, exist_ok=True)
            out = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, fps, (w, h))
            print(f"[INFO] Webcam video size = {w}x{h}, FPS = {fps}")

        results = model.predict(frame, conf=CONF_THRES, device=DEVICE, verbose=False)
        annotated = results[0].plot()

        cv2.imshow("Smoking detection - WEBCAM", annotated)
        if out is not None:
            out.write(annotated)

        # nhấn q để thoát
        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("[INFO] Người dùng ấn q, thoát webcam.")
            break

    cap.release()
    if out is not None:
        out.release()
        print("[INFO] Saved webcam video to:", OUTPUT_VIDEO_PATH)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
