import cv2
from ultralytics import YOLO
from pathlib import Path

# ================= CẤU HÌNH =================
MODEL_PATH = r"F:\BACH\GROUP_PROJECT\training\runs\detect\train6\weights\best.pt"

# ẢNH INPUT & OUTPUT
INPUT_IMAGE_PATH = r"C:\Users\Admin\Documents\Downloads\Smoke_(34942422652).jpg"
OUTPUT_IMAGE_PATH = r"F:\BACH\GROUP_PROJECT\OUTPUT\output_image_smoking.jpg"

CONF_THRES = 0.25
DEVICE = 0  # GPU 0, muốn CPU thì dùng "cpu"
MAX_W, MAX_H = 1280, 720  # kích thước tối đa khi hiển thị
# ===========================================


def main():
    print("[INFO] Loading model...")
    model = YOLO(MODEL_PATH)

    print("[INFO] Reading image:", INPUT_IMAGE_PATH)
    img = cv2.imread(INPUT_IMAGE_PATH)
    if img is None:
        print("[ERROR] Không mở được ảnh!")
        return

    # Chạy YOLO
    results = model.predict(img, conf=CONF_THRES, device=DEVICE, verbose=False)
    annotated = results[0].plot()  # ảnh full size

    # Lưu ảnh full size
    Path(OUTPUT_IMAGE_PATH).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(OUTPUT_IMAGE_PATH, annotated)
    print("[INFO] Saved output to:", OUTPUT_IMAGE_PATH)

    # Thu nhỏ ảnh để hiển thị cho khỏi bị zoom
    display = annotated.copy()
    h, w = display.shape[:2]
    scale = min(MAX_W / w, MAX_H / h, 1.0)
    if scale < 1.0:
        new_w, new_h = int(w * scale), int(h * scale)
        display = cv2.resize(display, (new_w, new_h))

    cv2.imshow("Smoking detection - IMAGE", display)
    print("[INFO] Nhấn phím bất kỳ để đóng...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
