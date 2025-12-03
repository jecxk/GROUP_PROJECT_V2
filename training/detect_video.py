import cv2
from ultralytics import YOLO

# ==== CẤU HÌNH ĐƯỜNG DẪN ====
MODEL_PATH = r"C:\Users\Admin\smoking-detection\runs\detect\train3\weights\best.pt"
INPUT_VIDEO_PATH = r"C:\Users\Admin\Downloads\4310184-hd_1920_1080_24fps.mp4"
OUTPUT_PATH = r"C:\Users\Admin\smoking-project\training\output_smoking.mp4"


def main():
    print("[INFO] Loading model...")
    model = YOLO(MODEL_PATH)

    print("[INFO] Opening video:", INPUT_VIDEO_PATH)
    cap = cv2.VideoCapture(INPUT_VIDEO_PATH)
    if not cap.isOpened():
        print("[ERROR] Không mở được video input!")
        return

    # Lấy thông tin video gốc
    fps = cap.get(cv2.CAP_PROP_FPS) or 25

    # VideoWriter sẽ tạo sau khi đọc frame đầu tiên (lấy đúng width, height)
    out = None
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    # Cửa sổ hiển thị cho phép resize tự do
    cv2.namedWindow("Smoking detection", cv2.WINDOW_NORMAL)

    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[INFO] Hết video.")
            break

        frame_idx += 1

        # Khởi tạo VideoWriter với đúng kích thước gốc
        if out is None:
            h, w = frame.shape[:2]
            print(f"[INFO] Video size = {w}x{h}, FPS = {fps}")
            out = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (w, h))

        # Chạy YOLO trên frame gốc
        results = model.predict(
            frame,
            conf=0.25,       # có thể chỉnh thấp hơn nếu muốn nhiều bbox hơn
            verbose=False
        )

        # Vẽ bbox lên frame gốc (giữ nguyên size gốc)
        annotated_frame = results[0].plot()

        # Ghi đúng frame gốc (không resize) vào file output
        out.write(annotated_frame)

        # ----- HIỂN THỊ LÊN MÀN HÌNH (THU NHỎ CHO ĐỠ "ZOOM") -----
        display_h = 540
        display_w = int(annotated_frame.shape[1] * display_h / annotated_frame.shape[0])
        display_frame = cv2.resize(annotated_frame, (display_w, display_h))

        cv2.imshow("Smoking detection", display_frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("[INFO] Bạn bấm q -> thoát sớm.")
            break

    cap.release()
    if out is not None:
        out.release()
    cv2.destroyAllWindows()
    print(f"[DONE] Video output đã lưu tại:\n{OUTPUT_PATH}")


if __name__ == "__main__":
    main()
