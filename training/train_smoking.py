from ultralytics import YOLO

def main():
    model = YOLO("yolo11s.pt")

    model.train(
        data="../data.yaml",
        epochs=60,        # giữ như bạn đang chạy
        imgsz=640,
        batch=8,          # <<< GIẢM từ 16 xuống 8
        device=0,
        workers=4,
        pretrained=True,
        cache=False,      # RAM bạn hơi hạn chế, tắt cache ảnh
        close_mosaic=10
    )

if __name__ == "__main__":
    main()
