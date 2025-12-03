from ultralytics import YOLO

def main():
    model = YOLO("yolo11n.pt")  # model nhỏ nhất, nhanh nhất

    model.train(
        data="smoking.yaml",
        epochs=20,
        imgsz=640,
        batch=8,
        device="cpu",          # QUAN TRỌNG: bạn không có GPU
        pretrained=True,
        workers=1              # tránh lỗi Windows
    )

    # Export sang ONNX để đưa vào app Android
    model.export(format="onnx")

if __name__ == "__main__":
    main()
