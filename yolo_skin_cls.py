import os
import mlflow
import numpy as np
from ultralytics import YOLO
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import torch
from torchvision import transforms

# تعديل مسار MLflow ليكون داخل المشروع
os.environ["MLFLOW_TRACKING_URI"] = "file:///D:/grad_project/mlruns"
mlflow.set_experiment("YOLOv8_skin_cls_experiment")

# تعديل مسار البيانات
data_path = "D:/grad_project/skin_dataset"
val_dir = os.path.join(data_path, "val")
class_names = sorted(os.listdir(val_dir))

with mlflow.start_run(run_name="YOLOv8n_cls_run"):

    model = YOLO("yolov8n-cls.pt")
    results = model.train(
        data=data_path,
        epochs=20,
        imgsz=224,
        project="D:/grad_project/skin_yolo_project",
        name="yolov8_skin_cls42",
        pretrained=True,
        patience=100,
        batch=32,
        fliplr=0.5,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        erasing=0.4,
        mixup=0.1,
        mosaic=0.0,
        copy_paste=0.0,
        augment=True
    )

    y_true, y_pred = [], []
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    for label_idx, class_name in enumerate(class_names):
        class_dir = os.path.join(val_dir, class_name)
        for img_file in os.listdir(class_dir):
            if img_file.lower().endswith((".jpg", ".jpeg", ".png")):
                img_path = os.path.join(class_dir, img_file)
                img = Image.open(img_path).convert("RGB")
                img_tensor = transform(img).unsqueeze(0)

                with torch.no_grad():
                    probs = model(img_tensor, verbose=False)[0].probs
                    pred_label = torch.argmax(probs.data).item()

                y_true.append(label_idx)
                y_pred.append(pred_label)

    report_dict = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    report_str = classification_report(y_true, y_pred, target_names=class_names)

    mlflow.log_metric("accuracy", float(report_dict["accuracy"]))
    mlflow.log_metric("f1_score", float(np.mean([report_dict[c]["f1-score"] for c in class_names])))
    mlflow.log_metric("precision", float(np.mean([report_dict[c]["precision"] for c in class_names])))
    mlflow.log_metric("recall", float(np.mean([report_dict[c]["recall"] for c in class_names])))

    for cls in class_names:
        mlflow.log_metric(f"precision_{cls}", float(report_dict[cls]["precision"]))
        mlflow.log_metric(f"recall_{cls}", float(report_dict[cls]["recall"]))
        mlflow.log_metric(f"f1_score_{cls}", float(report_dict[cls]["f1-score"]))

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=class_names, yticklabels=class_names, cmap="Blues")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    cm_path = "D:/grad_project/confusion_matrix.png"
    plt.savefig(cm_path)
    mlflow.log_artifact(cm_path)

    report_txt_path = "D:/grad_project/classification_report.txt"
    with open(report_txt_path, "w") as f:
        f.write(report_str)
    mlflow.log_artifact(report_txt_path)

    print("Training, evaluation, and MLflow logging complete.")
