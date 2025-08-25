Person Detection in Aerial Images (VisDrone + YOLOv5)

This repository implements an object detection pipeline to identify persons in aerial images (e.g., disaster-response scenarios such as landslides and collapses), supporting search & rescue operations.

Scope: data preparation, label formatting (YOLO), training configuration, selected hyperparameters, evaluation (Precision/Recall/mAP), and discussion of limitations/next steps.
Note: this MVP uses a reduced subset of VisDrone 2019 to keep training time feasible.

1) Problem Statement

Goal: detect persons in drone imagery to assist emergency response.

Challenges: low resolution, irregular lighting, occlusions, small objects.

Approach: YOLOv5 (real-time, good small-object performance).

2) Dataset

Source: VisDrone 2019 (public aerial imagery with bounding-box annotations).

Classes: multiple objects; we filter to person to reduce noise.

Split: 80% train / 20% test (YOLOv5 internal val during training).

Subset: a reduced portion of VisDrone was used to speed up experiments.

Labels: normalized to YOLO format class x_center y_center width height (all normalized to image size). Malformed labels were checked and filtered.

3) Data Preparation

Keep only valid purchases (oops, not applicable) → keep valid samples, normalize bboxes to YOLO format.

Augmentations (YOLOv5): flips, mosaic, scaling.

Focus: only the person class.

(Optional) Script to filter original VisDrone labels to a person-only subset.

Expected structure:
dataset/
  ├─ images/
  │   ├─ train/    *.jpg
  │   └─ val/      *.jpg
  └─ labels/
      ├─ train/    *.txt  # YOLO: class cx cy w h (normalized)
      └─ val/      *.txt

4) Modeling & Training

Model: YOLOv5 (baseline).

Selected hyperparameters (examples used):

warmup_epochs = 3.0 — stabilizes early updates.

hsv_h = 0.015, hsv_s = 0.7, hsv_v = 0.4 — lighting/color variability.

mosaic = 1.0 — strong augmentation for small/medium datasets.

epochs = 5 — short run for MVP scope.

(Also experimented with image size and learning rate.)

Potential improvements: try yolov5m.pt (larger backbone), train longer, tune anchors, experiment with advanced augmentation schedules and ensembling.

5) Evaluation

Metrics (object detection standard):

Precision (good: > 0.7) — few false positives.

Recall (good: > 0.7) — few missed persons.

mAP@50 (good: > 0.5) — balances precision/recall at IoU ≥ 0.5.

mAP@50–95 (good: > 0.4) — stricter, across IoU 0.5–0.95.

Observed outcome (MVP):

Performance below expectations on the small subset; difficulty detecting persons in some images.

No clear overfitting; the constraint appears to be data-limited generalization.

6) How to Run
6.1 Environment
   pip install -r requirements.txt

6.2 Train (example — adjust paths)

If you use the Ultralytics package:
yolo detect train \
  model=yolov5s.pt \
  data=data/visdrone-person.yaml \
  imgsz=640 \
  epochs=5 \
  batch=16

If you cloned the yolov5 repo:
python train.py \
  --weights yolov5s.pt \
  --data data/visdrone-person.yaml \
  --img 640 \
  --batch 16 \
  --epochs 5

6.3 Validate / Inference (examples)
# Ultralytics
yolo detect val model=runs/detect/train/weights/best.pt data=data/visdrone-person.yaml imgsz=640
yolo detect predict model=runs/detect/train/weights/best.pt source=path/to/test/images imgsz=640

# yolov5 repo
python val.py --weights runs/train/exp/weights/best.pt --data data/visdrone-person.yaml --img 640
python detect.py --weights runs/train/exp/weights/best.pt --source path/to/test/images --img 640

7) Known Limitations

Small training subset → likely data-limited generalization.

Aerial imagery complexity (scale, occlusions, variable lighting).

Person-only focus may ignore helpful context from other classes.

8) Roadmap

Increase dataset size/representativeness; longer training.

Test larger backbones (e.g., yolov5m.pt) and anchor tuning.

Explore ensembles for higher accuracy.

Curate class-balanced samples and robust augmentation schedules.

9) Reproducibility

Fix random seeds when possible (PyTorch/Ultralytics flags).

Log runs with the default YOLO logger (or integrate Weights & Biases).

10) Citation & License

Cite VisDrone 2019 as per the dataset’s official instructions.

For educational/research purposes. Respect YOLOv5/Ultralytics and VisDrone licenses.
