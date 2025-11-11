import sys
import cv2
import numpy as np
import torch
from PIL import Image
from ultralytics import YOLO, SAM
from ultralytics.engine.results import Boxes


# --- 1. LOAD MODELS ---
def load_models():
    """Loads YOLOv12 and SAM models."""
    print("Loading models... This may take a moment.")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    yolo_detector = YOLO("/home/paul/Development/src/RealSense/yolo12x.pt").to(device)
    sam_segmenter = SAM("sam_l.pt").to(device)

    print("Models loaded successfully.")
    return yolo_detector, sam_segmenter


# --- 2. RUN PIPELINE ---
def run_perception_pipeline(image_path: str):
    """Detects all objects using YOLOv12, then refines bounding boxes using SAM masks."""
    yolo, sam = load_models()

    # Load image
    try:
        image_pil = Image.open(image_path).convert("RGB")
        image_cv = cv2.imread(image_path)
    except FileNotFoundError:
        print(f"Error: Image not found at {image_path}")
        return

    # --- YOLO DETECTION ---
    print("Running YOLOv12 detection...")
    yolo_results = yolo.predict(image_pil, verbose=False)
    result = yolo_results[0]

    if len(result.boxes) == 0:
        print("YOLOv12 found no objects.")
        cv2.imshow("YOLOv12 + SAM Results (No Detections)", image_cv)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return

    print(f"YOLOv12 found {len(result.boxes)} objects.")

    # --- SAM SEGMENTATION ---
    print("Running SAM on detected objects...")
    sam_results = sam.predict(image_pil, bboxes=result.boxes.xyxy, verbose=False)
    if not sam_results or not sam_results[0].masks:
        print("SAM failed to produce masks.")
        return

    masks_obj = sam_results[0].masks
    masks_np = masks_obj.data
    if isinstance(masks_np, torch.Tensor):
        masks_np = masks_np.cpu().numpy()
    if masks_np.ndim == 2:
        masks_np = masks_np[np.newaxis, ...]

    # --- COMPUTE SAM BOXES ---
    sam_boxes = []
    for mask in masks_np:
        ys, xs = np.where(mask > 0)
        if len(xs) == 0 or len(ys) == 0:
            sam_boxes.append([0, 0, 0, 0])
            continue
        x1, y1 = xs.min(), ys.min()
        x2, y2 = xs.max(), ys.max()
        sam_boxes.append([x1, y1, x2, y2])
    sam_boxes = np.array(sam_boxes, dtype=float)

    if sam_boxes.size == 0:
        print("No valid SAM boxes.")
        return

    # --- EXTRACT YOLO BOX DATA ---
    yolo_boxes = result.boxes.xyxy.cpu().numpy()
    yolo_conf = result.boxes.conf.cpu().numpy()
    yolo_cls = result.boxes.cls.cpu().numpy()

    # --- IOU FUNCTION ---
    def iou_boxes(a, b):
        iou = np.zeros((len(a), len(b)))
        for i in range(len(a)):
            x1a, y1a, x2a, y2a = a[i]
            area_a = (x2a - x1a + 1) * (y2a - y1a + 1)
            for j in range(len(b)):
                x1b, y1b, x2b, y2b = b[j]
                inter_x1, inter_y1 = max(x1a, x1b), max(y1a, y1b)
                inter_x2, inter_y2 = min(x2a, x2b), min(y2a, y2b)
                if inter_x2 < inter_x1 or inter_y2 < inter_y1:
                    continue
                inter_area = (inter_x2 - inter_x1 + 1) * (inter_y2 - inter_y1 + 1)
                area_b = (x2b - x1b + 1) * (y2b - y1b + 1)
                union = area_a + area_b - inter_area
                iou[i, j] = inter_area / union if union > 0 else 0
        return iou

    iou_mat = iou_boxes(sam_boxes, yolo_boxes)

    # --- ASSIGN LABELS/CONFIDENCE ---
    confs_out, classes_out = np.zeros((len(sam_boxes), 1)), np.zeros((len(sam_boxes), 1))

    for i in range(len(sam_boxes)):
        best_j = int(np.argmax(iou_mat[i]))
        best_iou = float(iou_mat[i, best_j])

        if best_iou >= 0.1:
            confs_out[i, 0] = yolo_conf[best_j]
            classes_out[i, 0] = yolo_cls[best_j]
        else:
            # fallback → nearest-center
            sx1, sy1, sx2, sy2 = sam_boxes[i]
            scx, scy = (sx1 + sx2) / 2, (sy1 + sy2) / 2
            ycx = (yolo_boxes[:, 0] + yolo_boxes[:, 2]) / 2
            ycy = (yolo_boxes[:, 1] + yolo_boxes[:, 3]) / 2
            dists = np.sqrt((ycx - scx) ** 2 + (ycy - scy) ** 2)
            nearest = int(np.argmin(dists))
            confs_out[i, 0] = yolo_conf[nearest]
            classes_out[i, 0] = yolo_cls[nearest]

    # --- BUILD NEW BOX TENSOR ---
    sam_boxes_full = np.hstack([sam_boxes, confs_out, classes_out])
    device_for_boxes = result.boxes.data.device
    sam_boxes_tensor = torch.tensor(sam_boxes_full, dtype=torch.float32, device=device_for_boxes)

    # Replace YOLO boxes with SAM-refined ones
    result.boxes = Boxes(sam_boxes_tensor, orig_shape=result.orig_shape)
    result.masks = sam_results[0].masks

    # --- VISUALIZATION ---
    print("Drawing final results...")
    annotated_frame = result.plot()

    cv2.imshow("YOLOv12 + SAM (Refined Boxes)", annotated_frame)
    print("\nDone! Press 'q' to close the window.")
    while True:
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()


# --- MAIN ---
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python detection_pipeline.py <path_to_image>")
        sys.exit(1)
    image_file = sys.argv[1]
    run_perception_pipeline(image_file)
