"""
kserve_client.py
================
Standalone client to test the KServe YOLOv8n endpoint.

Usage:
    python kserve_client.py --image path/to/image.jpg
    python kserve_client.py --demo          # uses a synthetic image
"""
import argparse
import json
import subprocess
import sys
import numpy as np
import requests
from pathlib import Path


# ── Helpers ───────────────────────────────────────────────────────────────────

def get_cluster_endpoint(namespace: str, model_name: str) -> tuple[str, str]:
    """Returns (base_url, host_header) for the KServe endpoint."""
    try:
        ingress_ip = subprocess.check_output(
            ["minikube", "ip"], text=True
        ).strip()
        port = subprocess.check_output(
            ["kubectl", "-n", "istio-system", "get", "svc",
             "istio-ingressgateway", "-o",
             "jsonpath={.spec.ports[?(@.name==\"http2\")].nodePort}"],
            text=True,
        ).strip()
        host = subprocess.check_output(
            ["kubectl", "-n", namespace, "get", "inferenceservice", model_name,
             "-o", "jsonpath={.status.url}"],
            text=True,
        ).strip().replace("http://", "")
        base_url = f"http://{ingress_ip}:{port}"
        return base_url, host
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Could not get cluster endpoint: {e}")
        sys.exit(1)


def preprocess_image(image_path: str, size: int = 640) -> np.ndarray:
    """Loads and preprocesses an image to (1, 3, size, size) float32 [0, 1]."""
    from PIL import Image
    img = Image.open(image_path).convert("RGB").resize((size, size))
    arr = np.array(img, dtype=np.float32) / 255.0  # HWC [0,1]
    arr = arr.transpose(2, 0, 1)                    # CHW
    return arr[np.newaxis, ...]                      # NCHW


def make_payload(data: np.ndarray, input_name: str = "images") -> dict:
    return {
        "inputs": [{
            "name": input_name,
            "shape": list(data.shape),
            "datatype": "FP32",
            "data": data.flatten().tolist(),
        }]
    }


def parse_detections(response: dict, conf_thresh: float = 0.25) -> list[dict]:
    """
    Parse raw YOLOv8 output (1, 84, 8400) into detection dicts.
    Format: 84 = 4 (box xywh) + 80 (class scores)
    """
    outputs = response.get("outputs", [])
    if not outputs:
        return []

    raw = np.array(outputs[0]["data"], dtype=np.float32)
    raw = raw.reshape(1, 84, 8400)[0]  # (84, 8400)

    boxes  = raw[:4, :].T    # (8400, 4) — xywh normalized
    scores = raw[4:, :].T    # (8400, 80)

    detections = []
    for i in range(scores.shape[0]):
        cls_id   = int(np.argmax(scores[i]))
        conf     = float(scores[i, cls_id])
        if conf < conf_thresh:
            continue
        x, y, w, h = boxes[i]
        detections.append({
            "class_id": cls_id,
            "confidence": round(conf, 4),
            "box_xywh": [round(float(v), 4) for v in [x, y, w, h]],
        })

    detections.sort(key=lambda d: d["confidence"], reverse=True)
    return detections[:20]  # top-20


COCO_CLASSES = [
    "person","bicycle","car","motorcycle","airplane","bus","train","truck","boat",
    "traffic light","fire hydrant","stop sign","parking meter","bench","bird","cat",
    "dog","horse","sheep","cow","elephant","bear","zebra","giraffe","backpack",
    "umbrella","handbag","tie","suitcase","frisbee","skis","snowboard","sports ball",
    "kite","baseball bat","baseball glove","skateboard","surfboard","tennis racket",
    "bottle","wine glass","cup","fork","knife","spoon","bowl","banana","apple",
    "sandwich","orange","broccoli","carrot","hot dog","pizza","donut","cake","chair",
    "couch","potted plant","bed","dining table","toilet","tv","laptop","mouse",
    "remote","keyboard","cell phone","microwave","oven","toaster","sink","refrigerator",
    "book","clock","vase","scissors","teddy bear","hair drier","toothbrush",
]


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="KServe YOLOv8n inference client")
    parser.add_argument("--image",     type=str,  default=None,         help="Path to input image")
    parser.add_argument("--demo",      action="store_true",             help="Use synthetic random image")
    parser.add_argument("--namespace", type=str,  default="yolo-project")
    parser.add_argument("--model",     type=str,  default="yolov8n")
    parser.add_argument("--conf",      type=float, default=0.25,        help="Confidence threshold")
    args = parser.parse_args()

    if not args.image and not args.demo:
        parser.print_help()
        sys.exit(1)

    # ── Prepare input ─────────────────────────────────────────────────────────
    if args.demo:
        print("[demo] Generating random 640×640 image...")
        data = np.random.rand(1, 3, 640, 640).astype(np.float32)
    else:
        print(f"[client] Loading: {args.image}")
        data = preprocess_image(args.image)

    print(f"[client] Input shape : {data.shape}")
    payload = make_payload(data)

    # ── Endpoint ──────────────────────────────────────────────────────────────
    base_url, host = get_cluster_endpoint(args.namespace, args.model)
    infer_url = f"{base_url}/v2/models/{args.model}/infer"
    print(f"[client] Endpoint    : {infer_url}")
    print(f"[client] Host header : {host}")

    # ── Send request ──────────────────────────────────────────────────────────
    print("[client] Sending inference request...")
    resp = requests.post(
        infer_url,
        headers={"Host": host, "Content-Type": "application/json"},
        json=payload,
        timeout=60,
    )
    resp.raise_for_status()
    result = resp.json()
    print(f"[client] HTTP {resp.status_code}")

    # ── Parse ─────────────────────────────────────────────────────────────────
    detections = parse_detections(result, conf_thresh=args.conf)
    print(f"\n{'='*55}")
    print(f" Detections (conf ≥ {args.conf}): {len(detections)}")
    print(f"{'='*55}")
    for d in detections:
        cls_name = COCO_CLASSES[d["class_id"]] if d["class_id"] < len(COCO_CLASSES) else str(d["class_id"])
        box = d["box_xywh"]
        print(f"  [{d['confidence']:.2%}]  {cls_name:<20}  box=[{box[0]:.3f},{box[1]:.3f},{box[2]:.3f},{box[3]:.3f}]")

    if not detections:
        print("  (no detections above threshold — expected for a random image)")
    print(f"{'='*55}\n")


if __name__ == "__main__":
    main()
