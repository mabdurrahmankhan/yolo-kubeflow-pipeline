from kfp import dsl, compiler
from kfp.dsl import Output, Input, Dataset, Model, Metrics, Artifact

YOLO_IMAGE = "yolo-kfp:v2"


@dsl.component(base_image=YOLO_IMAGE)
def get_coco128(dataset: Output[Dataset]):
    import shutil, urllib.request
    url = "https://ultralytics.com/assets/coco128.zip"
    tmp = "/tmp/coco128.zip"
    urllib.request.urlretrieve(url, tmp)
    # One file, so MinIO stores a single object the UI can open.
    shutil.copy(tmp, dataset.path)


@dsl.component(base_image=YOLO_IMAGE)
def export_onnx(onnx_model: Output[Model]):
    import os, shutil
    from ultralytics import YOLO
    model = YOLO("yolov8n.pt")  # pretrained weights baked into the image
    exported = model.export(format="onnx", imgsz=640)
    os.makedirs(os.path.dirname(onnx_model.path), exist_ok=True)
    shutil.copy(exported, onnx_model.path)


@dsl.component(base_image=YOLO_IMAGE)
def evaluate_onnx(
    dataset: Input[Dataset],
    onnx_model: Input[Model],
    report: Output[Artifact],
    metrics: Output[Metrics],
):
    import os, json, shutil, zipfile
    import ultralytics
    from ultralytics import YOLO

    workdir = "/tmp/data"
    with zipfile.ZipFile(dataset.path) as z:
        z.extractall(workdir)

    # Ultralytics needs the .onnx suffix to recognize the format.
    onnx_file = "/tmp/yolov8n.onnx"
    shutil.copy(onnx_model.path, onnx_file)

    # Build the dataset config as plain text, reusing the class names
    # from Ultralytics' bundled COCO128 config. This avoids importing yaml.
    cfg_path = os.path.join(os.path.dirname(ultralytics.__file__), "cfg", "datasets", "coco128.yaml")
    with open(cfg_path) as f:
        lines = f.read().splitlines()
    names_idx = next(i for i, l in enumerate(lines) if l.startswith("names:"))

    data_yaml = "/tmp/coco128_eval.yaml"
    with open(data_yaml, "w") as f:
        f.write(f"path: {os.path.join(workdir, 'coco128')}\n")
        f.write("train: images/train2017\n")
        f.write("val: images/train2017\n")
        f.write("\n".join(lines[names_idx:]) + "\n")

    model = YOLO(onnx_file)
    m = model.val(data=data_yaml, imgsz=640, batch=1, plots=False,
                  verbose=False, workers=0, project="/tmp/val", name="eval", exist_ok=True)

    results = {
        "model": "yolov8n (ONNX)",
        "dataset": "COCO128 (train2017 subset, NOT held-out)",
        "precision": float(m.box.mp),
        "recall": float(m.box.mr),
        "mAP50": float(m.box.map50),
        "mAP50_95": float(m.box.map),
    }

    metrics.log_metric("precision", results["precision"])
    metrics.log_metric("recall", results["recall"])
    metrics.log_metric("mAP50", results["mAP50"])
    metrics.log_metric("mAP50_95", results["mAP50_95"])

    # Single file, so the artifact link resolves.
    with open(report.path, "w") as f:
        json.dump(results, f, indent=2)


@dsl.pipeline(name="yolo-onnx-export-eval")
def yolo_pipeline():
    data = get_coco128()
    onnx = export_onnx()
    evaluate = evaluate_onnx(
        dataset=data.outputs["dataset"],
        onnx_model=onnx.outputs["onnx_model"],
    )
    data.set_caching_options(False)
    onnx.set_caching_options(False)
    evaluate.set_caching_options(False)


if __name__ == "__main__":
    compiler.Compiler().compile(yolo_pipeline, "yolo_pipeline.yaml")
    print("wrote yolo_pipeline.yaml")