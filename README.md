# Kubeflow YOLOv8n Pipeline

A minimal Kubeflow Pipelines project that downloads COCO128, exports a pretrained YOLOv8n model to ONNX, and evaluates the ONNX model. Each step passes typed artifacts through MinIO.

**Platform:** Fedora Linux, 32 GB RAM, Python 3.11 or 3.12 (not 3.14)

---

## Pipeline

```
[1] get-coco128  ──►  [3] evaluate-onnx
                         ▲
[2] export-onnx  ────────┘
```

- **get-coco128** downloads COCO128 and outputs it as a `Dataset` artifact.
- **export-onnx** loads the pretrained `yolov8n.pt` (baked into the image) and outputs an `onnx_model` artifact.
- **evaluate-onnx** runs the ONNX model with Ultralytics validation and logs `precision`, `recall`, `mAP50`, and `mAP50_95` as metrics. It also writes a `report` JSON file.

Results are on COCO128, which the pretrained model was trained on, so the scores are optimistic. They show the pipeline works end to end, not a fair accuracy number. A held-out labeled set is needed for that.

---

## One-time System Setup

```bash
# Networking tools required by minikube
sudo dnf install -y conntrack-tools iptables iproute-tc ethtool ebtables

# Python 3.11
sudo dnf install -y python3.11 python3.11-venv

# kubectl
curl -fsSLo kubectl "https://dl.k8s.io/release/v1.30.2/bin/linux/amd64/kubectl"
chmod +x kubectl && sudo mv kubectl /usr/local/bin/

# minikube
curl -fsSLo minikube "https://storage.googleapis.com/minikube/releases/v1.33.1/minikube-linux-amd64"
chmod +x minikube && sudo mv minikube /usr/local/bin/

# kustomize (the installer writes to the current directory, so move it)
curl -fsSL "https://raw.githubusercontent.com/kubernetes-sigs/kustomize/master/hack/install_kustomize.sh" | bash
sudo mv kustomize /usr/local/bin/

kubectl version --client && minikube version && kustomize version
```

---

## Python Environment

```bash
python3.11 -m venv ~/kubeflow-venv
source ~/kubeflow-venv/bin/activate
pip install -r requirements.txt
```

`kubectl`, `minikube`, `docker`, and `kustomize` are system tools and work without the venv. Only the Python scripts need it.

---

## Run Order

```bash
cd ~/Downloads/kubeflow-yolo-project
chmod +x scripts/*.sh
source ~/kubeflow-venv/bin/activate

# 1. Start the cluster and deploy Kubeflow (about 15–20 minutes)
bash scripts/01-setup-minikube.sh

# 2. Open the Central Dashboard, keeping this running in its own terminal
kubectl port-forward -n istio-system svc/istio-ingressgateway 8080:80
# → http://localhost:8080   (user@example.com / see install output)

# 3. Build the pipeline image (includes Ultralytics and the pretrained weights)
cd pipeline
docker build -t yolo-kfp:v2 .
minikube image load yolo-kfp:v2

# 4. Compile the pipeline
python yolo_pipeline.py          # writes yolo_pipeline.yaml

# 5. Upload yolo_pipeline.yaml in the dashboard, create a run in the
#    yolo-project namespace, and watch the steps at http://localhost:8080

# 6. Cleanup (deletes the cluster and all data)
bash scripts/99-cleanup.sh
```

---

## Scripts

| Script | Purpose |
|---|---|
| `01-setup-minikube.sh` | Starts minikube and deploys Kubeflow Pipelines |
| `99-cleanup.sh` | Deletes the minikube cluster and all data |

---

## Results

In the dashboard, open the run:
- **Metrics tab:** `precision`, `recall`, `mAP50`, `mAP50_95`
- **Artifacts:** `dataset`, `onnx_model`, and `report` (JSON)

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `conntrack not found` on minikube start | `sudo dnf install -y conntrack-tools iptables iproute-tc` |
| `cannot change memory for existing cluster` | `minikube delete`, then re-run the setup script |
| Pods stuck in `ImagePullBackOff` | Pull the image on the host, then `minikube image load <image>` |
| `kfp==2.7.0` install error | Python 3.14 is too new; use `python3.11 -m venv` |
| Pipeline run stays `Skipped` | Stale cache; change a component's code or clear `cachedb.execution_caches` |
| Dashboard unreachable | Restart the port-forward to `istio-ingressgateway` |
| `docker: permission denied` | `sudo usermod -aG docker $USER`, then log out and back in |
