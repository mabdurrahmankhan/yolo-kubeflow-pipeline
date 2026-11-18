# Kubeflow YOLOv8n — Pipelines + Notebooks

Minimal project demonstrating two core Kubeflow features.

| Feature | What you learn |
|---|---|
| **Kubeflow Pipelines** | 4-step DAG, parallel steps, typed artifacts, metrics |
| **Kubeflow Notebooks** | Jupyter as a K8s pod, PVC storage, interactive exploration |

**Platform:** Fedora Linux, 32 GB RAM, Python 3.11/3.12 (not 3.14)

---

## One-time System Setup

```bash
# Install networking tools required by minikube/kubeadm
sudo dnf install -y conntrack-tools iptables iproute-tc ethtool ebtables

# Install Python 3.11 (system Python 3.14 is too new for kfp/ultralytics)
sudo dnf install -y python3.11 python3.11-venv

# kubectl
curl -fsSLo kubectl "https://dl.k8s.io/release/v1.30.2/bin/linux/amd64/kubectl"
chmod +x kubectl && sudo mv kubectl /usr/local/bin/

# minikube
curl -fsSLo minikube "https://storage.googleapis.com/minikube/releases/v1.33.1/minikube-linux-amd64"
chmod +x minikube && sudo mv minikube /usr/local/bin/

# kustomize — NOTE: installs to current directory, move it
curl -fsSL "https://raw.githubusercontent.com/kubernetes-sigs/kustomize/master/hack/install_kustomize.sh" | bash
sudo mv kustomize /usr/local/bin/

# Verify all tools
kubectl version --client && minikube version && kustomize version
```

---

## Python Virtual Environment

```bash
# Create venv with Python 3.11 (do this ONCE)
python3.11 -m venv ~/kubeflow-venv

# Activate — do this every time you open a new terminal
source ~/kubeflow-venv/bin/activate
# Prompt shows: (kubeflow-venv) abdurrehman@fedora:~$

# Install Python dependencies
cd ~/Downloads/kubeflow-yolo-project
pip install -r requirements.txt
```

> **Note:** `kubectl`, `minikube`, `docker`, `kustomize` are system tools in
> `/usr/local/bin/` — they work without the venv. Only `kfp`, `ultralytics`
> etc. need the venv active.

---

## Run Order

```bash
cd ~/Downloads/kubeflow-yolo-project
chmod +x scripts/*.sh
source ~/kubeflow-venv/bin/activate

# 1. Start cluster + deploy Kubeflow (~15-20 min)
bash scripts/01-setup-minikube.sh

# IF any pods are stuck in PodInitializing / ImagePullBackOff:
bash scripts/00-fix-images.sh

# 2. Open Central Dashboard — keep running in its own terminal
kubectl port-forward svc/istio-ingressgateway -n istio-system 8080:80
# → http://localhost:8080
# Login: user@example.com / 12341234

# 3. Create Notebook server
bash scripts/02-create-notebook.sh
# → Dashboard → Notebooks → yolo-notebook → Connect
# → Open yolo_exploration.ipynb → Run All Cells

# 4. Compile & submit pipeline (venv must be active)
bash scripts/03-compile-pipeline.sh
# → Watch 4-step DAG at http://localhost:8080

# 5. Cleanup
bash scripts/99-cleanup.sh
```

---

## Script Guide

| Script | Purpose |
|---|---|
| `00-fix-images.sh` | **Run if pods stuck** — loads gcr.io images from host into minikube |
| `01-setup-minikube.sh` | Starts minikube (8 CPU, 16 GB), pre-loads images, deploys Kubeflow |
| `02-create-notebook.sh` | Creates Notebook CR + 20 GB PVC in yolo-project namespace |
| `03-compile-pipeline.sh` | Compiles pipeline YAML and submits to KFP (needs venv active) |
| `99-cleanup.sh` | Deletes minikube cluster and all data |

---

## Pipeline Architecture

```
  [1] download-coco128  ──┐
                          ├──► [3] evaluate-model ──► [4] register-result
  [2] export-to-onnx    ──┘
```

- Steps 1 & 2 run **in parallel** — independent containers, no data dependency
- Step 3 waits for both (typed KFP artifacts: `Dataset` + `Model`)
- Step 4 logs the verdict (pass if mAP50 ≥ 0.30)
- Metrics visible in KFP UI → Runs → your run → **Artifacts tab**

---

## Why pods get stuck (and the fix)

Minikube runs inside a Docker container. Its internal DNS sometimes can't
resolve `gcr.io` even when the host can pull images fine. The fix:

```bash
# Pull on host, bypass minikube DNS entirely
docker pull gcr.io/ml-pipeline/mysql:8.0.26
minikube image load gcr.io/ml-pipeline/mysql:8.0.26
```

Both `00-fix-images.sh` and `01-setup-minikube.sh` do this automatically
for all 12 KFP images before deploying.

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `conntrack not found` on minikube start | `sudo dnf install -y conntrack-tools iptables iproute-tc` |
| `cannot change memory for existing cluster` | `minikube delete` then re-run setup |
| Pods stuck `PodInitializing` | `bash scripts/00-fix-images.sh` |
| `ImagePullBackOff` for gcr.io | Same — `00-fix-images.sh` |
| `kfp==2.7.0` install error | Python 3.14 too new — use `python3.11 -m venv ~/kubeflow-venv` |
| kustomize installed to wrong dir | `sudo mv ~/Downloads/kubeflow-yolo-project/kustomize /usr/local/bin/` |
| Dashboard unreachable | Re-run: `kubectl port-forward svc/istio-ingressgateway -n istio-system 8080:80` |
| Pipeline submit fails | Ensure venv active + port-forward running on 8080 |
| `docker: permission denied` | `sudo usermod -aG docker $USER` then log out/in |
