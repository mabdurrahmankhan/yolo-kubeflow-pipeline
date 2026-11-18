#!/usr/bin/env bash
# =============================================================================
# 01-setup-minikube.sh
# Kubeflow Pipelines + Notebooks on Minikube
# Target: Fedora Linux, 32 GB RAM, no KServe
# =============================================================================
set -euo pipefail

MINIKUBE_CPUS=6
MINIKUBE_MEMORY=16000   # 16 GB — plenty for full Kubeflow stack
MINIKUBE_DISK="80g"
MANIFESTS_DIR="/tmp/kubeflow-manifests"

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'
info() { echo -e "${GREEN}[INFO]${NC}  $*"; }
warn() { echo -e "${YELLOW}[WARN]${NC}  $*"; }
die()  { echo -e "${RED}[ERROR]${NC} $*" >&2; exit 1; }

# ── 1. Prerequisites ──────────────────────────────────────────────────────────
info "Checking prerequisites..."
command -v docker    >/dev/null 2>&1 || die "Docker not found. Run: sudo systemctl start docker"
command -v kubectl   >/dev/null 2>&1 || die "kubectl not found."
command -v minikube  >/dev/null 2>&1 || die "minikube not found."
command -v kustomize >/dev/null 2>&1 || die "kustomize not found."
command -v conntrack >/dev/null 2>&1 || die "conntrack not found. Run: sudo dnf install -y conntrack-tools iptables iproute-tc ethtool ebtables"
info "All tools found."

# ── 2. Delete any existing cluster ───────────────────────────────────────────
if minikube status &>/dev/null; then
  warn "Existing minikube cluster found — deleting it for a clean start..."
  minikube delete
fi

# ── 3. Start Minikube ─────────────────────────────────────────────────────────
info "Starting Minikube (CPUs=${MINIKUBE_CPUS}, Memory=${MINIKUBE_MEMORY}MB, Disk=${MINIKUBE_DISK})..."
minikube start \
  --driver=docker \
  --cpus="${MINIKUBE_CPUS}" \
  --memory="${MINIKUBE_MEMORY}" \
  --disk-size="${MINIKUBE_DISK}" \
  --kubernetes-version=v1.30.0 \
  --addons=storage-provisioner \
  --addons=default-storageclass \
  --addons=metrics-server

minikube status

# ── 4. Pre-load gcr.io images ─────────────────────────────────────────────────
# Pull on host and load into minikube to avoid in-cluster DNS issues with gcr.io
info "Pre-loading gcr.io images into minikube..."

GCR_IMAGES=(
  "gcr.io/ml-pipeline/mysql:8.0.26"
  "gcr.io/ml-pipeline/minio:RELEASE.2023-01-12T02-06-16Z.fips"
  "gcr.io/ml-pipeline/api-server:2.2.0"
  "gcr.io/ml-pipeline/persistenceagent:2.2.0"
  "gcr.io/ml-pipeline/scheduledworkflow:2.2.0"
  "gcr.io/ml-pipeline/frontend:2.2.0"
  "gcr.io/ml-pipeline/viewer-crd-controller:2.2.0"
  "gcr.io/ml-pipeline/metadata-writer:2.2.0"
  "gcr.io/ml-pipeline/cache-server:2.2.0"
  "gcr.io/ml-pipeline/metadata-envoy:2.2.0"
  "gcr.io/tfx-oss-public/ml_metadata_store_server:1.14.0"
  "gcr.io/ml-pipeline/workflow-controller:v3.4.16-license-compliance"
)

for img in "${GCR_IMAGES[@]}"; do
  info "  Loading: ${img}"
  docker image inspect "${img}" >/dev/null 2>&1 || docker pull "${img}" || {
    warn "  Could not pull ${img} — will attempt in-cluster pull."
    continue
  }
  minikube image load "${img}" || warn "  Could not load ${img}."
done
info "Image pre-loading complete."

# ── 5. Clone manifests ────────────────────────────────────────────────────────
if [ ! -d "${MANIFESTS_DIR}" ]; then
  info "Cloning Kubeflow manifests v1.9.0..."
  git clone --depth 1 --branch v1.9.0 \
    https://github.com/kubeflow/manifests.git "${MANIFESTS_DIR}"
else
  info "Manifests already at ${MANIFESTS_DIR}"
fi
cd "${MANIFESTS_DIR}"

# ── Helpers ───────────────────────────────────────────────────────────────────
deploy() {
  local name="$1" path="$2"
  info "Deploying ${name}..."
  local attempts=0
  until kustomize build "${path}" | kubectl apply -f - 2>/dev/null; do
    attempts=$((attempts+1))
    [ $attempts -ge 5 ] && { warn "Skipping ${name} after 5 attempts."; return 0; }
    warn "  Retrying ${name} in 10s..."
    sleep 10
  done
  info "${name} applied."
}

wait_pods() {
  local ns="$1" label="$2" timeout="${3:-240s}"
  info "Waiting for pods in ${ns} (${label})..."
  kubectl -n "${ns}" wait --for=condition=ready pod \
    -l "${label}" --timeout="${timeout}" 2>/dev/null || \
    warn "  Some pods not ready yet — continuing."
}

# ── 6. cert-manager ───────────────────────────────────────────────────────────
deploy "cert-manager" "common/cert-manager/cert-manager/base"
wait_pods "cert-manager" "app=cert-manager" "180s"
sleep 15

# ── 7. Istio ──────────────────────────────────────────────────────────────────
deploy "istio-crds"      "common/istio-1-22/istio-crds/base"
deploy "istio-namespace" "common/istio-1-22/istio-namespace/base"
deploy "istio-install"   "common/istio-1-22/istio-install/overlays/oauth2-proxy"
wait_pods "istio-system" "app=istiod"               "240s"
wait_pods "istio-system" "app=istio-ingressgateway" "240s"

# ── 8. Auth ───────────────────────────────────────────────────────────────────
deploy "dex"          "common/dex/overlays/istio"
deploy "oauth2-proxy" "common/oidc-client/oauth2-proxy/overlays/m2m-self-signed"

# ── 9. Kubeflow core ──────────────────────────────────────────────────────────
deploy "kubeflow-namespace" "common/kubeflow-namespace/base"
deploy "kubeflow-roles"     "common/kubeflow-roles/base"
deploy "istio-resources"    "common/istio-1-22/kubeflow-istio-resources/base"

# ── 10. Kubeflow Pipelines ────────────────────────────────────────────────────
deploy "kubeflow-pipelines" \
  "apps/pipeline/upstream/env/cert-manager/platform-agnostic-multi-user"
wait_pods "kubeflow" "app=ml-pipeline" "360s"

# ── 11. Notebooks ─────────────────────────────────────────────────────────────
deploy "notebook-controller" \
  "apps/jupyter/notebook-controller/upstream/overlays/kubeflow"
deploy "jupyter-web-app" \
  "apps/jupyter/jupyter-web-app/upstream/overlays/istio"

# ── 12. Central Dashboard ─────────────────────────────────────────────────────
deploy "central-dashboard" "apps/centraldashboard/upstream/overlays/istio"

# ── 13. Profiles + webhook ────────────────────────────────────────────────────
deploy "profiles"          "apps/profiles/upstream/overlays/kubeflow"
deploy "admission-webhook" "apps/admission-webhook/upstream/overlays/cert-manager"
deploy "user-namespace"    "common/user-namespace/base"

# ── 14. yolo-project profile ──────────────────────────────────────────────────
info "Creating yolo-project profile..."
kubectl apply -f - <<EOF
apiVersion: kubeflow.org/v1
kind: Profile
metadata:
  name: yolo-project
spec:
  owner:
    kind: User
    name: user@example.com
  resourceQuotaSpec:
    hard:
      cpu: "8"
      memory: 12Gi
      persistentvolumeclaims: "10"
EOF
sleep 10

# ── 15. Final status ──────────────────────────────────────────────────────────
info "================================================================"
info " Kubeflow is ready!  (Pipelines + Notebooks)"
info ""
info " Open Central Dashboard:"
info "   kubectl port-forward svc/istio-ingressgateway \\"
info "     -n istio-system 8080:80"
info "   Then open: http://localhost:8080"
info "   Login: user@example.com  /  12341234"
info ""
info " Check all pods:"
info "   kubectl get pods -A"
info ""
info " Next step:"
info "   bash scripts/02-create-notebook.sh"
info "================================================================"
