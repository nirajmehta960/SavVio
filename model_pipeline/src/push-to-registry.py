"""
Push Model to GCP Artifact Registry.

Standalone script called by CI/CD (GitHub Actions Job 4) after all
gates pass. Not called during local development.

Pulls the latest registered model from MLflow, serializes it, and
uploads it to GCP Artifact Registry with full lineage metadata.

Usage (CI/CD):
    python scripts/push_to_registry.py

Requires:
    - GOOGLE_APPLICATION_CREDENTIALS env var set (service account key)
    - gcloud CLI authenticated
    - MLflow tracking server accessible
"""

import os
import sys
import json
import logging
import subprocess
import tempfile
from datetime import datetime

import joblib
import mlflow
from mlflow.tracking import MlflowClient

# Add src/ to path so we can import Config.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from config import Config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def get_latest_model_version(client, model_name):
    """Fetch the latest version of the registered model from MLflow."""
    versions = client.search_model_versions(f"name='{model_name}'")
    if not versions:
        raise RuntimeError(f"No versions found for model '{model_name}'")
    latest = max(versions, key=lambda v: int(v.version))
    logger.info("Latest MLflow model: %s v%s (run: %s)", model_name, latest.version, latest.run_id)
    return latest


def collect_metadata(client, model_version):
    """Collect lineage metadata from the MLflow run."""
    run = client.get_run(model_version.run_id)

    # Git commit hash (best effort).
    git_hash = "unknown"
    try:
        git_hash = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip()
    except Exception:
        logger.warning("Could not retrieve git commit hash.")

    # DVC data version (best effort).
    dvc_hash = "unknown"
    try:
        dvc_hash = subprocess.check_output(
            ["git", "log", "-1", "--format=%H", "--", "*.dvc"], text=True
        ).strip() or "unknown"
    except Exception:
        logger.warning("Could not retrieve DVC data version.")

    metadata = {
        "model_name": model_version.name,
        "model_version": model_version.version,
        "mlflow_run_id": model_version.run_id,
        "git_commit": git_hash,
        "dvc_data_version": dvc_hash,
        "metrics": run.data.metrics,
        "pushed_at": datetime.utcnow().isoformat(),
    }
    return metadata


def download_model(client, model_version, dest_dir):
    """Download the model artifact from MLflow to a local directory."""
    model_uri = f"models:/{model_version.name}/{model_version.version}"
    local_path = mlflow.artifacts.download_artifacts(artifact_uri=model_uri, dst_path=dest_dir)
    logger.info("Model downloaded to: %s", local_path)
    return local_path


def push_to_gcp(model_path, metadata, metadata_path):
    """Upload model + metadata to GCP Artifact Registry."""
    repo = Config.ARTIFACT_REGISTRY_REPO
    location = Config.GCP_REGION
    project = Config.GCP_PROJECT_ID
    version = metadata["model_version"]
    package = metadata["model_name"].lower().replace(" ", "-")

    # Ensure repository exists (idempotent).
    subprocess.run([
        "gcloud", "artifacts", "repositories", "describe",
        repo, f"--location={location}", f"--project={project}",
    ], capture_output=True)

    # Upload model artifact.
    logger.info("Pushing model to GCP Artifact Registry...")
    result = subprocess.run([
        "gcloud", "artifacts", "generic", "upload",
        f"--repository={repo}",
        f"--location={location}",
        f"--project={project}",
        f"--package={package}",
        f"--version={version}",
        f"--source={model_path}",
    ], capture_output=True, text=True)

    if result.returncode != 0:
        logger.error("GCP push failed: %s", result.stderr)
        raise RuntimeError(f"gcloud upload failed: {result.stderr}")

    # Upload metadata alongside the model.
    subprocess.run([
        "gcloud", "artifacts", "generic", "upload",
        f"--repository={repo}",
        f"--location={location}",
        f"--project={project}",
        f"--package={package}-metadata",
        f"--version={version}",
        f"--source={metadata_path}",
    ], capture_output=True, text=True)

    logger.info("Successfully pushed %s v%s to GCP Artifact Registry.", package, version)


def main():
    mlflow.set_tracking_uri(Config.MLFLOW_TRACKING_URI)
    client = MlflowClient()

    # 1. Get latest registered model from MLflow.
    model_version = get_latest_model_version(client, Config.REGISTERED_MODEL_NAME)

    # 2. Collect lineage metadata.
    metadata = collect_metadata(client, model_version)

    with tempfile.TemporaryDirectory() as tmp_dir:
        # 3. Download model artifact from MLflow.
        model_path = download_model(client, model_version, tmp_dir)

        # 4. Save metadata JSON alongside model.
        metadata_path = os.path.join(tmp_dir, "model_metadata.json")
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2, default=str)
        logger.info("Metadata written: %s", json.dumps(metadata, indent=2, default=str))

        # 5. Push to GCP Artifact Registry.
        push_to_gcp(model_path, metadata, metadata_path)

    logger.info("Registry push complete.")


if __name__ == "__main__":
    main()
