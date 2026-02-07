"""
Data Upload Orchestrator for SavVio Pipeline
Handles uploading of local datasets (processed, validated, features) to Google Cloud Storage.
This script ensures efficient syncing by verifying MD5 checksums before upload.
"""

import sys
import os
import argparse
import logging
import base64
import hashlib
from pathlib import Path

# Add current directory to path so imports work
current_dir = Path(__file__).resolve().parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

from ingestion.gcs_loader import GCSLoader
from config import (
    GCS_BUCKET_NAME,
    GCP_PROJECT_ID,
    GCP_CREDENTIALS_PATH,
    PROCESSED_DATA_DIR,
    VALIDATED_DATA_DIR,
    FEATURES_DATA_DIR,
    GCS_PROCESSED_PATH,
    GCS_VALIDATED_PATH,
    GCS_FEATURES_PATH
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s'
)
logger = logging.getLogger(__name__)


def calculate_md5(file_path):
    """Calculate the MD5 checksum of a local file."""
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return base64.b64encode(hash_md5.digest()).decode('utf-8')


def upload_directory_smart(local_dir: Path, gcs_prefix: str):
    """
    Upload files to GCS only if they are new or modified (checked via MD5 hash).
    """
    if not local_dir.exists():
        logger.warning(f"Local directory does not exist: {local_dir}")
        return

    # Initialize GCS Loader
    loader = GCSLoader(credentials_path=GCP_CREDENTIALS_PATH, project_id=GCP_PROJECT_ID)
    bucket = loader.client.bucket(GCS_BUCKET_NAME)

    files = [f for f in local_dir.iterdir() if f.is_file() and f.name != '.gitkeep']
    
    if not files:
        logger.warning(f"No files found in {local_dir}")
        return

    logger.info(f"Scanning {len(files)} files in {local_dir}...")

    for file_path in files:
        # Construct GCS blob path
        destination_blob_name = f"{gcs_prefix.rstrip('/')}/{file_path.name}"
        blob = bucket.blob(destination_blob_name)
        
        should_upload = True
        
        # Check if blob exists
        if blob.exists():
            # Refresh metadata to get MD5 hash
            blob.reload()
            remote_md5 = blob.md5_hash
            local_md5 = calculate_md5(file_path)
            
            if remote_md5 == local_md5:
                # print(f"  [SKIP] {file_path.name} (Unchanged)")
                should_upload = False
            else:
                logger.info(f"  [UPDATE] {file_path.name} (Changed)")
        else:
            logger.info(f"  [NEW] {file_path.name} (Missing in GCS)")

        # Upload if necessary
        if should_upload:
            try:
                loader.upload_blob(
                    bucket_name=GCS_BUCKET_NAME,
                    source_path=str(file_path),
                    destination_blob_name=destination_blob_name
                )
                print(f"✓ Uploaded: {file_path.name} -> gs://{GCS_BUCKET_NAME}/{destination_blob_name}")
            except Exception as e:
                logger.error(f"Failed to upload {file_path.name}: {e}")
        else:
            print(f"✓ Skipped: {file_path.name} (Already up to date)")


def main():
    parser = argparse.ArgumentParser(description="Smart Upload data to GCS")
    parser.add_argument(
        "--target", 
        type=str, 
        choices=['processed', 'validated', 'features'], 
        required=True,
        help="Which dataset stage to upload"
    )
    
    args = parser.parse_args()
    
    if args.target == 'processed':
        logger.info("Target: PROCESSED DATA (Smart Sync)")
        upload_directory_smart(PROCESSED_DATA_DIR, GCS_PROCESSED_PATH)
        
    elif args.target == 'validated':
        logger.info("Target: VALIDATED DATA (Smart Sync)")
        upload_directory_smart(VALIDATED_DATA_DIR, GCS_VALIDATED_PATH)
        
    elif args.target == 'features':
        logger.info("Target: FEATURE STORE (Smart Sync)")
        upload_directory_smart(FEATURES_DATA_DIR, GCS_FEATURES_PATH)

if __name__ == "__main__":
    main()
