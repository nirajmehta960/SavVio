## Runbook

```bash
# 1. Activate environment
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# OR use Docker:
docker compose up --build

# 2. Run feature engineering
python model-development/affordability_features.py

# 3. Train model
python model-development/src/train.py \
  --config model-development/config/training_config.yaml

# 5. Run post-training bias detection
python model-development/bias_detection.py

# 6. Push approved model to registry
python model-development/src/registry.py --model <best-run-id>
```
