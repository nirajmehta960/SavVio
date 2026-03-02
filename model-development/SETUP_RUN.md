## Runbook

### Option 1: Docker (recommended)
```bash
# Start all services (MLflow, PostgreSQL, RustFS, ml-trainer)
docker compose up --build

# Enter the training container
docker exec -it ml-trainer bash

# Run the modular pipeline (trains XGBoost, LightGBM, LinearBoost)
cd /app/src
python run_pipeline.py

# Or run the standalone linear pipeline
python pipeline_linear.py
```

### Option 2: Local virtualenv
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r model-requirements.txt

# Run the pipeline
cd src
python run_pipeline.py
```

### Individual Components
```bash
# Test DB connectivity (once DB integration is live)
python src/data/db_loader.py

# Run data validation after loading
python src/data/validate_data.py

# View MLflow UI
open http://localhost:5000
```

### Running Tests
```bash
pytest tests/ -v
```
