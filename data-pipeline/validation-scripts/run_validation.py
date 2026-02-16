"""
Unified validation runner for SavVio data pipeline.

Can run any combination of validation stages:
  - raw       → validates data/raw/ 
  - anomalies → checks for outliers 
  - processed → validates data/processed/ 
  - features  → validates data/validated/ 

Each stage returns a ValidationReport with a pipeline_action:
  CONTINUE  — all checks passed (or only INFO-level failures)
  ALERT     — WARNING-level failures (send email/Slack, continue)
  HALT      — CRITICAL failures (stop the pipeline)

Airflow integration:
  Each stage is a separate PythonOperator task. The callable
  functions raise AirflowException on HALT to stop downstream tasks.
"""

import logging
import sys
from pathlib import Path

# Ensure scripts/validate is on the path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from validation_config import ValidationReport
from validate.raw_validator import run_raw_validation
from validate.processed_validator import run_processed_validation
from validate.feature_validator import run_feature_validation
from anomaly.anomaly_validator import run_anomaly_validation

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# Airflow-compatible callables
# ═══════════════════════════════════════════════════════════════════════════

def validate_raw(**kwargs) -> dict:
    """
    Airflow PythonOperator callable for raw validation.
    Raises RuntimeError on CRITICAL failure to halt the DAG.
    """
    report = run_raw_validation()
    _handle_report(report)
    return report.summary


def validate_processed(**kwargs) -> dict:
    """
    Airflow PythonOperator callable for processed validation.
    Raises RuntimeError on CRITICAL failure to halt the DAG.
    """
    report = run_processed_validation()
    _handle_report(report)
    return report.summary


def validate_features(**kwargs) -> dict:
    """
    Airflow PythonOperator callable for feature validation.
    Raises RuntimeError on CRITICAL failure to halt the DAG.
    """
    report = run_feature_validation()
    _handle_report(report)
    return report.summary


def validate_anomalies(**kwargs) -> dict:
    """
    Airflow PythonOperator callable for anomaly detection.
    Raises RuntimeError on CRITICAL failure (though rare for anomalies).
    """
    report = run_anomaly_validation()
    _handle_report(report)
    return report.summary


def _handle_report(report: ValidationReport) -> None:
    """
    Central handler for validation reports.
    - HALT   → raise exception (stops Airflow task)
    - ALERT  → send notification (email/Slack placeholder)
    - CONTINUE → log and proceed
    """
    action = report.summary["pipeline_action"]

    if action == "HALT":
        msg = (
            f"VALIDATION FAILED [{report.stage}]: "
            f"{report.summary['failed_critical']} CRITICAL failures. "
            f"Pipeline halted."
        )
        logger.critical(msg)
        raise RuntimeError(msg)

    elif action == "ALERT":
        logger.warning(
            "Validation [%s] passed with %d warnings — triggering alerts",
            report.stage, report.summary["failed_warning"],
        )
        _send_alert(report)

    else:
        logger.info("Validation [%s] passed — continuing pipeline", report.stage)


def _send_alert(report: ValidationReport) -> None:
    """
    Placeholder for alert dispatch.
    In production, wire up to Airflow EmailOperator or SlackWebhookOperator.
    """
    summary = report.summary
    failed_checks = [r for r in report.results if not r.passed]

    subject = f"SavVio Validation Warning — {summary['stage'].upper()}"
    body_lines = [
        f"Stage: {summary['stage']}",
        f"Timestamp: {summary['timestamp']}",
        f"Warnings: {summary['failed_warning']}",
        f"Info: {summary['failed_info']}",
        "",
        "Failed checks:",
    ]
    for c in failed_checks:
        body_lines.append(f"  [{c.severity.name}] {c.check_name}: {c.details}")

    body = "\n".join(body_lines)
    logger.warning("ALERT:\n%s\n%s", subject, body)

    # TODO: Replace with actual alert
    # from airflow.operators.email import EmailOperator
    # EmailOperator(to="team@savvio.dev", subject=subject, html_content=body)


# ═══════════════════════════════════════════════════════════════════════════
# CLI — run all stages or a specific one
# ═══════════════════════════════════════════════════════════════════════════

def main():
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    )

    parser = argparse.ArgumentParser(description="SavVio Data Validation Runner")
    parser.add_argument(
        "stage",
        choices=["raw", "processed", "features", "anomalies", "all"],
        help="Which validation stage to run",
    )
    args = parser.parse_args()

    exit_code = 0

    # Map stage names to callables
    STAGE_MAP = {
        "raw": validate_raw,
        "processed": validate_processed,
        "features": validate_features,
        "anomalies": validate_anomalies,
    }

    if args.stage == "all":
        stages_to_run = ["raw", "anomalies", "processed", "features"]
    else:
        stages_to_run = [args.stage]

    for stage_name in stages_to_run:
        validator_func = STAGE_MAP.get(stage_name)
        if not validator_func:
            logger.error(f"Unknown stage: {stage_name}")
            continue
            
        try:
            logger.info(f"Running validation for stage: {stage_name}")
            validator_func()
        except RuntimeError:
            exit_code = 1
            if args.stage != "all":
                sys.exit(1)
        except RuntimeError:
            exit_code = 1
            logger.error(f"Validation for stage '{stage_name}' failed critically.")
            if args.stage != "all": # If only one stage was requested, exit immediately
                sys.exit(1)
        except Exception as e:
            exit_code = 1
            logger.exception(f"An unexpected error occurred during validation for stage '{stage_name}': {e}")
            if args.stage != "all":
                sys.exit(1)

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
