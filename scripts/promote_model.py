"""
Check MLflow model metrics and promote to Production stage if thresholds are met.

Thresholds (configurable via env vars):
  MIN_BLEU4   — minimum BLEU-4 score   (default 0.10)
  MIN_AUC_ROC — minimum AUC-ROC macro  (default 0.80)
"""

import os
import sys

MIN_BLEU4 = float(os.getenv("MIN_BLEU4", "0.10"))
MIN_AUC_ROC = float(os.getenv("MIN_AUC_ROC", "0.80"))


def promote_best_model():
    try:
        import mlflow
        from mlflow.tracking import MlflowClient
    except ImportError:
        print("ERROR: Install mlflow:  pip install mlflow")
        sys.exit(1)

    client = MlflowClient()
    model_name = os.getenv("MLFLOW_MODEL_NAME", "xray-report-generator")

    # Get all models in Staging
    staging_versions = client.get_latest_versions(model_name, stages=["Staging"])
    if not staging_versions:
        print("No model in Staging — nothing to promote.")
        return

    for version in staging_versions:
        run = client.get_run(version.run_id)
        metrics = run.data.metrics

        bleu4 = metrics.get("val_bleu4", 0.0)
        auc = metrics.get("val_auc_roc_macro", 0.0)

        print(f"Version {version.version}  BLEU-4={bleu4:.4f}  AUC-ROC={auc:.4f}")

        if bleu4 >= MIN_BLEU4 and auc >= MIN_AUC_ROC:
            client.transition_model_version_stage(
                name=model_name,
                version=version.version,
                stage="Production",
                archive_existing_versions=True,
            )
            print(f"  → Promoted version {version.version} to Production.")
        else:
            print(f"  → Thresholds not met (BLEU-4≥{MIN_BLEU4}, AUC-ROC≥{MIN_AUC_ROC}). Not promoting.")


if __name__ == "__main__":
    promote_best_model()
