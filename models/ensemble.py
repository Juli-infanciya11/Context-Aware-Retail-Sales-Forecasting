import pandas as pd
import numpy as np
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.xgboost_model import train_xgboost
from models.prophet_model import train_prophet
from config import OUTPUT_PATH, CATEGORIES

def train_all_models(master_df):
    """Train XGBoost + Prophet for every category and ensemble"""
    results = {}

    print("\n  Training XGBoost (all categories)...")
    xgb_all = train_xgboost(master_df)

    for category in CATEGORIES:
        print(f"\n  ── {category} ──")

        # XGBoost per category
        xgb = train_xgboost(master_df, category)

        # Prophet per category
        prophet = train_prophet(master_df, category)

        # Ensemble — weighted average (XGB 60%, Prophet 40%)
        xgb_pred     = xgb["predictions"]["predicted"].values
        prophet_pred = prophet["predictions"]["predicted"].values
        min_len      = min(len(xgb_pred), len(prophet_pred))

        ensemble_pred = (0.6 * xgb_pred[:min_len] +
                         0.4 * prophet_pred[:min_len])

        actual = xgb["predictions"]["actual"].values[:min_len]

        from sklearn.metrics import mean_absolute_error, r2_score
        import numpy as np
        ens_mae = mean_absolute_error(actual, ensemble_pred)
        ens_r2  = r2_score(actual, ensemble_pred)

        print(f"    Ensemble MAE : {ens_mae:,.0f}")
        print(f"    Ensemble R²  : {ens_r2:.4f}")

        results[category] = {
            "xgboost"       : xgb,
            "prophet"       : prophet,
            "ensemble_pred" : ensemble_pred,
            "ensemble_metrics": {"MAE": ens_mae, "R2": ens_r2}
        }

    # Save combined results summary
    summary_rows = []
    for cat, res in results.items():
        summary_rows.append({
            "category"    : cat,
            "xgb_mae"     : round(res["xgboost"]["metrics"]["MAE"], 2),
            "xgb_r2"      : round(res["xgboost"]["metrics"]["R2"], 4),
            "prophet_mae" : round(res["prophet"]["metrics"]["MAE"], 2),
            "prophet_r2"  : round(res["prophet"]["metrics"]["R2"], 4),
            "ensemble_mae": round(res["ensemble_metrics"]["MAE"], 2),
            "ensemble_r2" : round(res["ensemble_metrics"]["R2"], 4),
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_path = os.path.join(OUTPUT_PATH, "model_summary.csv")
    summary_df.to_csv(summary_path, index=False)

    print("\n" + "=" * 50)
    print("  MODEL PERFORMANCE SUMMARY")
    print("=" * 50)
    print(summary_df.to_string(index=False))
    print(f"\n  Summary saved: {summary_path}")

    return results, summary_df

if __name__ == "__main__":
    df = pd.read_csv("data/processed/master_dataset.csv")
    results, summary = train_all_models(df)