"""
Pipeline de Modelado Supervisado y AutoML para Predicción de Partidos (1X2).
Compara Baseline de clase mayoritaria, Random Forest (con y sin balanceo),
LightGBM, XGBoost, Regresión Logística y FLAML AutoML optimizado para macro-F1.
Garantiza partición temporal estricta (Train: 2022-2023, Test: 2024).
"""

import os
import sys
import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Tuple

import pandas as pd
import numpy as np
import duckdb
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    log_loss,
    classification_report,
    confusion_matrix
)
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from flaml import AutoML

# Ajustar PYTHONPATH
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import LAKEHOUSE_DB_PATH

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("SupervisedModeling")

MODELS_DIR = PROJECT_ROOT / "models"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Lista estricta de variables predictoras (100% pre-partido, cero fuga)
FEATURE_COLS = [
    "home_pts_ult5", "home_gf_ult5", "home_ga_ult5", "home_gd_ult5", "home_games_tracked",
    "away_pts_ult5", "away_gf_ult5", "away_ga_ult5", "away_gd_ult5", "away_games_tracked",
    "diff_pts_ult5", "diff_gd_ult5",
    "home_cum_pts", "home_cum_gf", "home_cum_ga", "home_cum_pj", "home_cum_ppg",
    "away_cum_pts", "away_cum_gf", "away_cum_ga", "away_cum_pj", "away_cum_ppg",
    "diff_cum_ppg",
    "home_rest_days", "away_rest_days", "diff_rest_days"
]
TARGET_COL = "resultado_1x2"
CLASSES = sorted(["Local", "Empate", "Visita"])  # ['Empate', 'Local', 'Visita'] orden lexicográfico oficial


def load_train_test_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Carga los datos de matches_features desde DuckDB y realiza la separación temporal."""
    con = duckdb.connect(str(LAKEHOUSE_DB_PATH), read_only=True)
    df = con.execute("SELECT * FROM matches_features ORDER BY match_date ASC").fetchdf()
    con.close()

    train_mask = df["split_set"] == "train"
    test_mask = df["split_set"] == "test"

    X_train = df.loc[train_mask, FEATURE_COLS].copy()
    y_train = df.loc[train_mask, TARGET_COL].copy()

    X_test = df.loc[test_mask, FEATURE_COLS].copy()
    y_test = df.loc[test_mask, TARGET_COL].copy()

    logger.info(f"Dataset cargado: Train={X_train.shape} (760), Test={X_test.shape} (380)")
    logger.info(f"Variables predictoras ({len(FEATURE_COLS)}): {FEATURE_COLS}")
    return X_train, X_test, y_train, y_test


def evaluate_model(name: str, model: Any, X_train: pd.DataFrame, y_train: pd.Series,
                   X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Any]:
    """Entrena y evalúa un modelo sobre el conjunto de prueba temporal."""
    logger.info(f"Entrenando modelo: {name}...")
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test) if hasattr(model, "predict_proba") else None

    acc = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
    
    # Pérdida logarítmica (Log Loss)
    try:
        loss = log_loss(y_test, y_pred_proba, labels=CLASSES) if y_pred_proba is not None else None
    except Exception:
        loss = None

    # Métricas detalladas por clase
    clf_rep = classification_report(y_test, y_pred, target_names=CLASSES, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_test, y_pred, labels=CLASSES)

    metrics = {
        "model_name": name,
        "accuracy": round(acc, 4),
        "macro_f1": round(macro_f1, 4),
        "weighted_f1": round(weighted_f1, 4),
        "log_loss": round(loss, 4) if loss else None,
        "per_class": {
            cls_name: {
                "precision": round(clf_rep[cls_name]["precision"], 4),
                "recall": round(clf_rep[cls_name]["recall"], 4),
                "f1_score": round(clf_rep[cls_name]["f1-score"], 4),
                "support": clf_rep[cls_name]["support"]
            }
            for cls_name in CLASSES
        },
        "confusion_matrix": cm.tolist()
    }

    # Guardar binario del modelo
    model_filename = MODELS_DIR / f"{name.lower().replace(' ', '_').replace('/', '_')}.joblib"
    joblib.dump(model, model_filename)
    logger.info(f"Modelo guardado en: {model_filename}")

    return metrics


def run_supervised_experiments() -> Dict[str, Any]:
    """Ejecuta la suite comparativa de modelos supervisados y AutoML."""
    X_train, X_test, y_train, y_test = load_train_test_data()

    results = {}

    # 1. Baseline: Clase Mayoritaria
    baseline = DummyClassifier(strategy="most_frequent")
    results["baseline_most_frequent"] = evaluate_model(
        "Baseline Most Frequent", baseline, X_train, y_train, X_test, y_test
    )

    # 2. Random Forest Sin Balanceo
    rf_unweighted = RandomForestClassifier(
        n_estimators=200, max_depth=6, random_state=42
    )
    results["rf_unweighted"] = evaluate_model(
        "Random Forest Unweighted", rf_unweighted, X_train, y_train, X_test, y_test
    )

    # 3. Random Forest Con Balanceo de Clases
    rf_weighted = RandomForestClassifier(
        n_estimators=200, max_depth=6, class_weight="balanced", random_state=42
    )
    results["rf_weighted"] = evaluate_model(
        "Random Forest Weighted", rf_weighted, X_train, y_train, X_test, y_test
    )

    # 4. Regresión Logística Multinomial (Escalada y Balanceada)
    logit_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("logit", LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42))
    ])
    results["logistic_regression"] = evaluate_model(
        "Logistic Regression Balanced", logit_pipe, X_train, y_train, X_test, y_test
    )

    # 5. LightGBM (Con Balanceo)
    lgbm = LGBMClassifier(
        n_estimators=150, max_depth=4, learning_rate=0.05,
        class_weight="balanced", random_state=42, verbose=-1
    )
    results["lightgbm_weighted"] = evaluate_model(
        "LightGBM Weighted", lgbm, X_train, y_train, X_test, y_test
    )

    # 6. XGBoost (Mapeo numérico para XGBoost)
    label_map = {"Local": 0, "Empate": 1, "Visita": 2}
    inv_label_map = {0: "Local", 1: "Empate", 2: "Visita"}
    y_train_num = y_train.map(label_map)
    y_test_num = y_test.map(label_map)

    xgb = XGBClassifier(
        n_estimators=150, max_depth=3, learning_rate=0.05,
        eval_metric="mlogloss", random_state=42
    )
    xgb.fit(X_train, y_train_num)
    xgb_pred_num = xgb.predict(X_test)
    xgb_pred = pd.Series(xgb_pred_num).map(inv_label_map)
    xgb_proba = xgb.predict_proba(X_test)

    acc = accuracy_score(y_test, xgb_pred)
    macro_f1 = f1_score(y_test, xgb_pred, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_test, xgb_pred, average="weighted", zero_division=0)
    loss = log_loss(y_test, xgb_proba, labels=CLASSES)
    clf_rep = classification_report(y_test, xgb_pred, target_names=CLASSES, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_test, xgb_pred, labels=CLASSES)

    results["xgboost"] = {
        "model_name": "XGBoost",
        "accuracy": round(acc, 4),
        "macro_f1": round(macro_f1, 4),
        "weighted_f1": round(weighted_f1, 4),
        "log_loss": round(loss, 4),
        "per_class": {
            cls_name: {
                "precision": round(clf_rep[cls_name]["precision"], 4),
                "recall": round(clf_rep[cls_name]["recall"], 4),
                "f1_score": round(clf_rep[cls_name]["f1-score"], 4),
                "support": clf_rep[cls_name]["support"]
            }
            for cls_name in CLASSES
        },
        "confusion_matrix": cm.tolist()
    }
    joblib.dump(xgb, MODELS_DIR / "xgboost.joblib")

    # 7. FLAML AutoML (Optimización formal para macro_f1)
    logger.info("Iniciando búsqueda AutoML con FLAML (time_budget=60s, metric=macro_f1)...")
    automl = AutoML()
    automl_settings = {
        "time_budget": 60,
        "metric": "macro_f1",
        "task": "classification",
        "seed": 42,
        "estimator_list": ["lgbm", "rf", "extra_tree", "xgboost", "lrl1"],
        "verbose": 0
    }
    automl.fit(X_train=X_train, y_train=y_train, **automl_settings)

    results["flaml_automl"] = evaluate_model(
        "FLAML AutoML Best", automl, X_train, y_train, X_test, y_test
    )
    results["flaml_automl"]["best_estimator"] = automl.best_estimator
    results["flaml_automl"]["best_config"] = automl.best_config

    # Guardar métricas en JSON
    metrics_path = MODELS_DIR / "supervised_metrics_summary.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    logger.info(f"Métricas comparativas guardadas en: {metrics_path}")

    # Generar visualización diagnóstica de matrices de confusión
    plot_confusion_matrices(results)

    return results


def plot_confusion_matrices(results: Dict[str, Any]) -> None:
    """Genera gráfico comparativo de matrices de confusión normalizadas."""
    models_to_plot = [
        "baseline_most_frequent",
        "rf_unweighted",
        "rf_weighted",
        "logistic_regression",
        "lightgbm_weighted",
        "flaml_automl"
    ]

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for idx, mkey in enumerate(models_to_plot):
        data = results.get(mkey)
        if not data:
            continue
        cm = np.array(data["confusion_matrix"])
        # Normalizar por fila (True label)
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        sns.heatmap(
            cm_norm, annot=True, fmt=".2f", cmap="Blues",
            xticklabels=CLASSES, yticklabels=CLASSES, ax=axes[idx], cbar=False
        )
        axes[idx].set_title(f"{data['model_name']}\nMacro-F1: {data['macro_f1']:.3f} | Acc: {data['accuracy']:.3f}")
        axes[idx].set_xlabel("Predicción")
        axes[idx].set_ylabel("Valor Real")

    plt.tight_layout()
    fig_path = FIGURES_DIR / "supervised_confusion_matrices.png"
    plt.savefig(fig_path, dpi=300)
    plt.close()
    logger.info(f"Matriz de confusión comparativa guardada en: {fig_path}")


if __name__ == "__main__":
    res = run_supervised_experiments()
    print("\n" + "=" * 85)
    print(f"{'MODELO':<30} | {'ACCURACY':<9} | {'MACRO-F1':<9} | {'WEIGHTED-F1':<11} | {'RECALL EMPATE'}")
    print("=" * 85)
    for k, v in res.items():
        rec_empate = v["per_class"]["Empate"]["recall"]
        print(f"{v['model_name']:<30} | {v['accuracy']:<9.4f} | {v['macro_f1']:<9.4f} | {v['weighted_f1']:<11.4f} | {rec_empate:.4f}")
    print("=" * 85)
