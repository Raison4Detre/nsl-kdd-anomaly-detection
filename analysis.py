# -*- coding: utf-8 -*-
"""
Индивидуальный проект по дисциплине
«Интеллектуальный анализ больших данных»
Тема: «Выявление аномалий в сетевом трафике»

Датасет: NSL-KDD (усовершенствованная версия KDD Cup 1999).

Решаемая задача – обнаружение аномального сетевого трафика
(сетевых атак) методами интеллектуального анализа данных:
 – Isolation Forest;
 – Local Outlier Factor (LOF);
 – One-Class SVM;
 – K-Means + порог расстояния.

Скрипт выполняет полный конвейер: загрузка данных,
предобработка, разведочный анализ, формальная проверка
статистических гипотез и построение моделей с оценкой качества.
Все результаты (рисунки, сводные таблицы, метрики) сохраняются
в каталоге figures/ и отдельных JSON / CSV артефактах.
"""
from __future__ import annotations

import json
import os
import time
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", font_scale=1.0)
plt.rcParams["font.family"] = "DejaVu Sans"

PROJECT_DIR = Path(__file__).resolve().parent
FIG_DIR = PROJECT_DIR / "figures"
FIG_DIR.mkdir(exist_ok=True)
RANDOM_STATE = 20250526

# ---------------------------------------------------------------------------
# 1. Загрузка и описание признаков NSL-KDD
# ---------------------------------------------------------------------------
COLUMN_NAMES = [
    "duration", "protocol_type", "service", "flag",
    "src_bytes", "dst_bytes", "land", "wrong_fragment", "urgent",
    "hot", "num_failed_logins", "logged_in", "num_compromised",
    "root_shell", "su_attempted", "num_root", "num_file_creations",
    "num_shells", "num_access_files", "num_outbound_cmds",
    "is_host_login", "is_guest_login", "count", "srv_count",
    "serror_rate", "srv_serror_rate", "rerror_rate", "srv_rerror_rate",
    "same_srv_rate", "diff_srv_rate", "srv_diff_host_rate",
    "dst_host_count", "dst_host_srv_count", "dst_host_same_srv_rate",
    "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate", "dst_host_serror_rate",
    "dst_host_srv_serror_rate", "dst_host_rerror_rate",
    "dst_host_srv_rerror_rate", "label", "difficulty",
]

CATEGORICAL = ["protocol_type", "service", "flag"]
BINARY = [
    "land", "logged_in", "root_shell", "su_attempted",
    "is_host_login", "is_guest_login",
]

# Группировка типов атак NSL-KDD на 4 класса (DoS / Probe / R2L / U2R) + normal
ATTACK_CLASSES = {
    "normal": "normal",
    "back": "DoS", "land": "DoS", "neptune": "DoS", "pod": "DoS",
    "smurf": "DoS", "teardrop": "DoS", "mailbomb": "DoS",
    "apache2": "DoS", "processtable": "DoS", "udpstorm": "DoS",
    "worm": "DoS",
    "ipsweep": "Probe", "nmap": "Probe", "portsweep": "Probe",
    "satan": "Probe", "mscan": "Probe", "saint": "Probe",
    "ftp_write": "R2L", "guess_passwd": "R2L", "imap": "R2L",
    "multihop": "R2L", "phf": "R2L", "spy": "R2L",
    "warezclient": "R2L", "warezmaster": "R2L", "sendmail": "R2L",
    "named": "R2L", "snmpgetattack": "R2L", "snmpguess": "R2L",
    "xlock": "R2L", "xsnoop": "R2L", "httptunnel": "R2L",
    "buffer_overflow": "U2R", "loadmodule": "U2R", "perl": "U2R",
    "rootkit": "U2R", "ps": "U2R", "sqlattack": "U2R", "xterm": "U2R",
}


def load_nsl_kdd() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Читает train/test файлы NSL-KDD и присваивает имена признакам."""
    train = pd.read_csv(PROJECT_DIR / "KDDTrain+.txt",
                        header=None, names=COLUMN_NAMES)
    test = pd.read_csv(PROJECT_DIR / "KDDTest+.txt",
                       header=None, names=COLUMN_NAMES)
    for df in (train, test):
        df["attack_class"] = df["label"].map(ATTACK_CLASSES).fillna("unknown")
        df["is_anomaly"] = (df["label"] != "normal").astype(int)
    return train, test


# ---------------------------------------------------------------------------
# 2. Предобработка
# ---------------------------------------------------------------------------
def preprocess(df: pd.DataFrame, encoders: Dict | None = None,
               scaler: StandardScaler | None = None,
               fit: bool = True) -> Tuple[pd.DataFrame, Dict, StandardScaler]:
    """Очистка, кодирование, масштабирование признаков NSL-KDD.

    – Удаление дубликатов и константных признаков.
    – One-Hot кодирование для protocol_type, service, flag.
    – Логарифмическое преобразование сильно-скошенных
      количественных признаков (duration, src_bytes, dst_bytes).
    – Создание двух производных признаков:
      bytes_ratio = log1p(src_bytes) − log1p(dst_bytes),
      error_rate  = mean([serror_rate, rerror_rate]).
    – Стандартизация (z-нормировка) численных признаков.
    """
    df = df.drop_duplicates().reset_index(drop=True)

    # Производные признаки
    df["log_duration"] = np.log1p(df["duration"])
    df["log_src_bytes"] = np.log1p(df["src_bytes"])
    df["log_dst_bytes"] = np.log1p(df["dst_bytes"])
    df["bytes_ratio"] = df["log_src_bytes"] - df["log_dst_bytes"]
    df["error_rate"] = df[["serror_rate", "rerror_rate"]].mean(axis=1)

    # Удалим заведомо константные / служебные столбцы
    df = df.drop(columns=["num_outbound_cmds"], errors="ignore")

    # One-Hot encoding
    if encoders is None:
        encoders = {}
    cat_frames = []
    for col in CATEGORICAL:
        if fit:
            cats = sorted(df[col].astype(str).unique())
            encoders[col] = cats
        cats = encoders[col]
        dummies = pd.get_dummies(
            pd.Categorical(df[col].astype(str), categories=cats),
            prefix=col, dtype=int,
        )
        cat_frames.append(dummies)
    df_enc = pd.concat([df.drop(columns=CATEGORICAL), *cat_frames], axis=1)

    # Численные признаки для масштабирования
    drop_for_scale = {
        "label", "attack_class", "is_anomaly", "difficulty",
        *CATEGORICAL,
    }
    numeric_cols = [c for c in df_enc.columns
                    if c not in drop_for_scale and df_enc[c].dtype != object]

    if fit:
        scaler = StandardScaler()
        df_enc[numeric_cols] = scaler.fit_transform(df_enc[numeric_cols])
    else:
        df_enc[numeric_cols] = scaler.transform(df_enc[numeric_cols])

    return df_enc, encoders, scaler


# ---------------------------------------------------------------------------
# 3. Разведочный анализ
# ---------------------------------------------------------------------------
def descriptive_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """Описательные статистики ключевых числовых признаков."""
    cols = [
        "duration", "src_bytes", "dst_bytes", "count", "srv_count",
        "serror_rate", "rerror_rate", "same_srv_rate", "diff_srv_rate",
        "dst_host_count", "dst_host_srv_count",
    ]
    stats_df = df[cols].describe(percentiles=[0.25, 0.5, 0.75]).T
    stats_df["skewness"] = df[cols].skew()
    stats_df["kurtosis"] = df[cols].kurtosis()
    return stats_df.round(4)


def plot_class_distribution(df: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    order = ["normal", "DoS", "Probe", "R2L", "U2R"]
    counts = df["attack_class"].value_counts().reindex(order, fill_value=0)
    sns.barplot(x=counts.index, y=counts.values,
                hue=counts.index, palette="Set2",
                legend=False, ax=ax)
    for i, v in enumerate(counts.values):
        ax.text(i, v + counts.max() * 0.01, f"{v:,}".replace(",", " "),
                ha="center", fontsize=10)
    ax.set_xlabel("Класс трафика")
    ax.set_ylabel("Количество записей")
    ax.set_title("Распределение классов в обучающей выборке NSL-KDD")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_feature_distributions(df: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    feats = [("log_duration", "log(1 + duration)"),
             ("log_src_bytes", "log(1 + src_bytes)"),
             ("log_dst_bytes", "log(1 + dst_bytes)")]
    for ax, (col, lbl) in zip(axes, feats):
        for cls, color in zip(["normal", "anomaly"], ["#4c9be8", "#e88a4c"]):
            data = df.loc[df["is_anomaly"] == int(cls == "anomaly"), col]
            ax.hist(data, bins=50, density=True, alpha=0.55,
                    color=color, label=cls)
        ax.set_xlabel(lbl)
        ax.set_ylabel("Плотность")
        ax.legend()
    fig.suptitle("Сравнение распределений признаков для нормального и аномального трафика")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_boxplots(df: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    items = [
        ("serror_rate", "Доля SYN-ошибок"),
        ("same_srv_rate", "Доля обращений к одному сервису"),
        ("dst_host_srv_count", "dst_host_srv_count"),
    ]
    for ax, (col, title) in zip(axes, items):
        sns.boxplot(data=df, x="attack_class", y=col,
                    order=["normal", "DoS", "Probe", "R2L", "U2R"],
                    ax=ax, palette="Set2", hue="attack_class", legend=False)
        ax.set_title(title)
        ax.set_xlabel("")
    fig.suptitle("Boxplot-сравнение ключевых признаков по классам атак")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_correlation_matrix(df: pd.DataFrame, path: Path) -> List[str]:
    cols = [
        "duration", "src_bytes", "dst_bytes", "count", "srv_count",
        "serror_rate", "srv_serror_rate", "rerror_rate", "srv_rerror_rate",
        "same_srv_rate", "diff_srv_rate", "dst_host_count",
        "dst_host_srv_count", "dst_host_same_srv_rate",
        "dst_host_serror_rate", "dst_host_rerror_rate",
        "is_anomaly",
    ]
    corr = df[cols].corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, annot=False, cmap="RdBu_r", center=0,
                vmin=-1, vmax=1, ax=ax, cbar_kws={"shrink": 0.7})
    ax.set_title("Матрица корреляций ключевых признаков (Пирсон)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)

    # Признаки, наиболее коррелированные с is_anomaly
    top = corr["is_anomaly"].drop("is_anomaly").abs().sort_values(ascending=False).head(5).index.tolist()
    return top


# ---------------------------------------------------------------------------
# 4. Проверка статистических гипотез
# ---------------------------------------------------------------------------
def hypothesis_tests(df_raw: pd.DataFrame) -> pd.DataFrame:
    """Формальная проверка гипотез о различии в распределении признаков
    между нормальным и аномальным трафиком.

    – χ² Пирсона: protocol_type / service / flag vs is_anomaly.
    – Манн-Уитни U: численные признаки vs is_anomaly.
    – ANOVA Краскела-Уоллеса: численные признаки по 5 классам атак.
    """
    rows = []

    for col in CATEGORICAL:
        tab = pd.crosstab(df_raw[col], df_raw["is_anomaly"])
        chi2, p, dof, _ = stats.chi2_contingency(tab)
        rows.append({
            "feature": col, "test": "χ² Пирсона", "statistic": chi2,
            "df": dof, "p_value": p,
            "decision": "H0 отвергается" if p < 0.001
            else ("H0 отвергается" if p < 0.05 else "не отвергается"),
        })

    numeric = ["duration", "src_bytes", "dst_bytes", "count",
               "serror_rate", "same_srv_rate", "dst_host_srv_count"]
    normal = df_raw[df_raw["is_anomaly"] == 0]
    anomaly = df_raw[df_raw["is_anomaly"] == 1]
    for col in numeric:
        u, p = stats.mannwhitneyu(normal[col], anomaly[col],
                                  alternative="two-sided")
        rows.append({
            "feature": col, "test": "Манн-Уитни U", "statistic": u,
            "df": float("nan"), "p_value": p,
            "decision": "H0 отвергается" if p < 0.05 else "не отвергается",
        })

    for col in numeric:
        groups = [g[col].values for _, g in df_raw.groupby("attack_class")
                  if len(g) > 30]
        h, p = stats.kruskal(*groups)
        rows.append({
            "feature": col, "test": "Краскел-Уоллис", "statistic": h,
            "df": len(groups) - 1, "p_value": p,
            "decision": "H0 отвергается" if p < 0.05 else "не отвергается",
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 5. Модели обнаружения аномалий
# ---------------------------------------------------------------------------
def train_isolation_forest(X_train: np.ndarray, X_test: np.ndarray,
                           contamination: float) -> Tuple[np.ndarray, np.ndarray, float]:
    model = IsolationForest(
        n_estimators=200, max_samples="auto",
        contamination=contamination, random_state=RANDOM_STATE, n_jobs=-1,
    )
    start = time.time()
    model.fit(X_train)
    elapsed = time.time() - start
    scores = -model.score_samples(X_test)
    pred = (model.predict(X_test) == -1).astype(int)
    return scores, pred, elapsed


def train_lof(X_train: np.ndarray, X_test: np.ndarray,
              contamination: float, sample_size: int = 20_000) \
        -> Tuple[np.ndarray, np.ndarray, float]:
    """Local Outlier Factor – на полной обучающей выборке слишком тяжёл,
    поэтому работаем на подвыборке размером ≤ sample_size."""
    rng = np.random.default_rng(RANDOM_STATE)
    if len(X_train) > sample_size:
        idx = rng.choice(len(X_train), sample_size, replace=False)
        X_fit = X_train[idx]
    else:
        X_fit = X_train
    model = LocalOutlierFactor(
        n_neighbors=20, contamination=contamination,
        novelty=True, n_jobs=-1,
    )
    start = time.time()
    model.fit(X_fit)
    elapsed = time.time() - start
    scores = -model.decision_function(X_test)
    pred = (model.predict(X_test) == -1).astype(int)
    return scores, pred, elapsed


def train_ocsvm(X_train: np.ndarray, X_test: np.ndarray,
                contamination: float, sample_size: int = 10_000) \
        -> Tuple[np.ndarray, np.ndarray, float]:
    """One-Class SVM с RBF-ядром обучается на подвыборке (квадратичный рост)."""
    rng = np.random.default_rng(RANDOM_STATE)
    if len(X_train) > sample_size:
        idx = rng.choice(len(X_train), sample_size, replace=False)
        X_fit = X_train[idx]
    else:
        X_fit = X_train
    model = OneClassSVM(kernel="rbf", gamma="scale", nu=contamination)
    start = time.time()
    model.fit(X_fit)
    elapsed = time.time() - start
    scores = -model.decision_function(X_test)
    pred = (model.predict(X_test) == -1).astype(int)
    return scores, pred, elapsed


def train_kmeans_distance(X_train: np.ndarray, X_test: np.ndarray,
                          contamination: float, k: int = 16) \
        -> Tuple[np.ndarray, np.ndarray, float]:
    """K-Means + порог расстояния: чем дальше точка от ближайшего центра,
    тем выше её аномальность."""
    model = KMeans(n_clusters=k, n_init=10, random_state=RANDOM_STATE)
    start = time.time()
    model.fit(X_train)
    elapsed = time.time() - start
    train_dists = np.min(model.transform(X_train), axis=1)
    threshold = np.quantile(train_dists, 1 - contamination)
    test_dists = np.min(model.transform(X_test), axis=1)
    pred = (test_dists > threshold).astype(int)
    return test_dists, pred, elapsed


def evaluate(name: str, scores: np.ndarray, pred: np.ndarray,
             y_true: np.ndarray, elapsed: float) -> Dict:
    return {
        "model": name,
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred, zero_division=0),
        "f1": f1_score(y_true, pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, scores),
        "pr_auc": average_precision_score(y_true, scores),
        "train_sec": elapsed,
    }


# ---------------------------------------------------------------------------
# 6. Визуализация результатов моделей
# ---------------------------------------------------------------------------
def plot_roc_curves(results: Dict[str, Tuple[np.ndarray, np.ndarray]],
                    y_true: np.ndarray, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    for name, (scores, _) in results.items():
        fpr, tpr, _ = roc_curve(y_true, scores)
        auc = roc_auc_score(y_true, scores)
        ax.plot(fpr, tpr, label=f"{name} (AUC = {auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4, label="Случайный классификатор")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC-кривые алгоритмов обнаружения аномалий")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_pr_curves(results: Dict[str, Tuple[np.ndarray, np.ndarray]],
                   y_true: np.ndarray, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    for name, (scores, _) in results.items():
        precision, recall, _ = precision_recall_curve(y_true, scores)
        ap = average_precision_score(y_true, scores)
        ax.plot(recall, precision, label=f"{name} (AP = {ap:.3f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall кривые алгоритмов")
    ax.legend(loc="lower left")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_confusion(name: str, pred: np.ndarray, y_true: np.ndarray,
                   path: Path) -> None:
    cm = confusion_matrix(y_true, pred)
    fig, ax = plt.subplots(figsize=(4.5, 4.0))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["norm", "anom"], yticklabels=["norm", "anom"], ax=ax)
    ax.set_xlabel("Предсказание")
    ax.set_ylabel("Истинная метка")
    ax.set_title(f"Матрица ошибок – {name}")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_pca_scatter(X: np.ndarray, y: np.ndarray, attack_class: np.ndarray,
                     path: Path) -> None:
    """2D-проекция (PCA) обучающей выборки с раскраской по типу атак."""
    rng = np.random.default_rng(RANDOM_STATE)
    idx = rng.choice(len(X), min(15_000, len(X)), replace=False)
    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    Z = pca.fit_transform(X[idx])
    fig, ax = plt.subplots(figsize=(9, 6))
    colors = {"normal": "#4c9be8", "DoS": "#e8584c",
              "Probe": "#f0c419", "R2L": "#9b59b6", "U2R": "#2ecc71"}
    for cls, color in colors.items():
        mask = attack_class[idx] == cls
        ax.scatter(Z[mask, 0], Z[mask, 1], s=6, alpha=0.5,
                   color=color, label=f"{cls} ({mask.sum()})")
    ax.set_xlabel(f"PC-1 ({pca.explained_variance_ratio_[0]*100:.1f} %)")
    ax.set_ylabel(f"PC-2 ({pca.explained_variance_ratio_[1]*100:.1f} %)")
    ax.set_title("PCA-проекция обучающей выборки NSL-KDD")
    ax.legend(loc="best", markerscale=2.0)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_metrics_bars(metrics_df: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    melted = metrics_df.melt(id_vars="model",
                             value_vars=["precision", "recall", "f1", "roc_auc", "pr_auc"],
                             var_name="metric", value_name="value")
    sns.barplot(data=melted, x="metric", y="value", hue="model",
                palette="Set2", ax=ax)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("")
    ax.set_ylabel("Значение метрики")
    ax.set_title("Сравнение алгоритмов обнаружения аномалий")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 7. Главный конвейер
# ---------------------------------------------------------------------------
def main():
    print("[1/7] Загрузка данных NSL-KDD…")
    train_raw, test_raw = load_nsl_kdd()
    print(f"  train shape = {train_raw.shape}, test shape = {test_raw.shape}")
    print(f"  доля аномалий: train = {train_raw['is_anomaly'].mean():.3f}, "
          f"test = {test_raw['is_anomaly'].mean():.3f}")

    print("[2/7] Предобработка…")
    train_enc, encoders, scaler = preprocess(train_raw, fit=True)
    test_enc, _, _ = preprocess(test_raw, encoders=encoders,
                                scaler=scaler, fit=False)

    # Согласование колонок (test может потерять некоторые service-категории)
    for col in train_enc.columns:
        if col not in test_enc.columns:
            test_enc[col] = 0
    test_enc = test_enc[train_enc.columns]

    feature_cols = [c for c in train_enc.columns
                    if c not in ("label", "attack_class", "is_anomaly",
                                 "difficulty")]
    X_train = train_enc[feature_cols].to_numpy(dtype=np.float32)
    y_train = train_enc["is_anomaly"].to_numpy(dtype=int)
    X_test = test_enc[feature_cols].to_numpy(dtype=np.float32)
    y_test = test_enc["is_anomaly"].to_numpy(dtype=int)
    print(f"  итоговая размерность признакового пространства: {X_train.shape[1]}")

    # Только нормальный трафик для обучения unsupervised-моделей
    X_train_norm = X_train[y_train == 0]
    print(f"  нормальных записей для обучения: {len(X_train_norm):,}".replace(",", " "))

    print("[3/7] Описательная статистика и EDA…")
    desc = descriptive_statistics(train_raw)
    desc.to_csv(PROJECT_DIR / "descriptive_stats.csv", encoding="utf-8")
    plot_class_distribution(train_raw, FIG_DIR / "class_distribution.png")
    plot_feature_distributions(train_raw.assign(
        log_duration=np.log1p(train_raw["duration"]),
        log_src_bytes=np.log1p(train_raw["src_bytes"]),
        log_dst_bytes=np.log1p(train_raw["dst_bytes"]),
    ), FIG_DIR / "feature_distributions.png")
    plot_boxplots(train_raw, FIG_DIR / "boxplots.png")
    top_corr = plot_correlation_matrix(train_raw,
                                       FIG_DIR / "correlation_matrix.png")
    print(f"  топ-5 признаков по |corr| c is_anomaly: {top_corr}")

    print("[4/7] Проверка статистических гипотез…")
    hyp_df = hypothesis_tests(train_raw)
    hyp_df.to_csv(PROJECT_DIR / "hypothesis_tests.csv",
                  index=False, encoding="utf-8")
    print(hyp_df.to_string(index=False))

    print("[5/7] PCA-проекция…")
    plot_pca_scatter(X_train, y_train,
                     train_raw["attack_class"].iloc[:len(X_train)].to_numpy(),
                     FIG_DIR / "pca_scatter.png")

    print("[6/7] Обучение моделей и оценка качества…")
    contamination = 0.10  # доля выбросов, подаваемая моделям (грубая оценка)
    results: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    metrics: List[Dict] = []

    print("  – Isolation Forest")
    s, p, t = train_isolation_forest(X_train_norm, X_test, contamination)
    results["Isolation Forest"] = (s, p)
    metrics.append(evaluate("Isolation Forest", s, p, y_test, t))

    print("  – Local Outlier Factor")
    s, p, t = train_lof(X_train_norm, X_test, contamination)
    results["LOF"] = (s, p)
    metrics.append(evaluate("LOF", s, p, y_test, t))

    print("  – One-Class SVM")
    s, p, t = train_ocsvm(X_train_norm, X_test, contamination)
    results["One-Class SVM"] = (s, p)
    metrics.append(evaluate("One-Class SVM", s, p, y_test, t))

    print("  – K-Means + порог расстояния")
    s, p, t = train_kmeans_distance(X_train_norm, X_test, contamination)
    results["K-Means"] = (s, p)
    metrics.append(evaluate("K-Means", s, p, y_test, t))

    metrics_df = pd.DataFrame(metrics).round(4)
    metrics_df.to_csv(PROJECT_DIR / "metrics.csv",
                      index=False, encoding="utf-8")
    print(metrics_df.to_string(index=False))

    print("[7/7] Визуализация результатов…")
    plot_roc_curves(results, y_test, FIG_DIR / "roc_curves.png")
    plot_pr_curves(results, y_test, FIG_DIR / "pr_curves.png")
    plot_metrics_bars(metrics_df, FIG_DIR / "metrics_bars.png")
    best = metrics_df.sort_values("f1", ascending=False).iloc[0]["model"]
    plot_confusion(best, results[best][1], y_test,
                   FIG_DIR / f"confusion_{best.replace(' ', '_')}.png")
    print(f"  лучшая модель по F1: {best}")

    # Сохраним финальную сводку для отчёта
    summary = {
        "n_train": int(len(train_raw)),
        "n_test": int(len(test_raw)),
        "n_features_raw": 41,
        "n_features_processed": int(X_train.shape[1]),
        "anomaly_share_train": float(train_raw["is_anomaly"].mean()),
        "anomaly_share_test": float(test_raw["is_anomaly"].mean()),
        "best_model": best,
        "metrics": metrics_df.to_dict(orient="records"),
        "top_corr_features": top_corr,
    }
    with open(PROJECT_DIR / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print("Готово. Артефакты:")
    for p in sorted(FIG_DIR.iterdir()):
        print(f"  – figures/{p.name}")
    print("  – descriptive_stats.csv, hypothesis_tests.csv, "
          "metrics.csv, summary.json")


if __name__ == "__main__":
    main()
