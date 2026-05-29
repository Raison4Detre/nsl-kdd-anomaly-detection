# -*- coding: utf-8 -*-
"""Интерактивный дашборд (Streamlit) для оценки и сравнения
алгоритмов обнаружения аномалий в сетевом трафике на NSL-KDD.

Запуск:
    streamlit run dashboard.py --server.port 8501
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

PROJECT_DIR = Path(__file__).resolve().parent

st.set_page_config(
    page_title="NSL-KDD · Обнаружение аномалий сетевого трафика",
    page_icon="🛡️",
    layout="wide",
)

st.title("Обнаружение аномалий в сетевом трафике – NSL-KDD")
st.caption("Интерактивный дашборд по результатам индивидуального проекта. "
           "Дисциплина «Интеллектуальный анализ больших данных», СПбГУТ, 2026 г.")


@st.cache_data(show_spinner=False)
def load_data():
    cols = [
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
    df = pd.read_csv(PROJECT_DIR / "KDDTrain+.txt", header=None, names=cols)
    classes = {
        "normal": "normal",
        "back": "DoS", "land": "DoS", "neptune": "DoS", "pod": "DoS",
        "smurf": "DoS", "teardrop": "DoS",
        "ipsweep": "Probe", "nmap": "Probe", "portsweep": "Probe",
        "satan": "Probe",
        "ftp_write": "R2L", "guess_passwd": "R2L", "imap": "R2L",
        "multihop": "R2L", "phf": "R2L", "spy": "R2L",
        "warezclient": "R2L", "warezmaster": "R2L",
        "buffer_overflow": "U2R", "loadmodule": "U2R", "perl": "U2R",
        "rootkit": "U2R",
    }
    df["attack_class"] = df["label"].map(classes).fillna("other_attack")
    df["is_anomaly"] = (df["label"] != "normal").astype(int)
    return df


@st.cache_data(show_spinner=False)
def load_metrics():
    with open(PROJECT_DIR / "summary.json", encoding="utf-8") as f:
        summary = json.load(f)
    metrics = pd.DataFrame(summary["metrics"])
    return summary, metrics


df = load_data()
summary, metrics = load_metrics()

# ============================ Sidebar ============================
st.sidebar.header("Фильтры")
proto = st.sidebar.multiselect(
    "Протокол", sorted(df["protocol_type"].unique()),
    default=sorted(df["protocol_type"].unique()),
)
classes = st.sidebar.multiselect(
    "Класс трафика", ["normal", "DoS", "Probe", "R2L", "U2R", "other_attack"],
    default=["normal", "DoS", "Probe", "R2L", "U2R"],
)
mask = df["protocol_type"].isin(proto) & df["attack_class"].isin(classes)
sub = df.loc[mask]

# ============================ KPI ============================
c1, c2, c3, c4 = st.columns(4)
c1.metric("Записей", f"{len(sub):,}".replace(",", " "))
c2.metric("Доля аномалий", f"{sub['is_anomaly'].mean()*100:.1f} %")
c3.metric("Уникальных сервисов", sub["service"].nunique())
c4.metric("Лучшая модель (F1)",
          f"{summary['best_model']} · "
          f"{max(m['f1'] for m in summary['metrics']):.3f}")

st.divider()

# ============================ Class distribution & protocol breakdown ============================
left, right = st.columns([3, 2])
with left:
    st.subheader("Распределение классов трафика")
    counts = (sub["attack_class"].value_counts()
              .reindex(["normal", "DoS", "Probe", "R2L", "U2R", "other_attack"],
                       fill_value=0))
    fig = px.bar(counts, labels={"index": "Класс", "value": "Количество"},
                 color=counts.index, color_discrete_sequence=px.colors.qualitative.Set2)
    fig.update_layout(showlegend=False, height=340)
    st.plotly_chart(fig, use_container_width=True)

with right:
    st.subheader("Структура по протоколам")
    proto_counts = sub.groupby(["protocol_type", "is_anomaly"]).size().reset_index(name="n")
    proto_counts["статус"] = proto_counts["is_anomaly"].map(
        {0: "normal", 1: "anomaly"})
    fig = px.bar(proto_counts, x="protocol_type", y="n", color="статус",
                 barmode="stack",
                 color_discrete_map={"normal": "#4c9be8", "anomaly": "#e8584c"})
    fig.update_layout(height=340)
    st.plotly_chart(fig, use_container_width=True)

# ============================ Feature distributions ============================
st.subheader("Распределение количественных признаков")
col = st.selectbox("Признак",
                   ["duration", "src_bytes", "dst_bytes", "count",
                    "serror_rate", "same_srv_rate", "dst_host_srv_count"])
log_y = st.checkbox("log-шкала", value=True)
fig = px.histogram(sub, x=col, color=sub["is_anomaly"].map({0: "normal", 1: "anomaly"}),
                   nbins=60, opacity=0.7, barmode="overlay",
                   color_discrete_map={"normal": "#4c9be8", "anomaly": "#e8584c"})
if log_y:
    fig.update_yaxes(type="log")
fig.update_layout(height=360, legend_title_text="Класс")
st.plotly_chart(fig, use_container_width=True)

# ============================ Metrics table & bar chart ============================
st.subheader("Сравнение алгоритмов обнаружения аномалий")
c1, c2 = st.columns([2, 3])
with c1:
    st.dataframe(metrics.round(3).set_index("model"), use_container_width=True)
with c2:
    melted = metrics.melt(id_vars="model",
                          value_vars=["precision", "recall", "f1", "roc_auc", "pr_auc"],
                          var_name="metric", value_name="value")
    fig = px.bar(melted, x="metric", y="value", color="model", barmode="group",
                 color_discrete_sequence=px.colors.qualitative.Set2)
    fig.update_yaxes(range=[0, 1.05])
    fig.update_layout(height=380)
    st.plotly_chart(fig, use_container_width=True)

# ============================ Top features ============================
st.subheader("Ключевые признаки, связанные с аномальностью трафика")
st.write(", ".join(f"`{f}`" for f in summary["top_corr_features"]))

st.caption("Источник данных: NSL-KDD (улучшенная версия KDD Cup 1999). "
           "Подготовлено для индивидуального проекта по дисциплине "
           "«Интеллектуальный анализ больших данных».")
