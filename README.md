# Выявление аномалий в сетевом трафике (NSL-KDD)

Индивидуальный проект по дисциплине «Интеллектуальный анализ больших
данных» (магистратура, СПбГУТ).

Тема: «Выявление аномалий в сетевом трафике».

## Состав

- `analysis.py` — основной конвейер: загрузка NSL-KDD, предобработка,
  EDA, статистические гипотезы, обучение четырёх моделей обнаружения
  аномалий (Isolation Forest, LOF, One-Class SVM, K-Means + порог
  расстояния), формирование графиков и таблиц.
- `dashboard.py` — интерактивный дашборд на Streamlit с фильтрами,
  KPI-блоком и сравнением метрик.
- `screenshot_dashboard.py` — служебный скрипт получения скриншотов
  дашборда через Playwright (для иллюстраций в отчёте).
- `KDDTrain+.txt`, `KDDTest+.txt` — данные NSL-KDD (Canadian Institute
  for Cybersecurity).
- `figures/` — сгенерированные PNG-графики и скриншот дашборда.
- `descriptive_stats.csv`, `hypothesis_tests.csv`, `metrics.csv`,
  `summary.json` — численные артефакты.

## Запуск

```bash
pip install -r requirements.txt   # см. ниже состав
python analysis.py                # обработка данных и обучение моделей
streamlit run dashboard.py        # запуск дашборда (localhost:8501)
```

Минимальный набор зависимостей:

```
pandas
numpy
scipy
scikit-learn
matplotlib
seaborn
streamlit
plotly
python-docx
pymupdf
playwright
```

## Результаты

| Модель           | Precision | Recall | F1    | ROC-AUC | PR-AUC |
|------------------|-----------|--------|-------|---------|--------|
| Isolation Forest | 0,968     | 0,744  | 0,841 | 0,947   | 0,959  |
| LOF              | 0,867     | 0,842  | 0,854 | 0,871   | 0,839  |
| One-Class SVM    | 0,922     | 0,877  | 0,899 | 0,934   | 0,935  |
| **K-Means**      | **0,914** | **0,891** | **0,903** | **0,947** | **0,943** |

Лучшая модель по F1 — K-Means + порог расстояния (F1 = 0,903).
