# Лабораторная работа №2 — Regularization и KNN

Системы искусственного интеллекта · Михайлов Петр Сергеевич, P3311  
Преподаватель — Королёва Юлия Александровна.

Линейная и логистическая регрессии из ЛР1 дополнены штрафами L1, L2 и ElasticNet. KNN использует евклидово и манхэттенское расстояния, голосование для классификации и среднее для регрессии. Градиенты и расстояния вычисляются тензорными операциями PyTorch. Разбиение, предобработка и метрики реализованы по формулам в исходниках работы.

## Состав проекта

| Файл | Назначение |
|---|---|
| [lab2.ipynb](lab2.ipynb) | Подготовка данных, подбор параметров, графики и оценка |
| `models/linear_regression.py`, `models/logistic_regression.py` | Регрессии с полным градиентным спуском |
| `models/regularization.py` | Штраф и добавка к градиенту весов |
| `models/knn.py` | Общая реализация KNN для двух задач |
| `preprocessing.py`, `split.py`, `evaluation.py` | Предобработка, разбиения и расчёт метрик |
| `data/` | Исходные данные |
| [Отчёт PDF](docs/lab2_report.pdf), [исходник TeX](docs/lab2_report.tex) | Методика, результаты и выводы |
| `docs/figures/` | Графики, создаваемые ноутбуком |

## Запуск

Из папки «Лабораторная 2» в Git Bash на Windows:

```bash
py -m venv .venv
./.venv/Scripts/python.exe -m pip install -r requirements.txt
./.venv/Scripts/python.exe -m notebook lab2.ipynb
```

В ноутбуке выполнить **Restart Kernel and Run All Cells**. При запуске из PyCharm выбрать интерпретатор проекта `.venv/Scripts/python.exe` и рабочую папку лабораторной. Расчёт работает на CPU. Для Linux и macOS используйте `python3 -m venv .venv` и `.venv/bin/python`.

## Данные и проверка качества

- `day.csv` — [Bike Sharing Dataset](https://www.kaggle.com/datasets/marklvl/bike-sharing-dataset), целевая переменная `cnt`.
- `train.csv` — [Spaceship Titanic](https://www.kaggle.com/competitions/spaceship-titanic/data), целевая переменная `Transported`. Размеченные train, validation и test выделяются из этого файла.
- `Dry_Bean_Dataset/Dry_Bean_Dataset.arff` — Murat Koklu и Ilker Ali Özkan, [Dry Bean Dataset](https://doi.org/10.24432/C50S4B), CC BY 4.0. Целевая переменная `Class`. При подготовке выборок удаляются точные повторы.
- `SeoulBikeData.csv` — [Seoul Bike Sharing Demand](https://doi.org/10.24432/C5F62R), UCI Machine Learning Repository, CC BY 4.0. Целевая переменная `Rented Bike Count`.

Для случайных разбиений используется `seed=5267`. Seoul Bike делится хронологически целыми днями. В каждом эксперименте параметры предобработки определяются по train, гиперпараметры — по validation. Итоговая оценка проводится на test.
