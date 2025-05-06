import os
import difflib
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer

# --- Настройки ---
INPUT_XLSX = "input.xlsx"
OUTPUT_XLSX = "evaluation.xlsx"
REF_DIR = "ref"
_SENT_SPLIT = re.compile(r"(?<=[.?!])\s+")

# --- Загрузка данных ---
df = pd.read_excel(INPUT_XLSX, dtype=str).fillna("")
required = {"id", "URL", "lib_text"}
if not required.issubset(df.columns):
    raise RuntimeError(f"В input.xlsx должны быть колонки: {required}")


# --- Чтение эталонных текстов ---
def load_ref(id_):
    path = os.path.join(REF_DIR, f"{id_}.txt")
    return open(path, encoding="utf-8").read() if os.path.exists(path) else ""


df["reference_text"] = df["id"].apply(load_ref)


# --- Функция сравнения текстов ---
def compare_texts(ref: str, ext: str):
    rw, ew = ref.split(), ext.split()
    completeness = round(difflib.SequenceMatcher(None, rw, ew).ratio() * 100, 1)

    ref_sents = _SENT_SPLIT.split(ref)
    ext_sents = set(_SENT_SPLIT.split(ext))
    if not ref_sents:
        return completeness, "низкая"

    vec = TfidfVectorizer()
    X = vec.fit_transform(ref_sents)
    scores = X.sum(axis=1).A1

    lost = [s for s, sent in zip(scores, ref_sents) if sent not in ext_sents]
    if not lost:
        return completeness, "низкая"

    avg_lost = sum(lost) / len(lost)
    avg_all = scores.mean() or 1
    ratio = avg_lost / avg_all

    if ratio < 0.5:
        sig = "низкая"
    elif ratio < 1.0:
        sig = "средняя"
    else:
        sig = "высокая"

    return completeness, sig


# --- Расчёт метрик ---
metrics = df.apply(lambda r: compare_texts(r["reference_text"], r["lib_text"]), axis=1)
df[["completeness", "lost_significance"]] = pd.DataFrame(
    metrics.tolist(), index=df.index
)

df["ref_len"] = df["reference_text"].str.split().apply(len)
df["ext_len"] = df["lib_text"].str.split().apply(len)


# --- Генерация комментариев ---
def make_comment(row):
    comp = row["completeness"]
    sig = row["lost_significance"]

    # Базовая часть по значимости потерянного
    if sig == "высокая":
        base = "Пропущены ключевые предложения, требуется доработка парсинга лидов и цитат."
    elif sig == "средняя":
        base = "Пропущены отдельные важные фразы, можно добавить расширенные CSS‑селекторы."
    else:
        base = "Потери незначительны, алгоритм работает корректно."

    # Уточнение по полноте
    if comp < 50:
        extra = " Низкая полнота (<50%), проверить выбор корневого контейнера."
    elif comp < 80:
        extra = " Средняя полнота (50–80%), обратить внимание на теги <article>, <div>."
    else:
        extra = " Общая полнота высокая (≥80%)."

    return base + extra


df["comments"] = df.apply(make_comment, axis=1)

# --- Сохранение результата ---
cols = [
    "id",
    "ref_len",
    "ext_len",
    "completeness",
    "lost_significance",
    "comments",
    "URL",
]
df[cols].to_excel(OUTPUT_XLSX, index=False)
print(f"Done: {OUTPUT_XLSX}")
