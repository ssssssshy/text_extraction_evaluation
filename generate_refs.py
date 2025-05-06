from newspaper import Article
import pandas as pd
import os

# Читаем input.xlsx
df = pd.read_excel("input.xlsx", dtype=str).fillna("")

# Папка ref/
os.makedirs("ref", exist_ok=True)

for _, row in df.iterrows():
    id_ = row["id"]
    url = row["URL"]
    art = Article(url, language="ru")
    art.download()
    art.parse()
    text = art.text
    with open(f"ref/{id_}.txt", "w", encoding="utf-8") as f:
        f.write(text)
    print(f"Saved ref/{id_}.txt")
