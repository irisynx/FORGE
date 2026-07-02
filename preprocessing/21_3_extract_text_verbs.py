import pandas as pd
import spacy
import re
import os
from tqdm import tqdm

INPUT_FILE = "ord_data/ord_processed_samples_refined_cleaned_dataset_sane.csv"
OUTPUT_FILE = "ord_data/ord_processed_samples_refined_cleaned_dataset_sane_features.csv"
CHUNK_SIZE = 5000
N_PROCESS = 4

nlp = spacy.load("en_core_web_sm", disable=["ner", "parser", "lemmatizer"])

ACTION_WHITELIST = {
    "filtration",
    "extraction",
    "evaporation",
    "distillation",
    "recrystallization",
    "crystallization",
    "separation",
    "purification",
    "chromatography",
    "reflux",
    "heating",
    "cooling",
    "stirring",
    "washing",
    "drying",
    "concentration",
    "synthesis",
    "reaction",
    "workup",
    "addition",
    "irradiation",
    "precipitation",
    "centrifugation",
    "sublimation",
    "trituration",
    "added",
    "mixed",
    "stirred",
}

CONTEXT_BLACKLIST = {
    "solution",
    "mixture",
    "residue",
    "layer",
    "phase",
    "product",
    "compound",
    "solid",
    "oil",
    "crystals",
    "filtrate",
    "precipitate",
    "solvent",
    "water",
    "brine",
    "acid",
    "base",
    "yield",
    "title",
    "ether",
    "acetate",
    "methanol",
    "ethanol",
    "hcl",
    "naoh",
    "mgso4",
    "organic",
    "aqueous",
    "crude",
    "pure",
    "anhydrous",
    "saturated",
    "diluted",
    "volatiles",
    "mother",
    "liquor",
}

def regex_clean(text):
    """第一步：快速正则清洗 (CPU 密集型，不涉及 NLP)"""
    if not isinstance(text, str) or not text.strip():
        return ""

    text = text.lower()
    text = re.sub(r"\(.*?\)", "", text)
    text = re.sub(r"\b(mp|m\.p\.|bp|b\.p\.|nmr|ir|ms)\b.*", "", text)
    text = re.sub(r"\d+\.?\d*\s*(°|deg|c\.|k\.|f\.)", "", text)
    text = re.sub(r"\d+\.?\d*\s*(h|hr|hours|min|minutes|s|sec|days)\b", "", text)
    text = re.sub(r"\d+\.?\d*\s*(g|mg|kg|l|ml|ul|mmol|mol|eq|wt|%|m|n)\b", "", text)
    text = re.sub(r"\d+", "", text)
    return text

def batch_process_text(texts, n_process):
    """
    使用 spaCy 的 pipe 进行批量、多进程处理
    """

    pre_cleaned_texts = [regex_clean(t) for t in texts]

    results = []

    for doc in nlp.pipe(pre_cleaned_texts, batch_size=1000, n_process=n_process):
        kept_tokens = []
        for token in doc:
            word = token.text

            if len(word) < 2 or token.is_stop:
                continue
            if word in ACTION_WHITELIST:
                kept_tokens.append(word)
                continue
            if word in CONTEXT_BLACKLIST:
                continue
            if token.pos_ in ["VERB", "ADV", "PART"]:
                kept_tokens.append(word)

        results.append(" ".join(kept_tokens))

    return results

def main():

    if not os.path.exists(INPUT_FILE):
        print(f"错误: 找不到文件 {INPUT_FILE}")
        return

    print("正在计算文件总行数...")
    total_rows = sum(1 for _ in open(INPUT_FILE, "r", encoding="utf-8")) - 1
    print(f"总行数: {total_rows}")

    if os.path.exists(OUTPUT_FILE):
        os.remove(OUTPUT_FILE)

    reader = pd.read_csv(INPUT_FILE, chunksize=CHUNK_SIZE)

    target_cols = ["Procedure_Text", "Workup_Text"]

    with tqdm(total=total_rows, unit="row") as pbar:
        for i, chunk in enumerate(reader):

            for col in target_cols:

                texts = chunk[col].fillna("").tolist()

                processed_texts = batch_process_text(texts, n_process=N_PROCESS)

                chunk[col] = processed_texts

            write_header = i == 0
            chunk.to_csv(OUTPUT_FILE, mode="a", index=False, header=write_header)

            pbar.update(len(chunk))

    print(f"\n处理完成！结果已保存至 {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
