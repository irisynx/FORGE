import pandas as pd
import re
import gc
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")

INPUT_CSV = "./ord_data/ord_processed_samples.csv"
OUTPUT_CSV = "./ord_data/ord_processed_samples_refined.csv"
CHUNK_SIZE = 50000

REMOVE_PATTERNS = [
    r"in a (?:[\w\-]+ )*flask(?: equipped with .+?)?",
    r"equipped with (?:a )?(?:mechanical |magnetic )?stirrer",
    r"fitted with (?:a )?condenser",
    r"according to the procedure described in example \d+",
    r"in a manner similar to example \d+",
    r"the process of claims? \d+(?:, \d+)*",
    r"as described in .+?",
    r"prepared (?:as|by) .+?",
    r"see .+?",
    r"U\.S\. Pat\. No\. [\d,]+",
]

QUANTITY_PATTERNS = [
    r"\b\d+(?:\.\d+)?\s*(?:g|mg|kg|mol|mmol|ml|l|liter|equiv|eq)\b\.?",
    r"\(\s*\d+(?:\.\d+)?\s*(?:g|mg|mol|mmol|ml|l)\s*\)",
    r"\b\d+(?:\.\d+)?\s*%",
]

UNIT_MAP = {
    r"\bdeg\.?\s*C\b": "C",
    r"\bdegrees?\s*C\b": "C",
    r"\bhours?\b": "h",
    r"\bmins?\b": "min",
    r"\bseconds?\b": "s",
}

def clean_chemical_text(text):
    if pd.isna(text) or text == "":
        return ""

    text = str(text).lower()

    for pat in REMOVE_PATTERNS:
        text = re.sub(pat, " ", text)

    for pat in QUANTITY_PATTERNS:
        text = re.sub(pat, " ", text)

    for pat, repl in UNIT_MAP.items():
        text = re.sub(pat, repl, text)

    text = re.sub(r"\b(of|a|an|the|to|in|with)\b", " ", text)

    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"\s+\.", ".", text)
    text = re.sub(r"\.\s*\.", ".", text)

    return text

def process_chunk(df_chunk):

    if "Procedure_Text" in df_chunk.columns:
        df_chunk["Procedure_Text"] = df_chunk["Procedure_Text"].apply(
            clean_chemical_text
        )

    if "Workup_Text" in df_chunk.columns:
        df_chunk["Workup_Text"] = df_chunk["Workup_Text"].apply(clean_chemical_text)

    return df_chunk

def main():
    print(f"📂 读取文件: {INPUT_CSV}")

    try:
        total_rows = sum(1 for _ in open(INPUT_CSV, "r", encoding="utf-8")) - 1
    except FileNotFoundError:
        print("❌ 文件未找到！")
        return

    reader = pd.read_csv(INPUT_CSV, chunksize=CHUNK_SIZE)
    first_chunk = True

    print(f"🚀 开始清洗文本 (总行数: {total_rows})...")

    with tqdm(total=total_rows, unit="rows") as pbar:
        for chunk in reader:
            cleaned_chunk = process_chunk(chunk)

            mode = "w" if first_chunk else "a"
            header = first_chunk

            cleaned_chunk.to_csv(OUTPUT_CSV, mode=mode, header=header, index=False)

            first_chunk = False
            pbar.update(len(chunk))

    print(f"✅ 清洗完成！结果已保存至: {OUTPUT_CSV}")

    print("\n🔍 效果预览 (前 3 条):")
    df_preview = pd.read_csv(OUTPUT_CSV, nrows=3)
    for i, row in df_preview.iterrows():
        print(f"--- Sample {i+1} ---")
        print(f"Original Proc: {row.get('Procedure_Text', '')[:100]}...")
        print(f"Original Work: {row.get('Workup_Text', '')[:100]}...")

if __name__ == "__main__":
    main()
