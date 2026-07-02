import os
import gzip
import glob
import csv
import re
import gc
from tqdm import tqdm
from ord_schema.proto import dataset_pb2
from ord_schema.proto import reaction_pb2
from rdkit import RDLogger
from bs4 import BeautifulSoup

RDLogger.DisableLog("rdApp.*")

ORD_DATA_DIR = "./ord-data/data"
OUTPUT_DIR = "./ord-data"
CSV_PATH = os.path.join(OUTPUT_DIR, "ord_processed_samples.csv")
MAX_FILES = None
DEBUG_MODE = True

class TextExtractorV5:
    def __init__(self):

        self.time_patterns = [

            re.compile(
                r"(?:for|during|over)\s+(?:a period of\s+)?(?:about\s+)?(?P<val1>\d+(?:\.\d+)?)\s*(?:[-–to]\s*(?P<val2>\d+(?:\.\d+)?))?\s*(?P<unit>h|hr|hrs|hours?|min|mins?|minutes?|d|days?)",
                re.I,
            ),

            re.compile(
                r"(?:stirred|refluxed|heated)\s+(?:about\s+)?(?P<val1>\d+(?:\.\d+)?)\s*(?:[-–to]\s*(?P<val2>\d+(?:\.\d+)?))?\s*(?P<unit>h|hr|hrs|hours?|min|mins?|minutes?|d|days?)",
                re.I,
            ),

            re.compile(
                r"(?:reaction time|duration)[:\s]+(?P<val1>\d+(?:\.\d+)?)\s*(?P<unit>h|hr|hrs|hours?|min|mins?|minutes?|d|days?)",
                re.I,
            ),
        ]

        self.temp_pattern = re.compile(
            r"(?<!m\.p\.\s)(?<!b\.p\.\s)(?<!melting point\s)(?<!boiling point\s)"
            r"(?:at|to|kept at|maintained at)\s+(?:about\s+)?(?P<val1>-?\d+(?:\.\d+)?)\s*(?:°|deg|degrees?)?\s*(?:[-–to]\s*(?P<val2>-?\d+(?:\.\d+)?))?\s*(?:°|deg|degrees?)?\s*(?P<unit>C|F|K)",
            re.I,
        )

    def parse_time(self, text):
        if not text:
            return None
        text = text.replace("\n", " ").strip()

        if "overnight" in text.lower():
            return 16.0

        for pat in self.time_patterns:
            match = pat.search(text)
            if match:
                v1 = float(match.group("val1"))
                v2 = (
                    float(match.group("val2"))
                    if "val2" in match.groupdict() and match.group("val2")
                    else None
                )
                unit = match.group("unit").lower()

                val = (v1 + v2) / 2 if v2 else v1

                if "min" in unit:
                    return val / 60.0
                if "d" in unit:
                    return val * 24.0
                return val
        return None

    def parse_temp(self, text):
        if not text:
            return None
        text = text.replace("\n", " ").strip()

        if any(x in text.lower() for x in ["room temp", "rt", "r.t.", "ambient"]):
            return 25.0
        if "ice bath" in text.lower() or "0 c" in text.lower():
            return 0.0

        match = self.temp_pattern.search(text)
        if match:
            v1 = float(match.group("val1"))
            v2 = float(match.group("val2")) if match.group("val2") else None
            unit = match.group("unit").upper()
            val = (v1 + v2) / 2 if v2 else v1

            if unit == "K":
                return val - 273.15
            if unit == "F":
                return (val - 32) * 5 / 9
            return val
        return None

extractor = TextExtractorV5()

def load_pb_gz(filename):
    try:
        with gzip.open(filename, "rb") as f:
            content = f.read()
    except:
        with open(filename, "rb") as f:
            if f.read(100).startswith(b"version https://git-lfs"):
                return None
            f.seek(0)
            content = f.read()
    if not content:
        return None
    try:
        ds = dataset_pb2.Dataset()
        ds.ParseFromString(content)
        return ds
    except:
        return None

def clean_text(text):
    """
    轻量级文本清洗：去除HTML，去除首尾多余符号，压缩空格
    """
    if not text:
        return ""

    if "<" in text and ">" in text:
        try:
            text = BeautifulSoup(text, "html.parser").get_text()
        except:
            pass

    text = text.replace("\n", " ").replace("\r", " ").replace("\t", " ")

    text = re.sub(r"\s+", " ", text).strip()

    return text

def extract_yield(reaction):
    best = -1.0
    for outcome in reaction.outcomes:
        for prod in outcome.products:
            for m in prod.measurements:
                if m.type in [2, 3]:
                    if m.percentage.HasField("value"):
                        y = m.percentage.value
                        if 0 <= y <= 105:
                            best = max(best, y)
    return min(best, 100.0)

def process_dataset(dataset, debug=False):
    results = []

    for reaction in dataset.reactions:
        reactants, products, solvents, catalysts, reagents = [], [], [], [], []

        for input_msg in reaction.inputs.values():
            for comp in input_msg.components:
                s = next((i.value for i in comp.identifiers if i.type == i.SMILES), "")
                if not s:
                    continue

                role = comp.reaction_role
                if role == 1:
                    reactants.append(s)
                elif role == 2:
                    solvents.append(s)
                elif role == 3:
                    catalysts.append(s)
                elif role == 4:
                    reagents.append(s)

        if reaction.outcomes:
            outcome = reaction.outcomes[0]
            desired = [
                next((i.value for i in p.identifiers if i.type == i.SMILES), "")
                for p in outcome.products
                if getattr(p, "is_desired_product", False)
            ]
            all_p = [
                next((i.value for i in p.identifiers if i.type == i.SMILES), "")
                for p in outcome.products
            ]
            products = [p for p in (desired if desired else all_p) if p]

        if reactants and products:

            proc_text = ""
            if reaction.notes.procedure_details:
                proc_text = clean_text(reaction.notes.procedure_details)

            workup_parts = []
            for w in reaction.workups:
                if w.details:
                    workup_parts.append(clean_text(w.details))
            workup_text = " ".join(workup_parts)

            combined_text = f"{proc_text} {workup_text}"
            if reaction.conditions.details:
                combined_text += " " + clean_text(reaction.conditions.details)

            temp = extractor.parse_temp(combined_text)
            time_val = extractor.parse_time(combined_text)

            if debug and (proc_text or workup_text):
                print(f"\n[DEBUG] ID: {reaction.reaction_id}")
                print(f"  Procedure: {proc_text[:80]}...")
                print(f"  Workup:    {workup_text[:80]}...")

            results.append(
                {
                    "reaction_id": reaction.reaction_id,
                    "reactants": reactants,
                    "products": products,
                    "yield": extract_yield(reaction),
                    "solvents": solvents,
                    "catalysts": catalysts,
                    "reagents": reagents,
                    "temperature": temp,
                    "time": time_val,
                    "procedure_text": proc_text,
                    "workup_text": workup_text,
                }
            )
    return results

if __name__ == "__main__":
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    files = glob.glob(os.path.join(ORD_DATA_DIR, "**/*.pb.gz"), recursive=True)
    if MAX_FILES:
        files = files[:MAX_FILES]

    print(f"🚀 开始处理 {len(files)} 个文件...")

    stats = {"total": 0, "has_yield": 0, "has_text": 0}

    headers = [
        "Reaction ID",
        "Yield",
        "Temperature(C)",
        "Time(h)",
        "Reactants",
        "Products",
        "Solvents",
        "Catalysts",
        "Reagents",
        "Procedure_Text",
        "Workup_Text",
    ]

    with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(headers)

    pbar = tqdm(files)
    debug_counter = 0

    for fpath in pbar:
        try:
            ds = load_pb_gz(fpath)
            if not ds:
                continue

            is_debug = debug_counter < 3
            data = process_dataset(ds, debug=is_debug)
            if data:
                debug_counter += 1

            if data:
                with open(CSV_PATH, "a", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    for item in data:
                        stats["total"] += 1
                        if item["yield"] >= 0:
                            stats["has_yield"] += 1
                        if item["procedure_text"] or item["workup_text"]:
                            stats["has_text"] += 1

                        writer.writerow(
                            [
                                item["reaction_id"],
                                item["yield"],
                                (
                                    f"{item['temperature']:.2f}"
                                    if item["temperature"] is not None
                                    else "None"
                                ),
                                (
                                    f"{item['time']:.2f}"
                                    if item["time"] is not None
                                    else "None"
                                ),
                                " ; ".join(item["reactants"]),
                                " ; ".join(item["products"]),
                                " ; ".join(item["solvents"]),
                                " ; ".join(item["catalysts"]),
                                " ; ".join(item["reagents"]),
                                item["procedure_text"],
                                item["workup_text"],
                            ]
                        )

            pbar.set_postfix(
                total=stats["total"],
                text_rate=f"{stats['has_text']/max(1,stats['total']):.1%}",
            )

        except Exception as e:

            pass

    print(f"\n✅ 处理完成！")
    print(f"   总数据量: {stats['total']}")
    print(f"   含产率数据: {stats['has_yield']}")
    print(f"   含文本数据: {stats['has_text']}")
    print(f"   结果已保存至: {CSV_PATH}")
