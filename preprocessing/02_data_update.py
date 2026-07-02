import os
import torch
from tqdm import tqdm

DATA_DIR = "./data/processed_data"

def get_bond_type_from_wbo(wbo_val):
    if wbo_val < 1.3:
        return 0
    elif 1.3 <= wbo_val < 1.8:
        return 3
    elif 1.8 <= wbo_val < 2.5:
        return 1
    else:
        return 2

def process_file(file_path):
    try:

        data = torch.load(file_path, weights_only=False)

        edge_index = data.edge_index
        y_wbo = data.y_wbo

        num_edges = edge_index.size(1)
        edge_attrs = []

        for k in range(num_edges):
            i = edge_index[0, k].item()
            j = edge_index[1, k].item()

            wbo_val = y_wbo[i, j].item()

            b_type = get_bond_type_from_wbo(wbo_val)
            edge_attrs.append(b_type)

        data.edge_attr = torch.tensor(edge_attrs, dtype=torch.long)

        torch.save(data, file_path)
        return True

    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return False

def main():

    files = [f for f in os.listdir(DATA_DIR) if f.endswith(".pt")]
    print(f"Found {len(files)} files. Starting update...")

    success_count = 0
    for f in tqdm(files):
        path = os.path.join(DATA_DIR, f)
        if process_file(path):
            success_count += 1

    print(f"Done! Updated {success_count}/{len(files)} files.")

if __name__ == "__main__":
    main()
