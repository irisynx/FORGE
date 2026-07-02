import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch_geometric.data import Dataset, Batch, Data
from torch_geometric.nn import GATv2Conv, LayerNorm
from tqdm import tqdm
import pandas as pd
import numpy as np
from rdkit import Chem
from forge_pretrain_utils import (
    plot_training_curves,
    evaluate_and_plot_electronic,
    visualize_molecule_fukui,
)

ATOM_FEATURE_DIMS = [100, 10, 10, 10, 10, 10, 2, 2]

class AtomFeatureEncoder(nn.Module):
    def __init__(self, hidden_dim, feature_dims):
        super().__init__()
        self.embeddings = nn.ModuleList()
        for dim in feature_dims:
            self.embeddings.append(nn.Embedding(dim, hidden_dim))

    def forward(self, x):

        out = self.embeddings[0](x[:, 0])
        for i in range(1, len(self.embeddings)):
            out = out + self.embeddings[i](x[:, i])
        return out

class MoleculeEncoder(nn.Module):
    def __init__(
        self, hidden_dim=256, num_layers=6, num_heads=4, edge_embedding_dim=64
    ):
        super().__init__()
        self.atom_encoder = AtomFeatureEncoder(hidden_dim, ATOM_FEATURE_DIMS)
        self.edge_embedding = nn.Embedding(10, edge_embedding_dim)
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        for _ in range(num_layers):
            self.convs.append(
                GATv2Conv(
                    hidden_dim,
                    hidden_dim // num_heads,
                    heads=num_heads,
                    concat=True,
                    edge_dim=edge_embedding_dim,
                )
            )
            self.norms.append(LayerNorm(hidden_dim))

        self.dropout = nn.Dropout(0.3)

    def forward(self, x, edge_index, edge_attr):
        x = self.atom_encoder(x)
        edge_emb = self.edge_embedding(edge_attr)

        for conv, norm in zip(self.convs, self.norms):
            x_in = x
            x = conv(x, edge_index, edge_attr=edge_emb)
            x = norm(x)
            x = F.gelu(x)
            x = self.dropout(x)
            x = x + x_in
        return x

class ElectronicPretrainingModel(nn.Module):
    def __init__(self, encoder, hidden_dim=256):
        super().__init__()
        self.encoder = encoder

        self.fukui_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 2),
        )
        self.charge_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )

        edge_dim = 64
        self.pair_edge_emb = nn.Embedding(10, edge_dim)
        self.pair_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2 + edge_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        node_emb = self.encoder(x, edge_index, edge_attr)

        pred_fukui = self.fukui_head(node_emb)
        pred_charge = self.charge_head(node_emb)

        pred_wbo_masked, mask = self._predict_wbo_pairs(node_emb, data)

        return pred_fukui, pred_charge, pred_wbo_masked, mask, node_emb

    def _predict_wbo_pairs(self, node_emb, data):
        batch = data.batch
        N, D = node_emb.shape
        mask = batch.unsqueeze(0) == batch.unsqueeze(1)

        r_idx, c_idx = torch.where(mask)

        edge_map_idx = torch.zeros((N, N), dtype=torch.long, device=node_emb.device)
        src, dst = data.edge_index
        attr = data.edge_attr
        if attr.dim() > 1:
            attr = attr.squeeze()
        edge_map_idx[src, dst] = attr.long() + 1

        h_i = node_emb[r_idx]
        h_j = node_emb[c_idx]
        e_type = edge_map_idx[r_idx, c_idx]
        h_edge = self.pair_edge_emb(e_type)

        h_pair = torch.cat([h_i, h_j, h_edge], dim=-1)
        pred_wbo = self.pair_mlp(h_pair).squeeze(-1)

        return pred_wbo, mask

class ElectronicLoss(nn.Module):
    def __init__(self, w_fukui=1.0, w_charge=0.2, w_wbo=0.5):
        super().__init__()
        self.w_fukui = w_fukui
        self.w_charge = w_charge
        self.w_wbo = w_wbo
        self.mse = nn.MSELoss()

    def forward(self, model, data, target_wbo):

        pred_fukui, pred_charge, pred_wbo_masked, mask, _ = model(data)

        loss_fukui = self.mse(pred_fukui, data.y_fukui)
        loss_charge = self.mse(
            pred_charge,
            data.y_charge.unsqueeze(-1) if data.y_charge.dim() == 1 else data.y_charge,
        )

        target_wbo_masked = target_wbo[mask]
        loss_wbo = self.mse(pred_wbo_masked, target_wbo_masked)

        total_loss = (
            self.w_fukui * loss_fukui
            + self.w_charge * loss_charge
            + self.w_wbo * loss_wbo
        )

        return total_loss, {
            "loss_fukui": loss_fukui.item(),
            "loss_charge": loss_charge.item(),
            "loss_wbo": loss_wbo.item(),
        }

class VisualizationAdapter:
    """
    这是一个包装器，用于欺骗旧的可视化函数。
    旧的 utils.py 可能期待 model(data) 返回 (fukui, charge, emb)。
    新的 model(data) 返回 (fukui, charge, wbo, mask, emb)。
    """

    def __init__(self, model):
        self.model = model

    def __getattr__(self, name):
        return getattr(self.model, name)

    def __call__(self, data):

        out = self.model(data)

        fukui, charge, _, _, emb = out

        return fukui, charge, emb

class MolecularDataset(Dataset):
    def __init__(self, root, transform=None, pre_transform=None):
        self.root_abs = os.path.abspath(root)
        processed_path = os.path.join(self.root_abs, "processed")
        if os.path.exists(processed_path) and any(
            f.endswith(".pt") for f in os.listdir(processed_path)
        ):
            self.data_dir = processed_path
        elif any(f.endswith(".pt") for f in os.listdir(self.root_abs)):
            self.data_dir = self.root_abs
        else:
            raise FileNotFoundError(f"在 {self.root_abs} 中未找到 .pt 文件")
        self.pt_files = sorted(
            [f for f in os.listdir(self.data_dir) if f.endswith(".pt")]
        )
        super().__init__(root, transform, pre_transform)

    def len(self):
        return len(self.pt_files)

    def get(self, idx):
        file_path = os.path.join(self.data_dir, self.pt_files[idx])
        try:
            data = torch.load(file_path, weights_only=False)

            if data.x.dim() == 1:
                print(
                    f"Warning: Data {self.pt_files[idx]} is 1D. Please regenerate dataset with 8 features."
                )
        except Exception as e:
            return None
        return data

def custom_collate(data_list):
    valid_data = []
    wbo_list = []

    for i, data in enumerate(data_list):
        if data is None:
            continue

        if "y_wbo" not in data or "y_charge" not in data:

            continue

        n_atoms = data.x.shape[0]

        if data.y_wbo.shape != (n_atoms, n_atoms):
            continue

        if data.y_charge.dim() > 1:
            data.y_charge = data.y_charge.squeeze()
        if data.y_charge.shape[0] != n_atoms:
            continue

        wbo_list.append(data.y_wbo)

        del data.y_wbo

        valid_data.append(data)

    if len(valid_data) == 0:
        return None, None

    try:
        batch = Batch.from_data_list(valid_data)
        y_wbo_big = torch.block_diag(*wbo_list)
        return batch, y_wbo_big
    except Exception as e:
        print(f"Error during batching: {e}")

        return None, None

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    BATCH_SIZE = 24
    EPOCHS = 100
    os.makedirs("results", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)

    print("Loading Dataset...")

    dataset = MolecularDataset(root="./data/processed_data")

    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        collate_fn=custom_collate,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=custom_collate
    )

    encoder = MoleculeEncoder(hidden_dim=512, num_layers=6, num_heads=8)
    model = ElectronicPretrainingModel(encoder, hidden_dim=512).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-3)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, min_lr=1e-6
    )
    loss_fn = ElectronicLoss(w_fukui=1.0, w_charge=0.2, w_wbo=0.5)

    accumulation_steps = 4
    history = []

    print(f"Start Enhanced Training on {device}...")

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss_accum = 0
        logs = {"loss_fukui": 0, "loss_charge": 0, "loss_wbo": 0}
        optimizer.zero_grad()

        pbar = tqdm(train_loader, desc=f"Ep {epoch}", leave=False)
        for i, (batch, y_wbo) in enumerate(pbar):
            if batch is None:
                continue
            batch = batch.to(device)
            y_wbo = y_wbo.to(device)

            loss, loss_dict = loss_fn(model, batch, y_wbo)
            loss = loss / accumulation_steps
            loss.backward()

            current_loss = loss.item() * accumulation_steps
            total_loss_accum += current_loss
            for k, v in loss_dict.items():
                logs[k] += v

            if (i + 1) % accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
                pbar.set_postfix({"loss": f"{current_loss:.4f}"})

        avg_train_loss = total_loss_accum / len(train_loader)
        avg_train_logs = {k: v / len(train_loader) for k, v in logs.items()}

        model.eval()
        val_loss_accum = 0
        val_logs = {"loss_fukui": 0, "loss_charge": 0, "loss_wbo": 0}
        with torch.no_grad():
            for batch, y_wbo in val_loader:
                if batch is None:
                    continue
                batch = batch.to(device)
                y_wbo = y_wbo.to(device)
                loss, loss_dict = loss_fn(model, batch, y_wbo)
                val_loss_accum += loss.item()
                for k, v in loss_dict.items():
                    val_logs[k] += v

        avg_val_loss = val_loss_accum / len(val_loader)
        avg_val_logs = {k: v / len(val_loader) for k, v in val_logs.items()}

        scheduler.step(avg_val_loss)

        print(f"Ep {epoch} | Train: {avg_train_loss:.4f} | Val: {avg_val_loss:.4f}")

        epoch_data = {
            "epoch": epoch,
            "lr": optimizer.param_groups[0]["lr"],
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss,
            **{f"train_{k}": v for k, v in avg_train_logs.items()},
            **{f"val_{k}": v for k, v in avg_val_logs.items()},
        }
        history.append(epoch_data)
        pd.DataFrame(history).to_csv("results/training_history.csv", index=False)

        if epoch % 5 == 0:
            print(f"    Generating visualizations for epoch {epoch}...")
            try:

                plot_training_curves("results/training_history.csv")

                viz_model = VisualizationAdapter(model)

                evaluate_and_plot_electronic(viz_model, val_loader, device, epoch)
                visualize_molecule_fukui(viz_model, val_dataset, device, epoch)

            except Exception as e:
                print(f"Visualization failed: {e}")
                import traceback

                traceback.print_exc()

        if epoch % 10 == 0:
            torch.save(model.state_dict(), f"checkpoints/model_epoch_{epoch}.pth")

if __name__ == "__main__":
    main()
