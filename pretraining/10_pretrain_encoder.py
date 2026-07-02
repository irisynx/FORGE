import os
import os.path as osp
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch_geometric.data import Dataset, Batch
from torch_geometric.nn import GATv2Conv, LayerNorm
from tqdm import tqdm
import pandas as pd
import numpy as np
from forge_pretrain_utils import (
    plot_training_curves,
    evaluate_and_plot_electronic,
    visualize_molecule_fukui,
)

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
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            return None
        return data

def custom_collate(data_list):
    valid_data = []
    wbo_list = []

    for i, data in enumerate(data_list):
        if data is None:
            continue
        if not hasattr(data, "y_wbo") or not hasattr(data, "y_charge"):
            continue

        n_atoms = data.x.shape[0]
        if data.y_wbo.shape != (n_atoms, n_atoms):
            continue
        if data.y_charge.numel() != n_atoms:
            continue

        wbo_list.append(data.y_wbo)
        del data.y_wbo
        valid_data.append(data)

    if len(valid_data) == 0:
        return None, None

    batch = Batch.from_data_list(valid_data)
    y_wbo_big = torch.block_diag(*wbo_list)

    return batch, y_wbo_big

class MoleculeEncoder(nn.Module):
    def __init__(
        self, hidden_dim=256, num_layers=6, num_heads=4, edge_embedding_dim=64
    ):
        super().__init__()
        self.atom_embedding = nn.Embedding(100, hidden_dim)
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
        if x.dim() > 1:
            x = x.squeeze()
        x = self.atom_embedding(x)
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

        self.pair_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2 + edge_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )

        self.pair_edge_emb = nn.Embedding(10, edge_dim)

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        node_emb = self.encoder(x, edge_index, edge_attr)

        pred_fukui = self.fukui_head(node_emb)
        pred_charge = self.charge_head(node_emb)

        return pred_fukui, pred_charge, node_emb

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        node_emb = self.encoder(x, edge_index, edge_attr)

        pred_fukui = self.fukui_head(node_emb)
        pred_charge = self.charge_head(node_emb)

        return pred_fukui, pred_charge, node_emb

class ElectronicLoss(nn.Module):
    def __init__(self, w_fukui=1.0, w_charge=0.2, w_wbo=0.5, edge_dim=64):
        super().__init__()
        self.w_fukui = w_fukui
        self.w_charge = w_charge
        self.w_wbo = w_wbo
        self.mse = nn.MSELoss()

    def forward(self, model, data, target_wbo):
        pred_fukui, pred_charge, node_emb = model(data)

        loss_fukui = self.mse(pred_fukui, data.y_fukui)
        loss_charge = self.mse(
            pred_charge,
            data.y_charge.unsqueeze(-1) if data.y_charge.dim() == 1 else data.y_charge,
        )

        batch = data.batch
        N, D = node_emb.shape
        mask = batch.unsqueeze(0) == batch.unsqueeze(1)

        h_i = node_emb.unsqueeze(1).expand(N, N, D)
        h_j = node_emb.unsqueeze(0).expand(N, N, D)

        edge_map_idx = torch.zeros((N, N), dtype=torch.long, device=node_emb.device)

        src, dst = data.edge_index

        attr = data.edge_attr
        if attr.dim() > 1:
            attr = attr.squeeze()

        edge_map_idx[src, dst] = attr.long() + 1

        h_edge = model.pair_edge_emb(edge_map_idx)

        h_pair_full = torch.cat([h_i, h_j, h_edge], dim=-1)

        h_pair_masked = h_pair_full[mask]
        pred_wbo_masked = model.pair_mlp(h_pair_masked).squeeze()
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

def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    BATCH_SIZE = 16
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
        optimizer,
        mode="min",
        factor=0.5,
        patience=5,
        min_lr=1e-6,
        verbose=True,
    )

    loss_fn = ElectronicLoss(w_fukui=1.0, w_charge=0.2, w_wbo=0.5)
    accumulation_steps = 64
    history = []

    print(f"Start Training on {device}...")

    for epoch in range(1, EPOCHS + 1):

        model.train()
        total_loss_accum = 0
        logs = {"loss_fukui": 0, "loss_charge": 0, "loss_wbo": 0}
        optimizer.zero_grad()

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}", leave=False)

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

            if (i + 1) % accumulation_steps == 0 or (i + 1) == len(train_loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
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

        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch:03d} | LR: {current_lr:.2e} | Train: {avg_train_loss:.4f} | Val: {avg_val_loss:.4f}"
        )
        print(
            f"   Train Details: Fukui={avg_train_logs['loss_fukui']:.4f}, Charge={avg_train_logs['loss_charge']:.4f}, WBO={avg_train_logs['loss_wbo']:.4f}"
        )

        epoch_data = {
            "epoch": epoch,
            "lr": current_lr,
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
                evaluate_and_plot_electronic(model, val_loader, device, epoch)
                visualize_molecule_fukui(model, val_dataset, device, epoch)
            except Exception as e:
                print(f"Visualization failed: {e}")

        if epoch % 10 == 0:
            torch.save(model.state_dict(), f"checkpoints/model_epoch_{epoch}.pth")

    print("Training Complete!")

if __name__ == "__main__":
    main()
