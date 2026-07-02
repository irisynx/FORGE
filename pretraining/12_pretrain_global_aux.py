import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch_geometric.data import Dataset, Batch, Data
from torch_geometric.nn import (
    GATv2Conv,
    LayerNorm,
    GlobalAttention,
)
from tqdm import tqdm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

DATA_ROOT = "./data/processed_ready_20260131"

ATOM_FEATURE_DIMS = [120, 15, 15, 10, 10, 100, 10, 10]

def inspect_data_sample(dataset):
    print("\n" + "=" * 50)
    print("🔍 DATA INSPECTION (DEBUG MODE)")
    print("=" * 50)
    if len(dataset) == 0:
        print("❌ Error: Dataset is empty!")
        return False
    try:
        data = dataset[0]
        print(f"File loaded successfully.")
        print(f"x shape: {data.x.shape}")

        x_max = data.x.max(dim=0)[0]
        print(f"Max feature indices in sample 0: {x_max.tolist()}")
        for i, (max_val, limit) in enumerate(zip(x_max, ATOM_FEATURE_DIMS)):
            if max_val >= limit:
                print(f"⚠️ WARNING: Feature {i} has value {max_val} >= limit {limit}!")

        print("=" * 50 + "\n")
        return True
    except Exception as e:
        print(f"❌ Error reading sample: {e}")
        return False

class MolecularDataset(Dataset):
    def __init__(self, root, transform=None, pre_transform=None):
        self.root_abs = os.path.abspath(root)
        if os.path.isdir(os.path.join(self.root_abs, "processed")):
            self.data_dir = os.path.join(self.root_abs, "processed")
        else:
            self.data_dir = self.root_abs
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir, exist_ok=True)
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
            return data
        except Exception as e:
            return None

def custom_collate(data_list):
    valid_data = []
    for data in data_list:
        if data is None:
            continue
        required = ["x", "edge_index", "y_fukui", "y_charge", "homo", "lumo", "gap"]
        if not all(k in data for k in required):
            continue

        n_atoms = data.x.shape[0]
        if data.y_charge.dim() > 1:
            data.y_charge = data.y_charge.squeeze()
        if data.y_charge.shape[0] != n_atoms:
            continue

        if not isinstance(data.homo, torch.Tensor):
            data.homo = torch.tensor(data.homo)
        if not isinstance(data.lumo, torch.Tensor):
            data.lumo = torch.tensor(data.lumo)
        if not isinstance(data.gap, torch.Tensor):
            data.gap = torch.tensor(data.gap)

        if data.homo.numel() == 1:
            data.homo = data.homo.reshape(-1)
        if data.lumo.numel() == 1:
            data.lumo = data.lumo.reshape(-1)
        if data.gap.numel() == 1:
            data.gap = data.gap.reshape(-1)

        valid_data.append(data)

    if len(valid_data) == 0:
        return None
    try:
        return Batch.from_data_list(valid_data)
    except:
        return None

class AtomFeatureEncoder(nn.Module):
    def __init__(self, hidden_dim, feature_dims):
        super().__init__()
        self.embeddings = nn.ModuleList()
        self.feature_dims = feature_dims
        for dim in feature_dims:
            self.embeddings.append(nn.Embedding(dim, hidden_dim))

    def forward(self, x):
        out = 0
        for i in range(len(self.embeddings)):
            feat_col = x[:, i]
            max_idx = self.feature_dims[i] - 1
            feat_col_clamped = feat_col.clamp(0, max_idx)

            if i == 0:
                out = self.embeddings[i](feat_col_clamped)
            else:
                out = out + self.embeddings[i](feat_col_clamped)
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

        self.dropout = nn.Dropout(0.25)

    def forward(self, x, edge_index, edge_attr):
        x = self.atom_encoder(x)
        edge_attr = edge_attr.clamp(0, 9)
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
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 2),
        )
        self.charge_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

        self.att_pool = GlobalAttention(
            gate_nn=nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, 1),
            )
        )

        graph_input_dim = hidden_dim

        self.homo_head = nn.Sequential(
            nn.Linear(graph_input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 1),
        )
        self.lumo_head = nn.Sequential(
            nn.Linear(graph_input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, data):
        x, edge_index, edge_attr, batch = (
            data.x,
            data.edge_index,
            data.edge_attr,
            data.batch,
        )
        node_emb = self.encoder(x, edge_index, edge_attr)

        pred_fukui = self.fukui_head(node_emb)
        pred_charge = self.charge_head(node_emb)

        graph_emb = self.att_pool(node_emb, batch)

        pred_homo = self.homo_head(graph_emb).squeeze(-1)
        pred_lumo = self.lumo_head(graph_emb).squeeze(-1)

        pred_gap = pred_lumo - pred_homo

        return {
            "fukui": pred_fukui,
            "charge": pred_charge,
            "homo": pred_homo,
            "lumo": pred_lumo,
            "gap": pred_gap,
        }

class ElectronicLoss(nn.Module):
    def __init__(self, w_fukui=30.0, w_charge=30.0, w_global=20.0):
        super().__init__()
        self.w_fukui = w_fukui
        self.w_charge = w_charge
        self.w_global = w_global
        self.mse = nn.MSELoss()
        self.mae = nn.L1Loss()

    def update_weights(self, w_fukui, w_charge, w_global):
        self.w_fukui = w_fukui
        self.w_charge = w_charge
        self.w_global = w_global

    def forward(self, preds, data):
        loss_fukui = self.mse(preds["fukui"], data.y_fukui)

        target_charge = data.y_charge
        if target_charge.dim() == 1:
            target_charge = target_charge.unsqueeze(-1)
        loss_charge = self.mse(preds["charge"], target_charge)

        loss_homo = self.mse(preds["homo"], data.homo)
        loss_lumo = self.mse(preds["lumo"], data.lumo)
        loss_gap = self.mse(preds["gap"], data.gap)

        total_loss = (
            self.w_fukui * loss_fukui
            + self.w_charge * loss_charge
            + self.w_global * (loss_homo + loss_lumo + loss_gap)
        )

        return total_loss, {
            "loss_fukui": loss_fukui.item(),
            "loss_charge": loss_charge.item(),
            "loss_homo": loss_homo.item(),
            "loss_lumo": loss_lumo.item(),
            "loss_gap": loss_gap.item(),
            "mae_gap": self.mae(preds["gap"], data.gap).item(),
        }

class ModelEMA:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.model = model
        self.shadow = {}
        self.backup = {}
        self.register()

    def register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                new_average = (
                    1.0 - self.decay
                ) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data
                param.data = self.shadow[name]

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]
        self.backup = {}

def plot_training_history(history_df, save_dir="results"):
    if len(history_df) < 1:
        return
    epochs = history_df["epoch"]
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    axes[0].plot(epochs, history_df["train_loss"], label="Train Loss", linewidth=2)
    axes[0].plot(
        epochs, history_df["val_loss"], label="Val Loss", linewidth=2, linestyle="--"
    )
    axes[0].set_title("Total Loss Curve")
    axes[0].set_yscale("log")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(
        epochs, history_df["train_mae_gap"], label="Train Gap MAE", color="orange"
    )
    axes[1].plot(
        epochs,
        history_df["val_mae_gap"],
        label="Val Gap MAE",
        color="red",
        linestyle="--",
    )
    axes[1].set_title("Gap MAE (Key Metric)")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    if "val_loss_homo" in history_df.columns:
        axes[2].plot(epochs, history_df["val_loss_homo"], label="HOMO", alpha=0.7)
        axes[2].plot(epochs, history_df["val_loss_lumo"], label="LUMO", alpha=0.7)
        axes[2].plot(epochs, history_df["val_loss_gap"], label="Gap", alpha=0.7)
    axes[2].set_title("Validation Sub-Task Losses")
    axes[2].set_yscale("log")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "training_curves.png"), dpi=150)
    plt.close()

def evaluate_and_plot_globals(model, loader, device, epoch, save_dir="results"):
    """
    绘制真实值 vs 预测值的散点图
    """
    model.eval()
    trues = {"homo": [], "lumo": [], "gap": []}
    preds = {"homo": [], "lumo": [], "gap": []}

    with torch.no_grad():
        for batch in loader:
            if batch is None:
                continue
            batch = batch.to(device)
            out = model(batch)

            trues["homo"].append(batch.homo.view(-1).cpu().numpy())
            preds["homo"].append(out["homo"].view(-1).cpu().numpy())
            trues["lumo"].append(batch.lumo.view(-1).cpu().numpy())
            preds["lumo"].append(out["lumo"].view(-1).cpu().numpy())
            trues["gap"].append(batch.gap.view(-1).cpu().numpy())
            preds["gap"].append(out["gap"].view(-1).cpu().numpy())

    if len(trues["homo"]) == 0:
        return

    for k in trues:
        trues[k] = np.concatenate(trues[k])
        preds[k] = np.concatenate(preds[k])

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    titles = ["HOMO", "LUMO", "Gap"]
    keys = ["homo", "lumo", "gap"]

    for i, ax in enumerate(axes):
        key = keys[i]
        y_true = trues[key]
        y_pred = preds[key]

        mae = np.mean(np.abs(y_true - y_pred))

        if len(y_true) > 1:
            r2 = np.corrcoef(y_true, y_pred)[0, 1] ** 2
        else:
            r2 = 0

        ax.scatter(y_true, y_pred, alpha=0.4, s=15, edgecolors="none", c="blue")

        lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
        ax.plot(lims, lims, "r--", alpha=0.8, linewidth=2)

        ax.set_title(f"{titles[i]} | MAE: {mae:.4f} | R2: {r2:.3f}")
        ax.set_xlabel("DFT Truth")
        ax.set_ylabel("GNN Pred")
        ax.grid(True, alpha=0.3)

    plt.suptitle(f"Epoch {epoch} Predictions", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"global_preds_epoch_{epoch}.png"), dpi=150)
    plt.close()

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    BATCH_SIZE = 24
    EPOCHS = 200
    NUM_WORKERS = 4 if os.name != "nt" else 0

    os.makedirs("results", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)

    print(f"Loading Dataset from {DATA_ROOT}...")
    dataset = MolecularDataset(root=DATA_ROOT)

    if not inspect_data_sample(dataset):
        print("❌ Data inspection failed. Aborting.")
        return

    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        collate_fn=custom_collate,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        collate_fn=custom_collate,
        pin_memory=True,
    )

    encoder = MoleculeEncoder(hidden_dim=512, num_layers=6, num_heads=8)
    model = ElectronicPretrainingModel(encoder, hidden_dim=512).to(device)

    ema = ModelEMA(model, decay=0.995)

    optimizer = optim.AdamW(model.parameters(), lr=5e-4, weight_decay=5e-4)

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS, eta_min=1e-6
    )

    loss_fn = ElectronicLoss(w_fukui=30.0, w_charge=30.0, w_global=0.0)

    accumulation_steps = 4
    history = []

    print(f"Start Training on {device}...")

    for epoch in range(1, EPOCHS + 1):

        if epoch <= 20:

            current_w_global = 0.0
            phase_name = "Phase 1 (Node Only)"
        else:

            current_w_global = 20.0
            phase_name = "Phase 2 (Global + Node)"

        loss_fn.update_weights(w_fukui=30.0, w_charge=30.0, w_global=current_w_global)

        print(f"--> Epoch {epoch}: Entering {phase_name}")

        model.train()
        total_loss_accum = 0
        valid_batch_count = 0
        logs = {
            "loss_fukui": 0,
            "loss_charge": 0,
            "loss_homo": 0,
            "loss_lumo": 0,
            "loss_gap": 0,
            "mae_gap": 0,
        }
        optimizer.zero_grad()

        pbar = tqdm(train_loader, desc=f"Ep {epoch}", leave=False)
        for i, batch in enumerate(pbar):
            if batch is None:
                continue

            batch = batch.to(device)
            out = model(batch)
            loss, loss_dict = loss_fn(out, batch)

            loss = loss / accumulation_steps
            loss.backward()

            current_loss = loss.item() * accumulation_steps
            total_loss_accum += current_loss
            valid_batch_count += 1

            for k, v in loss_dict.items():
                logs[k] += v

            if (i + 1) % accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                ema.update()
                optimizer.zero_grad()
                pbar.set_postfix(
                    {
                        "loss": f"{current_loss:.4f}",
                        "gap": f"{loss_dict['mae_gap']:.3f}",
                    }
                )

        if valid_batch_count > 0:
            avg_train_loss = total_loss_accum / valid_batch_count
            avg_train_logs = {k: v / valid_batch_count for k, v in logs.items()}
        else:
            avg_train_loss = 0
            avg_train_logs = logs

        ema.apply_shadow()
        model.eval()
        val_loss_accum = 0
        val_batch_count = 0
        val_logs = {
            "loss_fukui": 0,
            "loss_charge": 0,
            "loss_homo": 0,
            "loss_lumo": 0,
            "loss_gap": 0,
            "mae_gap": 0,
        }

        with torch.no_grad():
            for batch in val_loader:
                if batch is None:
                    continue
                batch = batch.to(device)
                out = model(batch)
                loss, loss_dict = loss_fn(out, batch)
                val_loss_accum += loss.item()
                val_batch_count += 1
                for k, v in loss_dict.items():
                    val_logs[k] += v

        ema.restore()

        if val_batch_count > 0:
            avg_val_loss = val_loss_accum / val_batch_count
            avg_val_logs = {k: v / val_batch_count for k, v in val_logs.items()}
        else:
            avg_val_loss = 0
            avg_val_logs = val_logs

        scheduler.step()

        print(
            f"Ep {epoch} | lr: {optimizer.param_groups[0]['lr']:.2e} | Train: {avg_train_loss:.4f} | Val: {avg_val_loss:.4f} | Val Gap MAE: {avg_val_logs['mae_gap']:.4f}"
        )

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
        plot_training_history(pd.DataFrame(history), save_dir="results")

        if epoch % 5 == 0:

            ema.apply_shadow()
            evaluate_and_plot_globals(model, val_loader, device, epoch)
            ema.restore()

        if epoch % 10 == 0:

            ema.apply_shadow()
            torch.save(model.state_dict(), f"checkpoints/model_epoch_{epoch}.pth")
            ema.restore()

if __name__ == "__main__":
    main()
