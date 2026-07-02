import matplotlib.pyplot as plt
import numpy as np
import torch
import pandas as pd
import os

def plot_training_curves(history_file, save_dir='results'):
    """
    绘制训练过程中的 Loss 曲线和 LR 变化
    """
    if not os.path.exists(history_file):
        print(f"Warning: History file {history_file} not found.")
        return

    df = pd.read_csv(history_file)

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    axes[0, 0].plot(df['epoch'], df['train_loss'], label='Train', color='blue')
    axes[0, 0].plot(df['epoch'], df['val_loss'], label='Val', color='orange')
    axes[0, 0].set_title('Total Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, linestyle='--', alpha=0.7)

    if 'train_loss_fukui' in df.columns:
        axes[0, 1].plot(df['epoch'], df['train_loss_fukui'], label='Train', color='blue')
        axes[0, 1].plot(df['epoch'], df['val_loss_fukui'], label='Val', color='orange')
        axes[0, 1].set_title('Fukui Loss (MSE)')
        axes[0, 1].legend()

    if 'train_loss_charge' in df.columns and 'train_loss_wbo' in df.columns:
        axes[1, 0].plot(df['epoch'], df['train_loss_charge'], label='Charge', color='green')
        axes[1, 0].plot(df['epoch'], df['train_loss_wbo'], label='WBO', color='purple')
        axes[1, 0].set_title('Auxiliary Losses (Train)')
        axes[1, 0].legend()

    axes[1, 1].plot(df['epoch'], df['lr'], color='red')
    axes[1, 1].set_title('Learning Rate')
    axes[1, 1].set_yscale('log')

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_curves.png'), dpi=150)
    plt.close()

def evaluate_and_plot_electronic(model, loader, device, epoch, save_dir='results'):
    """
    绘制电子性质的 Parity Plots (Fukui, Charge, WBO)
    已修复: 支持带 Edge Embedding 的 WBO 预测
    """
    model.eval()

    preds_fukui, targets_fukui = [], []
    preds_charge, targets_charge = [], []
    preds_wbo, targets_wbo = [], []

    max_batches = 20

    with torch.no_grad():
        for i, (batch, y_wbo_gt) in enumerate(loader):
            if i >= max_batches: break
            if batch is None: continue

            batch = batch.to(device)
            y_wbo_gt = y_wbo_gt.to(device)

            pred_fukui, pred_charge, node_emb = model(batch)

            preds_fukui.append(pred_fukui.cpu().numpy())
            targets_fukui.append(batch.y_fukui.cpu().numpy())
            preds_charge.append(pred_charge.cpu().numpy())
            targets_charge.append(batch.y_charge.cpu().numpy())

            batch_idx = batch.batch
            mask = (batch_idx.unsqueeze(0) == batch_idx.unsqueeze(1))

            N, D = node_emb.shape
            h_i = node_emb.unsqueeze(1).expand(N, N, D)
            h_j = node_emb.unsqueeze(0).expand(N, N, D)

            edge_map_idx = torch.zeros((N, N), dtype=torch.long, device=device)

            src, dst = batch.edge_index
            attr = batch.edge_attr
            if attr.dim() > 1: attr = attr.squeeze()

            edge_map_idx[src, dst] = attr.long() + 1

            if hasattr(model, 'pair_edge_emb'):
                h_edge = model.pair_edge_emb(edge_map_idx)
                h_pair_full = torch.cat([h_i, h_j, h_edge], dim=-1)
            else:

                h_pair_full = torch.cat([h_i, h_j], dim=-1)

            h_pair_masked = h_pair_full[mask]
            pred_wbo_val = model.pair_mlp(h_pair_masked).squeeze()
            target_wbo_val = y_wbo_gt[mask]

            is_bond = target_wbo_val > 0.1
            bond_indices = torch.where(is_bond)[0]
            non_bond_indices = torch.where(~is_bond)[0]

            if len(bond_indices) > 500:
                bond_indices = bond_indices[torch.randperm(len(bond_indices))[:500]]
            if len(non_bond_indices) > 500:
                non_bond_indices = non_bond_indices[torch.randperm(len(non_bond_indices))[:500]]

            indices = torch.cat([bond_indices, non_bond_indices])

            preds_wbo.append(pred_wbo_val[indices].cpu().numpy())
            targets_wbo.append(target_wbo_val[indices].cpu().numpy())

    preds_fukui = np.concatenate(preds_fukui, axis=0)
    targets_fukui = np.concatenate(targets_fukui, axis=0)
    preds_charge = np.concatenate(preds_charge, axis=0).flatten()
    targets_charge = np.concatenate(targets_charge, axis=0).flatten()
    preds_wbo = np.concatenate(preds_wbo, axis=0)
    targets_wbo = np.concatenate(targets_wbo, axis=0)

    fig, axes = plt.subplots(2, 2, figsize=(12, 12))

    axes[0, 0].scatter(targets_fukui[:, 0], preds_fukui[:, 0], alpha=0.3, s=5, c='crimson')
    axes[0, 0].plot([0, 1], [0, 1], 'k--', lw=1)
    axes[0, 0].set_title(f'Fukui f(+) (MAE: {np.mean(np.abs(preds_fukui[:,0]-targets_fukui[:,0])):.4f})')

    axes[0, 1].scatter(targets_fukui[:, 1], preds_fukui[:, 1], alpha=0.3, s=5, c='royalblue')
    axes[0, 1].plot([0, 1], [0, 1], 'k--', lw=1)
    axes[0, 1].set_title(f'Fukui f(-) (MAE: {np.mean(np.abs(preds_fukui[:,1]-targets_fukui[:,1])):.4f})')

    axes[1, 0].scatter(targets_charge, preds_charge, alpha=0.3, s=5, c='green')
    axes[1, 0].plot([targets_charge.min(), targets_charge.max()], [targets_charge.min(), targets_charge.max()], 'k--', lw=1)
    axes[1, 0].set_title(f'Charge (MAE: {np.mean(np.abs(preds_charge-targets_charge)):.4f})')

    axes[1, 1].scatter(targets_wbo, preds_wbo, alpha=0.3, s=5, c='purple')
    axes[1, 1].plot([0, 3], [0, 3], 'k--', lw=1)
    axes[1, 1].set_title(f'WBO (MAE: {np.mean(np.abs(preds_wbo-targets_wbo)):.4f})')

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'epoch_{epoch}_electronic.png'), dpi=150)
    plt.close()

def visualize_molecule_fukui(model, dataset, device, epoch, save_dir='results'):
    """
    可视化单个分子的 Fukui 预测结果
    """
    model.eval()
    data = dataset[0]
    data.batch = torch.zeros(data.x.size(0), dtype=torch.long)
    data = data.to(device)

    with torch.no_grad():
        pred_fukui, _, _ = model(data)
        pred_fukui = pred_fukui.cpu().numpy()
        true_fukui = data.y_fukui.cpu().numpy()

    atoms = range(len(true_fukui))
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))

    width = 0.35
    axes[0].bar(np.array(atoms) - width/2, true_fukui[:, 0], width, label='True', color='lightgray')
    axes[0].bar(np.array(atoms) + width/2, pred_fukui[:, 0], width, label='Pred', color='crimson', alpha=0.7)
    axes[0].set_title('Fukui f(+)')
    axes[0].legend()

    axes[1].bar(np.array(atoms) - width/2, true_fukui[:, 1], width, label='True', color='lightgray')
    axes[1].bar(np.array(atoms) + width/2, pred_fukui[:, 1], width, label='Pred', color='royalblue', alpha=0.7)
    axes[1].set_title('Fukui f(-)')

    plt.savefig(os.path.join(save_dir, f'epoch_{epoch}_mol_viz.png'), dpi=150)
    plt.close()
