"""Train the MLP-based multi-view image fusion model.

The model (``MultiViewImage``) softmax-pools the 6 per-view image embeddings into a single
768-d vector via a learnable per-view weight, then refines it with an MLP + LayerNorm.
Training is self-supervised over the views (contrastive + max-view-similarity loss); it only
needs the per-model image embeddings from the mega CSV produced by ``generate_embeddings.py``.

Example:
    python train_multiview_image_mlp.py \
        --models_csv out/models_data.csv \
        --model_output_path models/multiview_image_mlp.pt
"""

import argparse
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import random_split
from tqdm import tqdm

# Make sibling modules (db_common, generate_embeddings) importable from the scripts/ root.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import db_common  # noqa: E402


class MultiViewImage(nn.Module):
    def __init__(self, embed_dim=768):
        super(MultiViewImage, self).__init__()

        # Learnable scalar weight per view (via MLP)
        self.view_weight_net = nn.Sequential(
            nn.Linear(embed_dim, 1)  # Score each view independently
        )

        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, 2 * embed_dim),
            nn.ReLU(),
            nn.Linear(2 * embed_dim, embed_dim)
        )

        self.layer_norm = nn.LayerNorm(embed_dim)

    def forward(self, embeddings):
        """embeddings: Tensor of shape (batch_size, 6, embed_dim)"""
        weights = self.view_weight_net(embeddings).squeeze(-1)  # (batch_size, 6)
        weights = F.softmax(weights, dim=1)  # Normalize across views

        pooled = torch.sum(embeddings * weights.unsqueeze(-1), dim=1)  # (batch_size, embed_dim)

        output = self.mlp(pooled)
        output = self.layer_norm(output)
        return output  # (batch_size, embed_dim)


def train_model(model, train_loader, val_loader, optimizer, num_epochs=100,
                temperature=0.05, alpha=0.5, device="cpu", loss_plot=None):
    model.train()
    train_losses = []
    val_losses = []

    for epoch in tqdm(range(num_epochs)):
        total_loss = 0.0
        for batch in train_loader:
            embeddings = batch[0].to(device)  # (batch_size, 6, embed_dim)
            optimizer.zero_grad()

            outputs = model(embeddings)  # (batch_size, embed_dim)
            outputs = F.normalize(outputs, dim=1)

            random_positive_views = torch.randint(0, 6, (outputs.size(0),))
            positive_embeddings = embeddings[torch.arange(outputs.size(0)), random_positive_views]
            positive_embeddings = F.normalize(positive_embeddings, dim=1)

            logits = torch.matmul(outputs, positive_embeddings.T) / temperature
            labels = torch.arange(outputs.size(0), device=outputs.device)
            contrastive_loss = F.cross_entropy(logits, labels)

            view_sims = torch.stack([
                F.cosine_similarity(outputs, F.normalize(embeddings[:, i, :], dim=1))
                for i in range(6)
            ], dim=1)  # (batch_size, 6)
            max_sim = view_sims.max(dim=1)[0]
            cos_loss = 1 - max_sim.mean()

            loss = (1 - alpha) * cos_loss + alpha * contrastive_loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)
        avg_val_loss = evaluate_model(model, val_loader, temperature, alpha, device)
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

    if loss_plot:
        _plot_losses(train_losses, val_losses, num_epochs,
                     "Training and Validation Loss for MLP based model", loss_plot)


def evaluate_model(model, val_loader, temperature=0.05, alpha=0.5, device="cpu"):
    model.eval()
    total_val_loss = 0.0
    with torch.no_grad():
        for batch in val_loader:
            embeddings = batch[0].to(device)
            outputs = model(embeddings)
            outputs = F.normalize(outputs, dim=1)

            random_positive_views = torch.randint(0, 6, (outputs.size(0),))
            positive_embeddings = embeddings[torch.arange(outputs.size(0)), random_positive_views]
            positive_embeddings = F.normalize(positive_embeddings, dim=1)

            logits = torch.matmul(outputs, positive_embeddings.T) / temperature
            labels = torch.arange(outputs.size(0), device=outputs.device)
            contrastive_loss = F.cross_entropy(logits, labels)

            view_sims = torch.stack([
                F.cosine_similarity(outputs, F.normalize(embeddings[:, i, :], dim=1))
                for i in range(6)
            ], dim=1)
            max_sim = view_sims.max(dim=1)[0]
            cos_loss = 1 - max_sim.mean()

            loss = cos_loss + alpha * contrastive_loss
            total_val_loss += loss.item()
    model.train()
    return total_val_loss / len(val_loader)


def _plot_losses(train_losses, val_losses, num_epochs, title, path):
    import matplotlib.pyplot as plt
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    plt.figure()
    plt.plot(range(1, num_epochs + 1), train_losses, label="Train Loss")
    plt.plot(range(1, num_epochs + 1), val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.savefig(path)


def load_model(path, device=None):
    """Load a trained MultiViewImage MLP checkpoint (state_dict) onto ``device``."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = MultiViewImage()
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    return model


def get_multiview_image_mlp_embeddings(image_embeddings, model, device=None):
    """Fuse a model's 6 view embeddings (Tensor (6, D) or list) into a (1, D) vector."""
    device = device or next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        if isinstance(image_embeddings, list):
            image_embeddings = torch.stack(image_embeddings)
        image_embeddings = image_embeddings.to(device)
        combined = model(image_embeddings.unsqueeze(0))  # add batch dim
        combined = F.normalize(combined, dim=1)  # (1, D)
    return combined


def _stack_embeddings(image_embeddings):
    """Stack per-model view embeddings into a single (num_models, 6, D) tensor,
    keeping only entries that share the modal shape."""
    tensors = [v if isinstance(v, torch.Tensor) else torch.stack(v)
               for v in image_embeddings.values()]
    if not tensors:
        raise SystemExit("No image embeddings found in the CSV.")
    ref_shape = tensors[0].shape
    kept = [t for t in tensors if t.shape == ref_shape]
    if len(kept) != len(tensors):
        print(f"[warn] Skipping {len(tensors) - len(kept)} models with mismatched view shapes.")
    return torch.stack(kept)


def main():
    parser = argparse.ArgumentParser(description="Train the MLP multi-view image fusion model.")
    parser.add_argument("--models_csv", required=True,
                        help="Mega CSV from generate_embeddings.py.")
    parser.add_argument("--model_output_path", default="models/multiview_image_mlp.pt",
                        help="Where to save the trained checkpoint.")
    parser.add_argument("--num_epochs", type=int, default=150)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--device", default=None, help="cuda/cpu (default: auto).")
    parser.add_argument("--loss_plot", default=None, help="Optional path to save a loss plot.")
    args = parser.parse_args()

    device = db_common.pick_device(args.device)
    df = db_common.load_models_df(args.models_csv)
    image_embeddings = db_common.load_image_embeddings(df)
    embeddings_tensor = _stack_embeddings(image_embeddings)

    train_size = int(0.7 * len(embeddings_tensor))
    val_size = len(embeddings_tensor) - train_size
    train_dataset, val_dataset = random_split(
        torch.utils.data.TensorDataset(embeddings_tensor), [train_size, val_size])
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    model = MultiViewImage(embed_dim=embeddings_tensor.size(-1)).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    train_model(model, train_loader, val_loader, optimizer,
                num_epochs=args.num_epochs, device=device, loss_plot=args.loss_plot)

    os.makedirs(os.path.dirname(os.path.abspath(args.model_output_path)), exist_ok=True)
    torch.save(model.state_dict(), args.model_output_path)
    print(f"[done] Model saved to {args.model_output_path}")


if __name__ == "__main__":
    main()
