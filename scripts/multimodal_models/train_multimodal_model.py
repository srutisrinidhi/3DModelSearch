"""Train the trained multimodal fusion model ("Trained Fusion Model").

``PromptAwareRetriever`` wraps a ``CrossModalAttentionFusion`` module in which the (MLP-pooled)
image embedding cross-attends to the video embedding; the fused vector is trained to align with
the caption's text embedding (loose-supervision loss).

Inputs all come from the mega CSV produced by ``generate_embeddings.py``: the per-model image
embeddings (pooled to a single vector via a pre-trained multi-view MLP model), the video
embeddings, and text embeddings computed on the fly from the caption column (LanguageBind).

Example:
    python train_multimodal_model.py \
        --models_csv out/models_data.csv \
        --multiview_model_path models/multiview_image_mlp.pt \
        --model_output_path models/multimodal_retriever_model.pt
"""

import argparse
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

# Make sibling modules (db_common, generate_embeddings) importable from the scripts/ root.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import db_common  # noqa: E402
from multimodal_models.train_multiview_image_mlp import (  # noqa: E402
    load_model, get_multiview_image_mlp_embeddings,
)


class CrossModalAttentionFusion(nn.Module):
    def __init__(self, embedding_dim, num_heads=4):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(embed_dim=embedding_dim, num_heads=num_heads, batch_first=True)
        self.norm = nn.LayerNorm(embedding_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(embedding_dim, embedding_dim)
        )

    def forward(self, image_emb, video_emb):
        image_emb = image_emb.unsqueeze(1)  # (B, 1, D)
        video_emb = video_emb.unsqueeze(1)
        image_emb = F.normalize(image_emb, dim=-1)
        video_emb = F.normalize(video_emb, dim=-1)

        # Cross-attention: Image attends to Video
        attn_output, _ = self.cross_attn(query=image_emb, key=video_emb, value=video_emb)
        fused = self.norm(image_emb + attn_output)

        combined = torch.cat([fused, image_emb], dim=-1)  # (B, 1, 2D)
        out = self.mlp(combined).squeeze(1)
        return out


class PromptAwareRetriever(nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()
        self.fusion = CrossModalAttentionFusion(embedding_dim)

    def forward(self, image_embedding, video_embedding):
        return self.fusion(image_embedding, video_embedding)


def info_nce_loss(text_emb, fused_emb, temperature=0.03):
    text_emb = F.normalize(text_emb, dim=-1)
    fused_emb = F.normalize(fused_emb, dim=-1)
    logits = torch.matmul(text_emb, fused_emb.T) / temperature
    labels = torch.arange(logits.size(0), device=logits.device)
    return F.cross_entropy(logits, labels)


def loose_supervision_loss(text_emb, fused_emb, image_emb, video_emb, alpha=0.2, beta=0.3):
    text_emb = F.normalize(text_emb, dim=-1)
    fused_emb = F.normalize(fused_emb, dim=-1)
    image_emb = F.normalize(image_emb, dim=-1)
    video_emb = F.normalize(video_emb, dim=-1)

    alignment_loss = 1 - F.cosine_similarity(text_emb, fused_emb).mean()
    modality_consistency = 1 - F.cosine_similarity(image_emb, video_emb).mean()
    fusion_consistency = 1 - (F.cosine_similarity(fused_emb, image_emb).mean() +
                              F.cosine_similarity(fused_emb, video_emb).mean()) / 2
    return alpha * alignment_loss + beta * modality_consistency + (1 - alpha - beta) * fusion_consistency


def train(model, dataloader, val_loader, optimizer, num_epochs=10, device="cuda", loss_plot=None):
    model = model.to(device)
    train_losses = []
    val_losses = []

    use_amp = (str(device) == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    for epoch in tqdm(range(num_epochs)):
        model.train()
        total_loss = 0.0
        total_samples = 0
        for batch in tqdm(dataloader):
            text_emb = batch["text_emb"].to(device, non_blocking=True)
            image_emb = batch["image_emb"].to(device, non_blocking=True)
            video_emb = batch["video_emb"].to(device, non_blocking=True)

            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=use_amp):
                fused_emb = model(image_emb, video_emb)
                loss = loose_supervision_loss(text_emb, fused_emb, image_emb, video_emb)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            batch_size = text_emb.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size

        train_losses.append(total_loss / total_samples)
        print(f"Epoch {epoch+1}: Loss = {total_loss / total_samples:.4f}")

        model.eval()
        with torch.no_grad():
            val_loss = 0.0
            val_samples = 0
            for val_batch in val_loader:
                text_emb = val_batch["text_emb"].to(device, non_blocking=True)
                image_emb = val_batch["image_emb"].to(device, non_blocking=True)
                video_emb = val_batch["video_emb"].to(device, non_blocking=True)

                fused_emb = model(image_emb, video_emb)
                loss = info_nce_loss(text_emb, fused_emb)
                batch_size = text_emb.size(0)
                val_loss += loss.item() * batch_size
                val_samples += batch_size
            val_losses.append(val_loss / max(val_samples, 1))
            print(f"Validation Loss: {val_losses[-1]:.4f}")

    if loss_plot:
        import matplotlib.pyplot as plt
        os.makedirs(os.path.dirname(os.path.abspath(loss_plot)), exist_ok=True)
        plt.figure()
        plt.plot(train_losses, label="Training Loss", color="blue")
        plt.plot(val_losses, label="Validation Loss", color="orange")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training and Validation Loss")
        plt.legend()
        plt.grid(True)
        plt.savefig(loss_plot)

    return model


def get_multimodal_embeddings(image_emb, video_emb, model):
    """Fuse a single model's (pooled) image and video embeddings into one vector."""
    device = next(model.parameters()).device
    if image_emb.ndim == 1:
        image_emb = image_emb.unsqueeze(0)
    if video_emb.ndim == 1:
        video_emb = video_emb.unsqueeze(0)
    image_emb = image_emb.to(device)
    video_emb = video_emb.to(device)
    with torch.no_grad():
        return model(image_emb, video_emb)


def load_multimodal_model(model_path, device=None):
    """Load a trained PromptAwareRetriever checkpoint (state_dict) onto ``device``."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = PromptAwareRetriever(embedding_dim=768)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model


class PromptRetrievalDataset(torch.utils.data.Dataset):
    def __init__(self, uids, image_embeddings, video_embeddings, text_embeddings):
        self.samples = []
        for uid in uids:
            if uid in image_embeddings and uid in video_embeddings and uid in text_embeddings:
                self.samples.append({
                    "text_emb": text_embeddings[uid],
                    "image_emb": image_embeddings[uid],
                    "video_emb": video_embeddings[uid],
                })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def build_training_embeddings(df, multiview_model_path, device):
    """Pool image views (via the multi-view MLP model), gather videos, and compute text
    embeddings from captions. Returns three {uid: tensor} dicts over the common uids."""
    from generate_embeddings import get_text_embeddings

    multiview_image_model = load_model(multiview_model_path, device=device)
    raw_image = db_common.load_image_embeddings(df, device=device)
    raw_video = db_common.load_video_embeddings(df, device=device)
    captions = db_common.load_captions(df)

    common = set(raw_image) & set(raw_video) & set(captions)
    print(f"[info] {len(common)} models have image + video + caption.")

    image_embeds, video_embeds, text_embeds = {}, {}, {}
    for uid in tqdm(sorted(common), desc="Preparing training embeddings"):
        ref_image = _reference_image(df, uid)
        if ref_image is None:
            continue
        image_embeds[uid] = get_multiview_image_mlp_embeddings(
            raw_image[uid], multiview_image_model, device=device).flatten(0, 1)
        video_embeds[uid] = raw_video[uid]
        text_embeds[uid] = get_text_embeddings(captions[uid], ref_image)
    return image_embeds, video_embeds, text_embeds


def _reference_image(df, uid):
    """Return a single view image path for a uid (LanguageBind needs one for text embeds)."""
    folder = db_common.image_folder_for(df, uid)
    if not folder:
        return None
    for fn in sorted(os.listdir(folder)):
        if fn.lower().endswith((".jpg", ".jpeg", ".png")):
            return os.path.join(folder, fn)
    return None


def main():
    parser = argparse.ArgumentParser(description="Train the trained multimodal fusion model.")
    parser.add_argument("--models_csv", required=True,
                        help="Mega CSV from generate_embeddings.py.")
    parser.add_argument("--multiview_model_path", default="models/multiview_image_mlp.pt",
                        help="Pre-trained multi-view MLP model used to pool the 6 image views.")
    parser.add_argument("--model_output_path", default="models/multimodal_retriever_model.pt",
                        help="Where to save the trained fusion checkpoint.")
    parser.add_argument("--num_epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--device", default=None, help="cuda/cpu (default: auto).")
    parser.add_argument("--loss_plot", default=None, help="Optional path to save a loss plot.")
    args = parser.parse_args()

    device = db_common.pick_device(args.device)
    df = db_common.load_models_df(args.models_csv)
    image_embeddings, video_embeddings, text_embeddings = build_training_embeddings(
        df, args.multiview_model_path, device)

    dataset = PromptRetrievalDataset(
        uids=list(image_embeddings.keys()),
        image_embeddings=image_embeddings,
        video_embeddings=video_embeddings,
        text_embeddings=text_embeddings,
    )
    if len(dataset) == 0:
        raise SystemExit("No training samples (need image + video + caption per model).")

    train_size = int(0.7 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    print(f"[info] train={train_size}, val={val_size}")

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    model = PromptAwareRetriever(embedding_dim=768)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    model = train(model, train_loader, val_loader, optimizer,
                  num_epochs=args.num_epochs, device=device, loss_plot=args.loss_plot)

    os.makedirs(os.path.dirname(os.path.abspath(args.model_output_path)), exist_ok=True)
    torch.save(model.state_dict(), args.model_output_path)
    print(f"[done] Model saved to {args.model_output_path}")


if __name__ == "__main__":
    main()
