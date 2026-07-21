"""Build the main retrieval database ("Mean of Multiview Image + Video").

This is the model intended for general use. For each model it pools the 6 image-view
embeddings with the pre-trained multi-view MLP fusion model, then averages that with the
video embedding, and stores the result in a ChromaDB ``multimodal_embeddings`` collection.

Inputs come from the mega CSV produced by ``generate_embeddings.py``. Requires a trained
multi-view MLP checkpoint (train it with ``multimodal_models/train_multiview_image_mlp.py``).

Example:
    python create_model_database.py \
        --models_csv out/models_data.csv \
        --model_path models/multiview_image_mlp.pt \
        --persist_directory databases/model_database
"""

import argparse
import os
import sys

import torch
from tqdm import tqdm

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
import db_common  # noqa: E402
from multimodal_models.train_multiview_image_mlp import (  # noqa: E402
    load_model, get_multiview_image_mlp_embeddings,
)

COLLECTION_NAME = "multimodal_embeddings"


def main():
    parser = argparse.ArgumentParser(
        description="Build the main multimodal (mean of multiview image + video) database.")
    parser.add_argument("--models_csv", required=True,
                        help="Mega CSV from generate_embeddings.py.")
    parser.add_argument("--persist_directory", default="databases/model_database",
                        help="Directory to persist the ChromaDB collection.")
    parser.add_argument("--model_path", default="models/multiview_image_mlp.pt",
                        help="Pre-trained multi-view MLP checkpoint used to pool image views.")
    parser.add_argument("--device", default=None, help="cuda/cpu (default: auto).")
    args = parser.parse_args()

    device = db_common.pick_device(args.device)
    df = db_common.load_models_df(args.models_csv)
    image_embeddings = db_common.load_image_embeddings(df, device=device)
    video_embeddings = db_common.load_video_embeddings(df, device=device)

    mlp_model = load_model(args.model_path, device=device)

    _, collection = db_common.new_collection(args.persist_directory, COLLECTION_NAME)

    added = 0
    for uid, image_embs in tqdm(image_embeddings.items(), desc="Building multimodal DB"):
        if uid not in video_embeddings:
            print(f"[warn] {uid}: no video embedding, skipping.")
            continue
        # Pool the 6 image views (MLP model) -> (1, D), then average with the video embedding.
        image_vec = get_multiview_image_mlp_embeddings(image_embs, mlp_model, device=device).squeeze(0)
        video_vec = video_embeddings[uid].to(image_vec.device)
        fused = (image_vec + video_vec) / 2
        collection.add(
            embeddings=[fused.detach().cpu().tolist()],
            ids=[str(uid)],
            metadatas=[{"uid": uid}],
        )
        added += 1

    print(f"[done] Added {added} models to '{COLLECTION_NAME}' at {args.persist_directory}")


if __name__ == "__main__":
    main()
