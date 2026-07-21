"""Ablation: trained multimodal fusion retrieval database ("Trained Fusion Model").

For each model it pools the 6 image views with the trained multi-view MLP model, then fuses
that with the video embedding using the trained ``PromptAwareRetriever`` fusion model, and
stores the result in a ChromaDB ``multimodal_embeddings`` collection (inner-product metric).

Requires two checkpoints:
  * --model_path            the trained fusion model (train_multimodal_model.py)
  * --multiview_model_path  the multi-view MLP pooler (train_multiview_image_mlp.py)

Example:
    python create_multimodal_trained_db.py \
        --models_csv out/models_data.csv \
        --model_path models/multimodal_retriever_model.pt \
        --multiview_model_path models/multiview_image_mlp.pt \
        --persist_directory databases/multimodal_trained
"""

import argparse
import os
import sys

from tqdm import tqdm

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import db_common  # noqa: E402
from multimodal_models.train_multiview_image_mlp import (  # noqa: E402
    load_model, get_multiview_image_mlp_embeddings,
)
from multimodal_models.train_multimodal_model import (  # noqa: E402
    load_multimodal_model, get_multimodal_embeddings,
)

COLLECTION_NAME = "multimodal_embeddings"


def main():
    parser = argparse.ArgumentParser(description="Build the trained-fusion multimodal database.")
    parser.add_argument("--models_csv", required=True, help="Mega CSV from generate_embeddings.py.")
    parser.add_argument("--persist_directory", default="databases/multimodal_trained",
                        help="Directory to persist the ChromaDB collection.")
    parser.add_argument("--model_path", default="models/multimodal_retriever_model.pt",
                        help="Trained fusion model checkpoint (PromptAwareRetriever).")
    parser.add_argument("--multiview_model_path", default="models/multiview_image_mlp.pt",
                        help="Trained multi-view MLP checkpoint used to pool image views.")
    parser.add_argument("--device", default=None, help="cuda/cpu (default: auto).")
    args = parser.parse_args()

    device = db_common.pick_device(args.device)
    df = db_common.load_models_df(args.models_csv)
    image_embeddings = db_common.load_image_embeddings(df, device=device)
    video_embeddings = db_common.load_video_embeddings(df, device=device)

    multiview_model = load_model(args.multiview_model_path, device=device)
    fusion_model = load_multimodal_model(args.model_path, device=device)

    _, collection = db_common.new_collection(args.persist_directory, COLLECTION_NAME)

    added = 0
    for uid, image_embs in tqdm(image_embeddings.items(), desc="Building trained-fusion DB"):
        if uid not in video_embeddings:
            print(f"[warn] {uid}: no video embedding, skipping.")
            continue
        image_vec = get_multiview_image_mlp_embeddings(image_embs, multiview_model, device=device)
        fused = get_multimodal_embeddings(image_vec, video_embeddings[uid], fusion_model)
        collection.add(embeddings=[fused.squeeze(0).detach().cpu().tolist()],
                       ids=[str(uid)], metadatas=[{"uid": uid}])
        added += 1

    print(f"[done] Added {added} models to '{COLLECTION_NAME}' at {args.persist_directory}")


if __name__ == "__main__":
    main()
