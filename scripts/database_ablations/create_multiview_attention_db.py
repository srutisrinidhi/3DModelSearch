"""Ablation: multi-view image fusion via attention.

Stores one embedding per model: the 6 view embeddings pooled by the trained multi-view
attention model. ChromaDB collection ``image_embeddings`` (inner-product metric). Requires a
trained attention checkpoint (see ``multimodal_models/train_multiview_image_attention.py``).

Example:
    python create_multiview_attention_db.py \
        --models_csv out/models_data.csv \
        --model_path models/multiview_image_attention.pt \
        --persist_directory databases/multiview_image_attention
"""

import argparse
import os
import sys

from tqdm import tqdm

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import db_common  # noqa: E402
from multimodal_models.train_multiview_image_attention import (  # noqa: E402
    load_model, get_multiview_image_attention_embeddings,
)

COLLECTION_NAME = "image_embeddings"


def main():
    parser = argparse.ArgumentParser(description="Build an attention-fused multi-view image database.")
    parser.add_argument("--models_csv", required=True, help="Mega CSV from generate_embeddings.py.")
    parser.add_argument("--persist_directory", default="databases/multiview_image_attention",
                        help="Directory to persist the ChromaDB collection.")
    parser.add_argument("--model_path", default="models/multiview_image_attention.pt",
                        help="Trained multi-view attention checkpoint.")
    parser.add_argument("--device", default=None, help="cuda/cpu (default: auto).")
    args = parser.parse_args()

    device = db_common.pick_device(args.device)
    df = db_common.load_models_df(args.models_csv)
    image_embeddings = db_common.load_image_embeddings(df, device=device)
    model = load_model(args.model_path, device=device)

    _, collection = db_common.new_collection(args.persist_directory, COLLECTION_NAME)

    added = 0
    for uid, embs in tqdm(image_embeddings.items(), desc="Building multiview-attention DB"):
        vec = get_multiview_image_attention_embeddings(embs, model, device=device).squeeze(0)
        collection.add(embeddings=[vec.detach().cpu().tolist()],
                       ids=[str(uid)], metadatas=[{"uid": uid}])
        added += 1

    print(f"[done] Added {added} models to '{COLLECTION_NAME}' at {args.persist_directory}")


if __name__ == "__main__":
    main()
