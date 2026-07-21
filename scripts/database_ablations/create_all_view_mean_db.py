"""Ablation: average-across-all-views retrieval database.

Stores one embedding per model: the mean of all 6 view embeddings. ChromaDB collection
``image_embeddings`` (inner-product metric). Input is the mega CSV from
``generate_embeddings.py``.

Example:
    python create_all_view_mean_db.py \
        --models_csv out/models_data.csv \
        --persist_directory databases/all_image_mean
"""

import argparse
import os
import sys

import torch
from tqdm import tqdm

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import db_common  # noqa: E402

COLLECTION_NAME = "image_embeddings"


def main():
    parser = argparse.ArgumentParser(description="Build a mean-of-all-views image database.")
    parser.add_argument("--models_csv", required=True, help="Mega CSV from generate_embeddings.py.")
    parser.add_argument("--persist_directory", default="databases/all_image_mean",
                        help="Directory to persist the ChromaDB collection.")
    args = parser.parse_args()

    df = db_common.load_models_df(args.models_csv)
    image_embeddings = db_common.load_image_embeddings(df)

    _, collection = db_common.new_collection(args.persist_directory, COLLECTION_NAME)

    added = 0
    for uid, embs in tqdm(image_embeddings.items(), desc="Building all-view-mean DB"):
        vec = torch.mean(embs, dim=0)  # average across the 6 views
        collection.add(embeddings=[vec.detach().cpu().tolist()],
                       ids=[str(uid)], metadatas=[{"uid": uid}])
        added += 1

    print(f"[done] Added {added} models to '{COLLECTION_NAME}' at {args.persist_directory}")


if __name__ == "__main__":
    main()
