"""Ablation: video-only retrieval database.

Stores one embedding per model: its video embedding. ChromaDB collection
``video_embeddings`` (inner-product metric). Input is the mega CSV from
``generate_embeddings.py``.

Example:
    python create_video_db.py \
        --models_csv out/models_data.csv \
        --persist_directory databases/video
"""

import argparse
import os
import sys

from tqdm import tqdm

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import db_common  # noqa: E402

COLLECTION_NAME = "video_embeddings"


def main():
    parser = argparse.ArgumentParser(description="Build a video-only database.")
    parser.add_argument("--models_csv", required=True, help="Mega CSV from generate_embeddings.py.")
    parser.add_argument("--persist_directory", default="databases/video",
                        help="Directory to persist the ChromaDB collection.")
    args = parser.parse_args()

    df = db_common.load_models_df(args.models_csv)
    video_embeddings = db_common.load_video_embeddings(df)

    _, collection = db_common.new_collection(args.persist_directory, COLLECTION_NAME)

    added = 0
    for uid, vec in tqdm(video_embeddings.items(), desc="Building video DB"):
        collection.add(embeddings=[vec.detach().cpu().tolist()],
                       ids=[str(uid)], metadatas=[{"uid": uid}])
        added += 1

    print(f"[done] Added {added} models to '{COLLECTION_NAME}' at {args.persist_directory}")


if __name__ == "__main__":
    main()
