"""Ablation: metadata (BM25F lexical) retrieval index.

Unlike the other builders this is a sparse lexical index, not a ChromaDB collection. It builds
a BM25F index (see ``scripts/bm25f.py``) over model metadata fields (name / tags / categories /
description) and pickles ``(idx, meta_by_id)`` to ``<persist_directory>/bm25f_index.pkl``.

Input is a CSV with a ``uid`` column and either an ``annotation`` column (a JSON dict keyed by
uid, with name/tags/categories/description) or, failing that, a ``caption`` column used as the
description. The mega CSV from ``generate_embeddings.py`` works as the caption-only fallback.

Example:
    python create_metadata_db.py \
        --metadata_csv out/models_data.csv \
        --persist_directory databases/metadata
"""

import argparse
import os
import pickle
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from bm25f import build_metadata_idx  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="Build a BM25F metadata index.")
    parser.add_argument("--metadata_csv", required=True,
                        help="CSV with a uid column and an annotation (JSON) or caption column.")
    parser.add_argument("--persist_directory", default="databases/metadata",
                        help="Directory to persist the pickled index.")
    args = parser.parse_args()

    idx, meta_by_id = build_metadata_idx(args.metadata_csv)
    os.makedirs(args.persist_directory, exist_ok=True)
    out_path = os.path.join(args.persist_directory, "bm25f_index.pkl")
    with open(out_path, "wb") as f:
        pickle.dump((idx, meta_by_id), f)
    print(f"[done] Saved BM25F index to {out_path}")


if __name__ == "__main__":
    main()
