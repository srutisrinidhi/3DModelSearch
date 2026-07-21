"""Retrieve 3D models for a text / image / video query.

Assumes you have already rendered the models, generated embeddings, trained any needed model,
and built the retrieval database (see the README). Prints the path to each retrieved model.

Examples:
    # text query
    python main.py --query "a red sports car" \
        --models_csv out/models_data.csv --persist_directory databases/model_database

    # image query (auto-detected from the file extension)
    python main.py --query query.jpg \
        --models_csv out/models_data.csv --persist_directory databases/model_database

    # video query
    python main.py --query query.mp4 \
        --models_csv out/models_data.csv --persist_directory databases/model_database
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
import retrieval  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="Retrieve 3D models for a query.")
    parser.add_argument("--query", required=True,
                        help="A text query, or a path to a query image / video.")
    parser.add_argument("--models_csv", required=True,
                        help="Mega CSV from generate_embeddings.py (maps uid -> model path).")
    parser.add_argument("--persist_directory", default="databases/model_database",
                        help="ChromaDB directory of the retrieval database.")
    parser.add_argument("--collection_name", default="multimodal_embeddings",
                        help="Name of the ChromaDB collection to query.")
    parser.add_argument("--modality", default="auto", choices=["auto", "text", "image", "video"],
                        help="Query modality (default: auto-detect from the input).")
    parser.add_argument("--top_k", type=int, default=5, help="Number of models to retrieve.")
    args = parser.parse_args()

    results = retrieval.search(
        query=args.query,
        models_csv=args.models_csv,
        persist_directory=args.persist_directory,
        collection_name=args.collection_name,
        modality=args.modality,
        top_k=args.top_k,
    )

    if not results:
        print("No models retrieved.", file=sys.stderr)
        return

    for rank, (uid, model_path) in enumerate(results, start=1):
        if model_path:
            print(model_path)
        else:
            print(f"# rank {rank}: uid {uid} has no model_path in the CSV", file=sys.stderr)


if __name__ == "__main__":
    main()
