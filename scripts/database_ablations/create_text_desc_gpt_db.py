"""Ablation: GPT-generated text-description retrieval database.

For each model it sends the rendered views to GPT-4o, asks for a detailed description, embeds
that description with LanguageBind's text encoder, and stores it in a ChromaDB
``text_description_embeddings`` collection (inner-product metric).

Requires an OpenAI API key in the ``OPENAI_API_KEY`` environment variable, plus LanguageBind
(for text embeddings). Input is the mega CSV from ``generate_embeddings.py``.

Example:
    export OPENAI_API_KEY=sk-...
    python create_text_desc_gpt_db.py \
        --models_csv out/models_data.csv \
        --persist_directory databases/text_descriptions_gpt
"""

import argparse
import base64
import os
import sys

from tqdm import tqdm

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import db_common  # noqa: E402
from generate_embeddings import get_text_embeddings  # noqa: E402

COLLECTION_NAME = "text_description_embeddings"
PROMPT = ("Each of these images are different views of the same 3D object. Describe the "
          "object in as much detail as possible. I will use this description to search for "
          "the object later.")


def encode_image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        encoded = base64.b64encode(image_file.read()).decode("utf-8")
    return f"data:image/jpeg;base64,{encoded}"


def ask_gpt(client, prompt, view_paths, model="gpt-4o", max_tokens=3500):
    content = [{"type": "image_url", "image_url": {"url": encode_image_to_base64(v)}}
               for v in view_paths]
    content.append({"type": "text", "text": prompt})
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": content}],
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content


def views_for(folder, image_ext):
    return [os.path.join(folder, f) for f in sorted(os.listdir(folder))
            if f.lower().endswith(f".{image_ext.lower()}")]


def main():
    parser = argparse.ArgumentParser(description="Build a GPT text-description database.")
    parser.add_argument("--models_csv", required=True, help="Mega CSV from generate_embeddings.py.")
    parser.add_argument("--persist_directory", default="databases/text_descriptions_gpt",
                        help="Directory to persist the ChromaDB collection.")
    parser.add_argument("--image_ext", default="jpg", help="View image extension.")
    parser.add_argument("--gpt_model", default="gpt-4o", help="OpenAI vision model to use.")
    parser.add_argument("--descriptions_out", default=None,
                        help="Optional CSV path to also save the generated descriptions.")
    args = parser.parse_args()

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("Set the OPENAI_API_KEY environment variable before running.")
    from openai import OpenAI
    client = OpenAI(api_key=api_key)

    df = db_common.load_models_df(args.models_csv)
    _, collection = db_common.new_collection(args.persist_directory, COLLECTION_NAME)

    descriptions = {}
    added = 0
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Building GPT-description DB"):
        uid = str(row["uid"])
        folder = db_common.image_folder_for(df, uid)
        if not folder:
            print(f"[warn] {uid}: no image folder, skipping.")
            continue
        views = views_for(folder, args.image_ext)
        if not views:
            print(f"[warn] {uid}: no view images, skipping.")
            continue
        try:
            description = ask_gpt(client, PROMPT, views, model=args.gpt_model)
            embedding = get_text_embeddings(description, reference_image_path=views[0])
            collection.add(embeddings=[embedding.detach().cpu().tolist()],
                           ids=[uid], metadatas=[{"uid": uid}])
            descriptions[uid] = description
            added += 1
        except Exception as exc:
            print(f"[error] {uid}: {exc}")

    if args.descriptions_out:
        import pandas as pd
        pd.DataFrame(
            [{"uid": u, "description": d} for u, d in descriptions.items()]
        ).to_csv(args.descriptions_out, index=False)
        print(f"[info] Saved descriptions to {args.descriptions_out}")

    print(f"[done] Added {added} models to '{COLLECTION_NAME}' at {args.persist_directory}")


if __name__ == "__main__":
    main()
