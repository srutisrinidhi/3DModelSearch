"""Shared helpers for the database-creation scripts.

All database builders consume the "mega CSV" produced by ``generate_embeddings.py``
(columns include ``uid``, ``image_path``, ``video_path``, ``image_embedding_path``,
``video_embedding_path``, ``caption``) and load the per-model ``.pt`` embedding files it
points at. These helpers centralize that loading plus a small ChromaDB convenience.
"""

import os

import pandas as pd
import torch


def load_models_df(models_csv):
    """Load the mega CSV as a DataFrame."""
    return pd.read_csv(models_csv)


def _valid_path(value):
    return isinstance(value, str) and value.strip() != "" and os.path.exists(value.strip())


def load_image_embeddings(df, device="cpu"):
    """Return ``{uid: tensor[N_views, D]}`` from each row's image_embedding_path."""
    out = {}
    for _, row in df.iterrows():
        path = row.get("image_embedding_path")
        if _valid_path(path):
            out[str(row["uid"])] = torch.load(path.strip(), map_location=device)
    return out


def load_video_embeddings(df, device="cpu"):
    """Return ``{uid: tensor[D]}`` from each row's video_embedding_path."""
    out = {}
    for _, row in df.iterrows():
        path = row.get("video_embedding_path")
        if _valid_path(path):
            out[str(row["uid"])] = torch.load(path.strip(), map_location=device)
    return out


def load_captions(df):
    """Return ``{uid: caption}`` for rows that have a non-empty caption."""
    out = {}
    if "caption" not in df.columns:
        return out
    for _, row in df.iterrows():
        cap = row.get("caption")
        if isinstance(cap, str) and cap.strip():
            out[str(row["uid"])] = cap.strip()
    return out


def image_folder_for(df, uid):
    """Return the image-render folder for a uid, or None."""
    match = df[df["uid"].astype(str) == str(uid)]
    if match.empty:
        return None
    folder = match.iloc[0].get("image_path")
    return folder.strip() if _valid_path(folder) else None


def pick_device(arg=None):
    """Resolve a torch device string ('cuda' if available, else 'cpu')."""
    if arg:
        return arg
    return "cuda" if torch.cuda.is_available() else "cpu"


def new_collection(persist_directory, name, metric="ip"):
    """Create (idempotently) a fresh ChromaDB collection.

    Any existing collection of the same name in ``persist_directory`` is dropped first,
    so re-running a builder rebuilds cleanly. Returns ``(client, collection)``.
    """
    import chromadb

    client = chromadb.PersistentClient(path=persist_directory)
    try:
        client.delete_collection(name)
    except Exception:
        pass
    collection = client.create_collection(name=name, metadata={"hnsw:space": metric})
    return client, collection
