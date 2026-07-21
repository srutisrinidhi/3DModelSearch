"""Shared retrieval logic: embed a text/image/video query and find the closest models.

Used by both ``main.py`` (CLI) and ``gradio_viewer.py`` (web demo). Assumes the renders,
embeddings, trained model(s), and the ChromaDB collection already exist.
"""

import os
import sys
import tempfile

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
import db_common  # noqa: E402
from generate_embeddings import (  # noqa: E402
    get_text_embeddings, get_image_embeddings, get_video_embeddings,
)

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
VIDEO_EXTS = (".mp4", ".mov", ".avi", ".mkv", ".webm")


def detect_modality(query):
    """Infer the query modality from the input (a file path vs. free text)."""
    if isinstance(query, str) and os.path.isfile(query):
        ext = os.path.splitext(query)[1].lower()
        if ext in IMAGE_EXTS:
            return "image"
        if ext in VIDEO_EXTS:
            return "video"
    return "text"


def _blank_reference_image():
    """LanguageBind's image processor needs an image even to encode text; a blank image
    works because the text embedding does not depend on the image. Cached on disk."""
    path = os.path.join(tempfile.gettempdir(), "3dmodelsearch_blank_ref.jpg")
    if not os.path.exists(path):
        from PIL import Image
        Image.new("RGB", (224, 224), (255, 255, 255)).save(path)
    return path


def embed_query(query, modality):
    """Return the LanguageBind embedding (Tensor [D]) for a query in the given modality."""
    if modality == "text":
        return get_text_embeddings(query, reference_image_path=_blank_reference_image())
    if modality == "image":
        return get_image_embeddings([query])
    if modality == "video":
        return get_video_embeddings([query])
    raise ValueError(f"Unknown modality: {modality}")


def load_collection(persist_directory, collection_name):
    """Open an existing ChromaDB collection."""
    import chromadb
    client = chromadb.PersistentClient(path=persist_directory)
    return client.get_collection(collection_name)


def retrieve_uids(collection, query_embedding, top_k=5):
    """Return the top-k model uids nearest to the query embedding.

    Over-fetches and de-duplicates by uid so this also works against per-view collections
    (e.g. the all-view-max ablation) that store multiple vectors per model.
    """
    try:
        count = collection.count()
    except Exception:
        count = top_k * 6
    n = min(max(top_k * 6, top_k), max(count, 1))
    results = collection.query(
        query_embeddings=[query_embedding.detach().cpu().tolist()],
        n_results=n,
        include=["metadatas", "distances"],
    )
    pairs = sorted(zip(results["metadatas"][0], results["distances"][0]), key=lambda x: x[1])
    ordered = []
    for meta, _ in pairs:
        uid = meta.get("uid")
        if uid is not None and uid not in ordered:
            ordered.append(uid)
        if len(ordered) >= top_k:
            break
    return ordered


def uid_to_model_path(df, uid):
    """Map a uid to its model file path via the mega CSV, or None."""
    row = df[df["uid"].astype(str) == str(uid)]
    if row.empty:
        return None
    path = row.iloc[0].get("model_path")
    return path if isinstance(path, str) and path.strip() else None


def search(query, models_csv, persist_directory, collection_name="multimodal_embeddings",
           modality="auto", top_k=5, collection=None, df=None):
    """Full pipeline: embed the query, retrieve top-k uids, map to model paths.

    Returns a list of ``(uid, model_path)`` tuples. Pass ``collection``/``df`` to reuse
    already-loaded objects (e.g. from a long-running Gradio session).
    """
    if modality == "auto":
        modality = detect_modality(query)
    if df is None:
        df = db_common.load_models_df(models_csv)
    if collection is None:
        collection = load_collection(persist_directory, collection_name)

    query_embedding = embed_query(query, modality)
    uids = retrieve_uids(collection, query_embedding, top_k=top_k)
    return [(uid, uid_to_model_path(df, uid)) for uid in uids]
