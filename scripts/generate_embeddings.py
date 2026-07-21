"""Generate LanguageBind embeddings for a set of 3D models.

Given a folder of models plus their pre-rendered image and video folders (see
`scripts/Blender/render_images.py` / `render_videos.py`), this script matches each
model to its renders **by name**, computes:

  * an image embedding per model  = stack of the per-view image embeddings, shape [N_views, D]
  * a video embedding per model   = a single embedding vector, shape [D]

and writes one `.pt` file per model, plus a "mega CSV" mapping every model to its
render and embedding paths.

Matching convention (matches the Step-1 render scripts' output layout):
    model file : <models_dir>/<uid>.<glb|gltf|fbx|obj>
    image dir  : <images_dir>/<uid>/view*.jpg
    video file : <videos_dir>/<uid>.mp4
where `uid` is the model filename without its extension.

Dependency: the LanguageBind package must be importable. Either install it, or point
`--languagebind_path` (or the LANGUAGEBIND_PATH env var) at a local LanguageBind clone.

Example:
    python generate_embeddings.py \
        --models_dir models/ --images_dir renders/images/ \
        --videos_dir renders/videos/ --output_dir out/ \
        --captions_csv captions.csv
"""

import argparse
import os
import sys

import pandas as pd
import torch
from tqdm import tqdm

SUPPORTED_EXTS = (".glb", ".gltf", ".fbx", ".obj")

# Device / HF cache dir. Auto-detect CUDA so callers that don't run _configure()
# (e.g. retrieval.py, the DB scripts) still use the GPU when available.
_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
_CACHE_DIR = "./cache_dir"

# Lazy singletons so each LanguageBind model + processor is loaded exactly once.
_IMAGE_MODEL = None
_VIDEO_MODEL = None


# ---------------------------------------------------------------------------
# LanguageBind setup
# ---------------------------------------------------------------------------

def _add_languagebind_to_path(languagebind_path=None):
    """Put the LanguageBind checkout on sys.path (honors LANGUAGEBIND_PATH env var)."""
    lb_path = languagebind_path or os.environ.get("LANGUAGEBIND_PATH")
    if lb_path and lb_path not in sys.path:
        sys.path.append(lb_path)


# Run at import time so ANY caller (retrieval, DB scripts, trainers) can import
# LanguageBind without first calling _configure().
_add_languagebind_to_path()


def _configure(device=None, cache_dir="./cache_dir", languagebind_path=None):
    """Set the global device / cache dir and make LanguageBind importable."""
    global _DEVICE, _CACHE_DIR
    _CACHE_DIR = cache_dir
    _DEVICE = device or ("cuda" if torch.cuda.is_available() else "cpu")
    _add_languagebind_to_path(languagebind_path)


def _load_image_model():
    """Load (once) the LanguageBind image model + processor."""
    global _IMAGE_MODEL
    if _IMAGE_MODEL is None:
        from LanguageBind.languagebind import (
            LanguageBindImage,
            LanguageBindImageTokenizer,
            LanguageBindImageProcessor,
        )
        ckpt = "LanguageBind/LanguageBind_Image"
        model = LanguageBindImage.from_pretrained(ckpt, cache_dir=_CACHE_DIR)
        tokenizer = LanguageBindImageTokenizer.from_pretrained(ckpt, cache_dir=_CACHE_DIR)
        processor = LanguageBindImageProcessor(model.config, tokenizer)
        model.eval().to(_DEVICE)
        _IMAGE_MODEL = (model, processor)
    return _IMAGE_MODEL


def _load_video_model():
    """Load (once) the LanguageBind video model + processor."""
    global _VIDEO_MODEL
    if _VIDEO_MODEL is None:
        from LanguageBind.languagebind import (
            LanguageBindVideo,
            LanguageBindVideoTokenizer,
            LanguageBindVideoProcessor,
        )
        ckpt = "LanguageBind/LanguageBind_Video_FT"  # also 'LanguageBind/LanguageBind_Video'
        model = LanguageBindVideo.from_pretrained(ckpt, cache_dir=_CACHE_DIR)
        tokenizer = LanguageBindVideoTokenizer.from_pretrained(ckpt, cache_dir=_CACHE_DIR)
        processor = LanguageBindVideoProcessor(model.config, tokenizer)
        model.eval().to(_DEVICE)
        _VIDEO_MODEL = (model, processor)
    return _VIDEO_MODEL


# ---------------------------------------------------------------------------
# Embedding helpers (same signatures / return values as the original script)
# ---------------------------------------------------------------------------

def get_image_embeddings(image_paths):
    """Return the LanguageBind image embedding for the given image path(s)."""
    model, processor = _load_image_model()
    data = processor(image_paths, [""], return_tensors="pt")
    data = {k: v.to(_DEVICE) for k, v in data.items()}
    with torch.no_grad():
        out = model(**data)
    return out.image_embeds[0].detach().cpu()


def get_video_embeddings(video_paths):
    """Return the LanguageBind video embedding for the given video path(s)."""
    model, processor = _load_video_model()
    data = processor(video_paths, "", return_tensors="pt")
    data = {k: v.to(_DEVICE) for k, v in data.items()}
    with torch.no_grad():
        out = model(**data)
    return out.image_embeds[0].detach().cpu()


def get_text_embeddings(text, reference_image_path):
    """Return the LanguageBind text embedding for `text`.

    The LanguageBind image processor requires an accompanying image, so a
    reference image path must be supplied (its embedding is discarded).
    """
    model, processor = _load_image_model()
    data = processor(reference_image_path, [text], return_tensors="pt")
    data = {k: v.to(_DEVICE) for k, v in data.items()}
    with torch.no_grad():
        out = model(**data)
    return out.text_embeds[0].detach().cpu()


# ---------------------------------------------------------------------------
# Batch driver
# ---------------------------------------------------------------------------

def load_captions(captions_csv):
    """Load a {uid: caption} map from a two-column CSV (col0=uid, col1=caption)."""
    if not captions_csv:
        return {}
    df = pd.read_csv(captions_csv)
    uid_col, cap_col = df.columns[0], df.columns[1]
    return {str(row[uid_col]): row[cap_col] for _, row in df.iterrows()}


def discover_models(models_dir):
    """Return a sorted list of (uid, model_path) for supported models in a folder."""
    models = []
    for fn in sorted(os.listdir(models_dir)):
        if fn.lower().endswith(SUPPORTED_EXTS):
            uid = os.path.splitext(fn)[0]
            models.append((uid, os.path.join(models_dir, fn)))
    return models


def embed_images(image_folder, image_ext):
    """Embed every view image in a folder and return a stacked [N_views, D] tensor,
    or None if the folder has no matching images."""
    if not os.path.isdir(image_folder):
        return None
    view_files = sorted(
        os.path.join(image_folder, f)
        for f in os.listdir(image_folder)
        if f.lower().endswith(f".{image_ext.lower()}")
    )
    if not view_files:
        return None
    embs = [get_image_embeddings([vf]) for vf in view_files]
    return torch.stack(embs)


def main():
    parser = argparse.ArgumentParser(
        description="Generate LanguageBind image/video embeddings for 3D models.")
    parser.add_argument("--models_dir", required=True,
                        help="Folder of model files (.glb/.gltf/.fbx/.obj).")
    parser.add_argument("--images_dir", required=True,
                        help="Folder of image renders, one subfolder per model (<uid>/).")
    parser.add_argument("--videos_dir", required=True,
                        help="Folder of video renders (<uid>.mp4).")
    parser.add_argument("--output_dir", required=True,
                        help="Output folder for Embeddings/ and the mega CSV.")
    parser.add_argument("--csv_name", default="models_data.csv",
                        help="Filename for the mega CSV written under --output_dir.")
    parser.add_argument("--captions_csv", default=None,
                        help="Optional CSV (col0=uid, col1=caption) for the caption column.")
    parser.add_argument("--image_ext", default="jpg",
                        help="Extension of the view images. Default: jpg.")
    parser.add_argument("--device", default=None,
                        help="Torch device override (e.g. cuda, cpu). Default: auto.")
    parser.add_argument("--cache_dir", default="./cache_dir",
                        help="HuggingFace cache dir for LanguageBind weights.")
    parser.add_argument("--languagebind_path", default=None,
                        help="Path to a local LanguageBind clone (else use "
                             "LANGUAGEBIND_PATH env var, or an installed package).")
    args = parser.parse_args()

    _configure(device=args.device, cache_dir=args.cache_dir,
               languagebind_path=args.languagebind_path)
    print(f"[info] device={_DEVICE}, cache_dir={_CACHE_DIR}")

    emb_dir = os.path.join(args.output_dir, "Embeddings")
    os.makedirs(emb_dir, exist_ok=True)

    captions = load_captions(args.captions_csv)
    models = discover_models(args.models_dir)
    if not models:
        raise SystemExit(f"No supported model files found in: {args.models_dir}")
    print(f"[info] {len(models)} model(s) found.")

    rows = []
    for uid, model_path in tqdm(models, total=len(models)):
        image_folder = os.path.join(args.images_dir, uid)
        video_file = os.path.join(args.videos_dir, f"{uid}.mp4")
        image_emb_path = os.path.join(emb_dir, f"{uid}_image.pt")
        video_emb_path = os.path.join(emb_dir, f"{uid}_video.pt")

        try:
            # ---- image embedding (resume if already computed) ----
            if os.path.exists(image_emb_path):
                pass  # reuse existing
            else:
                img_emb = embed_images(image_folder, args.image_ext)
                if img_emb is not None:
                    torch.save(img_emb, image_emb_path)
                    print(f"[ok] {uid}: image embedding {tuple(img_emb.shape)}")
                else:
                    print(f"[warn] {uid}: no images at {image_folder}")

            # ---- video embedding (resume if already computed) ----
            if os.path.exists(video_emb_path):
                pass  # reuse existing
            elif os.path.exists(video_file):
                vid_emb = get_video_embeddings([video_file])
                torch.save(vid_emb, video_emb_path)
                print(f"[ok] {uid}: video embedding {tuple(vid_emb.shape)}")
            else:
                print(f"[warn] {uid}: no video at {video_file}")
        except Exception as exc:  # keep going on a bad model
            print(f"[error] Failed to embed '{uid}': {exc}")

        rows.append({
            "uid": uid,
            "model_path": model_path,
            "image_path": image_folder if os.path.isdir(image_folder) else "",
            "video_path": video_file if os.path.exists(video_file) else "",
            "image_embedding_path": image_emb_path if os.path.exists(image_emb_path) else "",
            "video_embedding_path": video_emb_path if os.path.exists(video_emb_path) else "",
            "caption": captions.get(uid, ""),
        })

    csv_path = os.path.join(args.output_dir, args.csv_name)
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    print(f"[done] Wrote {len(rows)} rows to {csv_path}")


if __name__ == "__main__":
    main()
