# 3DModelSearch

Embeddings-based retrieval of 3D models. Each model is rendered to multi-view images and a
video, encoded into [LanguageBind](https://github.com/PKU-YuanGroup/LanguageBind) embeddings,
and indexed in a [ChromaDB](https://www.trychroma.com/) database you can query with text,
images, or video.

## Setup

LanguageBind is included as a git submodule:

```bash
git clone --recurse-submodules https://github.com/srutisrinidhi/3DModelSearch
cd 3DModelSearch
# already cloned without submodules? -> git submodule update --init --recursive

conda create -n ModelSearch python=3.9 -y
conda activate ModelSearch

# PyTorch (match the CUDA build to your machine)
python -m pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1 \
    --extra-index-url https://download.pytorch.org/whl/cu116

python -m pip install -r LanguageBind/requirements.txt   # LanguageBind deps
python -m pip install -r requirements.txt                # this project's deps
```

> Use `python -m pip` (not bare `pip`) so packages install into the active `ModelSearch` env
> rather than a system/`--user` Python. Verify with
> `python -c "import torch, chromadb, pandas, gradio; print('ok')"`.

The scripts import `LanguageBind.languagebind`, so point `LANGUAGEBIND_PATH` at the **repo root**
(the directory that *contains* `LanguageBind/`, not the `LanguageBind/` folder itself). Run this
from the repo root:

```bash
export LANGUAGEBIND_PATH="$(pwd)"   # e.g. /path/to/3DModelSearch  (NOT .../3DModelSearch/LanguageBind)
```

If you want to render your own data, you also need [Blender](https://www.blender.org/download/)
(headless) and [`ffmpeg`](https://ffmpeg.org/) (for video). If you're using the precomputed
downloads below, you can skip Blender/ffmpeg. Every script accepts `--help` for its full options.

## Datasets

The pipeline works on any folder of `.glb` / `.gltf` / `.fbx` / `.obj` models. The paper uses:

- **Objaverse** — install the [`objaverse`](https://pypi.org/project/objaverse/) package
  (`python -m pip install objaverse`), then download objects (they land in `~/.objaverse`):
  ```python
  import objaverse
  uids = objaverse.load_uids()
  annotations = objaverse.load_annotations()          # metadata (name / tags / categories / ...)
  objects = objaverse.load_objects(uids=uids[:100])   # {uid: local .glb path}
  ```
  Point the render scripts at the downloaded `.glb` files.
- **ModelNet40** — download from <https://modelnet.cs.princeton.edu/> and convert the meshes to
  `.glb` / `.obj` (the render scripts accept `.obj` directly).

## Precomputed downloads (optional)

Renders, embeddings, and trained checkpoints are hosted on Hugging Face:
**<https://huggingface.co/datasets/srutisrinidhi/3DModelSearch>**. Download them to skip
rendering/embedding/training and extract at the repo root:

```bash
python -m pip install huggingface_hub
REPO=srutisrinidhi/3DModelSearch

# renders + embeddings
huggingface-cli download $REPO embeddings.tar.gz --repo-type dataset --local-dir .
huggingface-cli download $REPO videos.tar.gz     --repo-type dataset --local-dir .
huggingface-cli download $REPO images.tar.gz     --repo-type dataset --local-dir .
tar -xzf embeddings.tar.gz   # -> out/  (per-model .pt + models_data.csv)
tar -xzf videos.tar.gz       # -> data/Videos/
tar -xzf images.tar.gz       # -> data/Images/

# trained model (recommended setup needs only this one)
huggingface-cli download $REPO models/multiview_image_mlp.pt --repo-type dataset --local-dir .
```

| Bundle | Extract to | Contents |
| --- | --- | --- |
| `embeddings.tar.gz` | `out/` | `Embeddings/<uid>_*.pt` + `models_data.csv` |
| `videos.tar.gz` | `data/Videos/` | turntable videos |
| `images.tar.gz` | `data/Images/` | multi-view renders |
| `models/*.pt` | `models/` | trained checkpoints (`multiview_image_mlp.pt` for the recommended setup; the rest are for ablations) |

Source `.glb` models aren't hosted — fetch them from Objaverse (see [Datasets](#datasets)).
All paths in `out/models_data.csv` are relative to the repo root, so run every script from there.
With these in place you can jump straight to [Databases](#databases).

## Pipeline

Starting from a folder of models (`.glb` / `.gltf` / `.fbx` / `.obj`):

```bash
# 1. Render multi-view images and a video per model (data/glbs/ = your folder of source models)
blender --background --python scripts/Blender/render_images.py -- --input data/glbs/ --output renders/images/
blender --background --python scripts/Blender/render_videos.py -- --input data/glbs/ --output renders/videos/

# 2. Encode embeddings + build the mega CSV (out/models_data.csv)
python scripts/generate_embeddings.py \
    --models_dir data/glbs/ --images_dir renders/images/ --videos_dir renders/videos/ --output_dir out/

# 3. Train the multi-view fusion model used by the main database
python scripts/multimodal_models/train_multiview_image_mlp.py \
    --models_csv out/models_data.csv --model_output_path models/multiview_image_mlp.pt

# 4. Build the retrieval database
python scripts/create_model_database.py \
    --models_csv out/models_data.csv --model_path models/multiview_image_mlp.pt \
    --persist_directory databases/model_database

# 5. Retrieve (text / image / video, auto-detected) — prints model paths
python scripts/main.py --query "a red sports car" \
    --models_csv out/models_data.csv --persist_directory databases/model_database

# 6. Or launch the interactive web demo
python scripts/gradio_viewer.py \
    --models_csv out/models_data.csv --persist_directory databases/model_database
```

## Rendering

`scripts/Blender/render_images.py` and `render_videos.py` autodetect the input format and take
either a single file or a folder (batch). Output layout — consumed by the next step by name:

- images → `renders/images/<model>/view<i>.jpg` (`0=front, 1=back, 2=left, 3=right, 4=bottom, 5=top`;
  choose a subset with `--views front,left,top`)
- videos → `renders/videos/<model>.mp4` (plays the model's built-in animation from a fixed view)

## Embeddings

`scripts/generate_embeddings.py` matches each model to its renders **by name** (`uid` = filename
without extension) and writes, under `--output_dir`:

- `Embeddings/<uid>_image.pt` — stacked per-view image embeddings `[N_views, D]`
- `Embeddings/<uid>_video.pt` — one video embedding `[D]`
- `models_data.csv` — the "mega CSV" mapping every model to its render and embedding paths
  (`uid, model_path, image_path, video_path, image_embedding_path, video_embedding_path, caption`)

Re-runs skip models that already have embeddings. Add captions with `--captions_csv uid,caption`.

## Databases

`create_model_database.py` builds the **recommended** database and is the best-performing
configuration in the paper: it pools the image views with the multi-view **MLP** model
(`models/multiview_image_mlp.pt` — the best image model) and simply **averages** that with the
video embedding, into a `multimodal_embeddings` collection. This plain average beats the trained
cross-modal fusion model, so you only need the MLP checkpoint here — **not** a separate
multimodal model.

```bash
python scripts/create_model_database.py \
    --models_csv out/models_data.csv --model_path models/multiview_image_mlp.pt \
    --persist_directory databases/model_database
```

The trained cross-modal fusion model (`create_multimodal_trained_db.py`) is included only as an
ablation and underperforms this average.

**Ablations** — `scripts/database_ablations/` has one builder per paper table row. Each takes
`--models_csv` and `--persist_directory`; model-based ones also take `--model_path`:

| Script | Fusion | Needs `--model_path` |
| --- | --- | --- |
| `create_single_image_db.py` | view 0 only | no |
| `create_all_view_mean_db.py` | mean of 6 views | no |
| `create_all_view_max_db.py` | per-view vectors, max at query | no |
| `create_video_db.py` | video embedding | no |
| `create_multiview_mlp_db.py` | MLP pooling | yes (MLP) |
| `create_multiview_attention_db.py` | attention pooling | yes (attention) |
| `create_multimodal_trained_db.py` | MLP-pool image + cross-modal fusion | yes (fusion + MLP) |
| `create_text_desc_gpt_db.py` | GPT-4o caption → text embedding | no (set `OPENAI_API_KEY`) |
| `create_metadata_db.py` | lexical BM25F index (pickled) | no |

Run an ablation (no trained model needed for this one):

```bash
python scripts/database_ablations/create_all_view_mean_db.py \
    --models_csv out/models_data.csv --persist_directory databases/all_image_mean
```

The GPT-caption ablation calls the OpenAI API, so set your key first:

```bash
export OPENAI_API_KEY=sk-...
python scripts/database_ablations/create_text_desc_gpt_db.py \
    --models_csv out/models_data.csv --persist_directory databases/text_descriptions_gpt
```

`create_metadata_db.py` builds a pickled BM25F lexical index instead of a vector collection.

Train the attention / cross-modal fusion models with the other scripts in
`scripts/multimodal_models/` (same `--models_csv`):

```bash
python scripts/multimodal_models/train_multiview_image_attention.py \
    --models_csv out/models_data.csv --model_output_path models/multiview_image_attention.pt
python scripts/multimodal_models/train_multimodal_model.py \
    --models_csv out/models_data.csv \
    --multiview_model_path models/multiview_image_mlp.pt \
    --model_output_path models/multimodal_retriever_model.pt
```

## Retrieval

- **CLI** — `scripts/main.py --query <text | image path | video path> --models_csv out/models_data.csv
  --persist_directory databases/model_database` prints the top-`--top_k` (default 5) model paths.
- **Web demo** — `scripts/gradio_viewer.py --models_csv out/models_data.csv
  --persist_directory databases/model_database` launches a Gradio page (text / image / video →
  top-5 3D viewers) with a public share link.
