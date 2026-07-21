"""Gradio web demo for 3D model retrieval.

Enter a text description, upload a reference image, or upload a video, and the app displays
the top-5 retrieved 3D models. Assumes renders / embeddings / trained model / database already
exist (see the README).

Example:
    export LANGUAGEBIND_PATH="$(pwd)"
    python scripts/gradio_viewer.py \
        --models_csv out/models_data.csv --persist_directory databases/model_database
"""

import argparse
import os
import shutil
import sys
import tempfile
import time

import gradio as gr

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
import db_common  # noqa: E402
import retrieval  # noqa: E402

VIDEO_TMP_DIR = os.path.join(tempfile.gettempdir(), "3dmodelsearch_video_queries")
os.makedirs(VIDEO_TMP_DIR, exist_ok=True)

# Populated in main() before the UI launches.
COLLECTION = None
MODELS_DF = None
TOP_K = 5


def _top_model_paths(results, n=5):
    paths = [model_path for _, model_path in results][:n]
    while len(paths) < n:
        paths.append(None)
    return tuple(paths)


def submit_text_search(input_text):
    print(f"[text] query: {input_text}")
    results = retrieval.search(input_text, models_csv=None, persist_directory=None,
                               modality="text", top_k=TOP_K,
                               collection=COLLECTION, df=MODELS_DF)
    print("[text] results:", results)
    return _top_model_paths(results)


def submit_image_search(image_path):
    print(f"[image] query: {image_path}")
    results = retrieval.search(image_path, models_csv=None, persist_directory=None,
                               modality="image", top_k=TOP_K,
                               collection=COLLECTION, df=MODELS_DF)
    print("[image] results:", results)
    return _top_model_paths(results)


def _coerce_to_path(file_input):
    """Gradio may pass a str path, a dict with 'name'/'path', or a file-like object."""
    if isinstance(file_input, str) and os.path.exists(file_input):
        return file_input
    if isinstance(file_input, dict):
        for key in ("name", "path"):
            p = file_input.get(key)
            if isinstance(p, str) and os.path.exists(p):
                return p
    if hasattr(file_input, "name") and isinstance(file_input.name, str) and os.path.exists(file_input.name):
        return file_input.name
    return None


def submit_video_search(video_input):
    src = _coerce_to_path(video_input)
    if not src:
        print("[video] could not resolve a file path from input:", type(video_input))
        return (None, None, None, None, None)

    ext = os.path.splitext(src)[1] or ".mp4"
    dst = os.path.join(VIDEO_TMP_DIR, f"vid_{int(time.time()*1000)}{ext}")
    shutil.copy(src, dst)
    print(f"[video] saved to: {dst}")

    results = retrieval.search(dst, models_csv=None, persist_directory=None,
                               modality="video", top_k=TOP_K,
                               collection=COLLECTION, df=MODELS_DF)
    print("[video] results:", results)
    return _top_model_paths(results)


def build_ui():
    with gr.Blocks(css=".big-textbox textarea {font-size: 18px !important;}") as demo:
        gr.Markdown("<h1 style='text-align: center;'>3D Model Retrieval</h1>")

        with gr.Row():
            with gr.Column(scale=3):
                gr.Markdown("<h2>Retrieved Top Model</h2>")
                top1_view = gr.Model3D(label="Top-1", height=420)

                gr.Markdown("<h3 style='margin-top: 4px;'>More candidates</h3>")
                with gr.Row():
                    top2_view = gr.Model3D(label="#2", height=210)
                    top3_view = gr.Model3D(label="#3", height=210)
                    top4_view = gr.Model3D(label="#4", height=210)
                    top5_view = gr.Model3D(label="#5", height=210)

            with gr.Column(scale=2):
                gr.Markdown("<h2>Search</h2>")
                text_input = gr.Textbox(placeholder="Describe the model…", lines=1,
                                        label="Text Query", elem_classes=["big-textbox"])
                text_btn = gr.Button("Submit Text", variant="primary")

                gr.Markdown("<hr>")

                image_input = gr.Image(type="filepath",
                                       label="Image Query (upload a reference image)", height=220)
                image_btn = gr.Button("Submit Image")

                gr.Markdown("<hr>")

                video_input = gr.Video(label="Video Query", height=220)
                video_btn = gr.Button("Submit Video")

        outputs = [top1_view, top2_view, top3_view, top4_view, top5_view]
        text_btn.click(fn=submit_text_search, inputs=text_input, outputs=outputs)
        image_btn.click(fn=submit_image_search, inputs=image_input, outputs=outputs)
        video_btn.click(fn=submit_video_search, inputs=video_input, outputs=outputs)

    return demo


def main():
    global COLLECTION, MODELS_DF, TOP_K

    parser = argparse.ArgumentParser(description="3D Model Retrieval demo.")
    parser.add_argument("--models_csv", required=True,
                        help="Mega CSV from generate_embeddings.py (maps uid -> model path).")
    parser.add_argument("--persist_directory", default="databases/model_database",
                        help="ChromaDB directory of the retrieval database.")
    parser.add_argument("--collection_name", default="multimodal_embeddings",
                        help="Name of the ChromaDB collection to query.")
    parser.add_argument("--model_root", default=None,
                        help="Extra directory to allow Gradio to serve model files from "
                             "(defaults to the directories of the models in the CSV).")
    parser.add_argument("--ip", default="127.0.0.1", help="Server bind address.")
    parser.add_argument("--top_k", type=int, default=5, help="Number of models to retrieve.")
    args = parser.parse_args()

    TOP_K = args.top_k
    MODELS_DF = db_common.load_models_df(args.models_csv)
    COLLECTION = retrieval.load_collection(args.persist_directory, args.collection_name)

    # Let Gradio serve the .glb/.obj files from wherever the models actually live.
    allowed = set()
    for p in MODELS_DF.get("model_path", []):
        if isinstance(p, str) and p.strip():
            allowed.add(os.path.dirname(os.path.abspath(p.strip())))
    if args.model_root:
        allowed.add(os.path.abspath(args.model_root))

    demo = build_ui()
    demo.queue()
    demo.launch(share=True, allowed_paths=sorted(allowed))


if __name__ == "__main__":
    main()
