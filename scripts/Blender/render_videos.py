"""Render turntable-free animation videos of 3D models with Blender (headless).

Autodetects the input format (.glb / .gltf / .fbx / .obj), places the camera at a
fixed view, and renders the model's built-in animation timeline to an MP4.
Accepts either a single model file or a folder of models (processed in a batch).

Frames are rendered as PNGs and encoded to MP4 with the system `ffmpeg`
(H.264 / yuv420p), so a working `ffmpeg` on PATH is required.

Usage:
    blender --background --python render_videos.py -- \
        --input PATH --output DIR [options]

    # single file
    blender --background --python render_videos.py -- \
        --input model.fbx --output videos/ --view front --fps 24

    # a folder of models
    blender --background --python render_videos.py -- \
        --input models/ --output videos/

Output layout:
    <output>/<model_name>.mp4
"""

import argparse
import math
import os
import shutil
import subprocess
import sys

import bpy
import mathutils

VIEW_TO_INDEX = {
    "front": 0,
    "back": 1,
    "left": 2,
    "right": 3,
    "bottom": 4,
    "top": 5,
}

SUPPORTED_EXTS = (".glb", ".gltf", ".fbx", ".obj")


# ---------------------------------------------------------------------------
# Scene / model management
# ---------------------------------------------------------------------------

def remove_default_objects():
    """Delete the default startup mesh objects (e.g. the Cube)."""
    for obj in list(bpy.data.objects):
        if obj.type == "MESH":
            bpy.data.objects.remove(obj, do_unlink=True)


def ensure_camera(camera_name):
    """Return the named camera, creating one if the scene does not have it."""
    cam = bpy.data.objects.get(camera_name)
    if cam and cam.type == "CAMERA":
        return cam

    cam_data = bpy.data.cameras.new(camera_name)
    cam = bpy.data.objects.new(camera_name, cam_data)
    bpy.context.scene.collection.objects.link(cam)
    return cam


def clear_scene(obj):
    """Delete a model object and all of its children from the scene."""
    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    for child in obj.children_recursive:
        child.select_set(True)
    bpy.ops.object.delete()


def load_and_rename_model(filepath, new_name="imported_model"):
    """Import a model based on its file extension and rename it. Returns the
    imported top-level object, or None if the format is unsupported."""
    ext = os.path.splitext(filepath)[1].lower()
    if ext in (".glb", ".gltf"):
        bpy.ops.import_scene.gltf(filepath=filepath)
    elif ext == ".obj":
        bpy.ops.import_scene.obj(filepath=filepath)
    elif ext == ".fbx":
        bpy.ops.import_scene.fbx(filepath=filepath)
    else:
        print(f"[skip] Unsupported file format: {ext}")
        return None

    for obj in bpy.context.selected_objects:
        obj.name = new_name

    return bpy.data.objects.get(new_name)


# ---------------------------------------------------------------------------
# Bounds / camera positioning
# ---------------------------------------------------------------------------

def get_total_bounds(obj):
    """Compute world-space bounds for obj and its children.
    Returns: xdiff, ydiff, zdiff, center, min_coord, max_coord."""
    children = list(obj.children_recursive)
    if obj.type != "EMPTY":
        children.append(obj)

    min_coord = mathutils.Vector((float("inf"), float("inf"), float("inf")))
    max_coord = mathutils.Vector((float("-inf"), float("-inf"), float("-inf")))

    depsgraph = bpy.context.evaluated_depsgraph_get()

    for child in children:
        if child.type not in {"MESH", "CURVE", "SURFACE", "META", "FONT"}:
            continue
        child_eval = child.evaluated_get(depsgraph)
        for corner in child_eval.bound_box:
            wc = child_eval.matrix_world @ mathutils.Vector(corner)
            min_coord.x = min(min_coord.x, wc.x)
            min_coord.y = min(min_coord.y, wc.y)
            min_coord.z = min(min_coord.z, wc.z)
            max_coord.x = max(max_coord.x, wc.x)
            max_coord.y = max(max_coord.y, wc.y)
            max_coord.z = max(max_coord.z, wc.z)

    if any(math.isinf(v) for v in min_coord) or any(math.isinf(v) for v in max_coord):
        min_coord = obj.location.copy()
        max_coord = obj.location.copy()

    size = max_coord - min_coord
    center = (max_coord + min_coord) / 2.0
    return size.x, size.y, size.z, center, min_coord, max_coord


def position_camera_for_object(camera, obj, distance_factor=1.8, view="front"):
    """Place the camera to frame obj from the named view."""
    idx = VIEW_TO_INDEX.get(view, 0)
    camera.data.clip_end = 500000
    xdiff, ydiff, zdiff, center, min_c, max_c = get_total_bounds(obj)

    if idx in (0, 1):  # along Y axis
        camera.data.sensor_fit = "HORIZONTAL" if xdiff > zdiff else "VERTICAL"
        xdist = (xdiff / 2) / math.tan(camera.data.angle_x / 2)
        zdist = (zdiff / 2) / math.tan(camera.data.angle_y / 2)
        dist = max(xdist, zdist) + abs(center.y - min_c.y)
    elif idx in (2, 3):  # along X axis
        camera.data.sensor_fit = "HORIZONTAL" if ydiff > zdiff else "VERTICAL"
        ydist = (ydiff / 2) / math.tan(camera.data.angle_y / 2)
        zdist = (zdiff / 2) / math.tan(camera.data.angle_x / 2)
        dist = max(ydist, zdist) + abs(center.x - min_c.x)
    else:  # along Z axis
        camera.data.sensor_fit = "HORIZONTAL" if xdiff > ydiff else "VERTICAL"
        xdist = (xdiff / 2) / math.tan(camera.data.angle_x / 2)
        ydist = (ydiff / 2) / math.tan(camera.data.angle_y / 2)
        dist = max(xdist, ydist) + abs(center.z - min_c.z)

    if math.isnan(dist) or math.isinf(dist) or dist <= 0:
        dist = 2.0
    dist *= float(distance_factor)

    positions = {
        0: (center.x, center.y - dist, center.z),
        1: (center.x, center.y + dist, center.z),
        2: (center.x - dist, center.y, center.z),
        3: (center.x + dist, center.y, center.z),
        4: (center.x, center.y, center.z - dist),
        5: (center.x, center.y, center.z + dist),
    }
    camera.location = mathutils.Vector(positions[idx])
    direction = camera.location - center
    camera.rotation_euler = direction.to_track_quat("Z", "Y").to_euler()


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def render_to_mp4(scene, out_mp4, fps, start_frame, end_frame, resolution,
                  frames_tmp_dir):
    """Render the scene's animation to PNG frames and encode them to MP4 with
    the system ffmpeg (no dependency on Blender's bundled FFmpeg)."""
    scene.render.resolution_x, scene.render.resolution_y = resolution
    scene.render.fps = fps
    scene.frame_start = start_frame
    scene.frame_end = end_frame

    out_mp4 = bpy.path.abspath(out_mp4)
    os.makedirs(os.path.dirname(out_mp4), exist_ok=True)

    frames_tmp_dir = bpy.path.abspath(frames_tmp_dir)
    os.makedirs(frames_tmp_dir, exist_ok=True)

    scene.render.image_settings.file_format = "PNG"
    scene.render.filepath = os.path.join(frames_tmp_dir, "frame_")
    bpy.ops.render.render(animation=True)

    ffmpeg = shutil.which("ffmpeg") or "/usr/bin/ffmpeg"
    if not os.path.isfile(ffmpeg):
        raise RuntimeError("ffmpeg not found. Install it with: sudo apt install ffmpeg")

    pattern = os.path.join(frames_tmp_dir, "frame_%04d.png")
    cmd = [
        ffmpeg, "-y",
        "-framerate", str(fps),
        "-start_number", str(start_frame),
        "-i", pattern,
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        out_mp4,
    ]
    subprocess.run(cmd, check=True)
    print(f"[ok] MP4 saved to: {out_mp4}")

    shutil.rmtree(frames_tmp_dir, ignore_errors=True)


def render_model_as_mp4(obj, output_mp4, camera, view, distance_factor,
                        resolution, fps, start_frame, end_frame, frames_tmp_dir):
    """Position the camera on obj and render its animation to output_mp4."""
    scene = bpy.context.scene
    scene.camera = camera

    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj

    position_camera_for_object(camera, obj, distance_factor=distance_factor, view=view)
    render_to_mp4(
        scene=scene,
        out_mp4=output_mp4,
        fps=fps,
        start_frame=start_frame,
        end_frame=end_frame,
        resolution=resolution,
        frames_tmp_dir=frames_tmp_dir,
    )


# ---------------------------------------------------------------------------
# Input handling + main
# ---------------------------------------------------------------------------

def collect_inputs(input_path):
    """Return a sorted list of model files. A file yields itself; a folder
    yields all supported model files directly inside it."""
    if os.path.isfile(input_path):
        return [input_path]
    if os.path.isdir(input_path):
        return [
            os.path.join(input_path, f)
            for f in sorted(os.listdir(input_path))
            if f.lower().endswith(SUPPORTED_EXTS)
        ]
    raise FileNotFoundError(f"Input path does not exist: {input_path}")


def parse_args(argv):
    parser = argparse.ArgumentParser(
        description="Render animation videos of 3D models with Blender.")
    parser.add_argument("--input", required=True,
                        help="Path to a model file or a folder of models.")
    parser.add_argument("--output", required=True,
                        help="Output folder for the MP4s.")
    parser.add_argument("--view", default="front", choices=list(VIEW_TO_INDEX),
                        help="Fixed camera view. Default: front.")
    parser.add_argument("--resolution", default="1920x1080",
                        help="Render resolution as WIDTHxHEIGHT.")
    parser.add_argument("--distance_factor", type=float, default=1.8,
                        help="Camera distance multiplier.")
    parser.add_argument("--fps", type=int, default=24,
                        help="Frames per second.")
    parser.add_argument("--start_frame", type=int, default=1,
                        help="First animation frame to render.")
    parser.add_argument("--end_frame", type=int, default=150,
                        help="Last animation frame to render.")
    parser.add_argument("--camera", default="Camera",
                        help="Name of the camera object to use.")
    parser.add_argument("--frames_tmp_dir", default="/tmp/blender_frames_tmp",
                        help="Scratch dir for intermediate PNG frames.")
    return parser.parse_args(argv)


def main():
    # Args after the '--' sentinel belong to this script, not to Blender.
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    args = parse_args(argv)

    try:
        width, height = (int(v) for v in args.resolution.lower().split("x"))
    except ValueError:
        raise SystemExit(f"Invalid --resolution '{args.resolution}'; use WIDTHxHEIGHT.")
    resolution = (width, height)

    model_files = collect_inputs(args.input)
    if not model_files:
        raise SystemExit(f"No supported model files found in: {args.input}")

    remove_default_objects()
    camera = ensure_camera(args.camera)

    print(f"[info] {len(model_files)} model(s) to render, view={args.view}, "
          f"frames {args.start_frame}-{args.end_frame} @ {args.fps}fps")

    for filepath in model_files:
        model_name = os.path.splitext(os.path.basename(filepath))[0]
        try:
            obj = load_and_rename_model(filepath, new_name="imported_model")
            if obj is None:
                continue
            out_mp4 = os.path.join(args.output, f"{model_name}.mp4")
            # Per-model scratch dir so batch runs never collide.
            frames_dir = os.path.join(args.frames_tmp_dir, model_name)
            render_model_as_mp4(
                obj, out_mp4, camera, args.view, args.distance_factor,
                resolution, args.fps, args.start_frame, args.end_frame,
                frames_dir,
            )
            clear_scene(obj)
        except Exception as exc:  # keep going on a bad model
            print(f"[error] Failed to render '{filepath}': {exc}")
            continue

    print("[done] Rendering complete.")


if __name__ == "__main__":
    main()
