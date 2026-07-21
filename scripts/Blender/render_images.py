"""Render orthographic axis views of 3D models with Blender (headless).

Autodetects the input format (.glb / .gltf / .fbx / .obj) and renders one image
per requested view. Accepts either a single model file or a folder of models
(processed in a batch).

Usage:
    blender --background --python render_images.py -- \
        --input PATH --output DIR [options]

    # single file, all six views
    blender --background --python render_images.py -- \
        --input model.glb --output renders/

    # a folder of models, only the front/left/top views
    blender --background --python render_images.py -- \
        --input models/ --output renders/ --views front,left,top

Output layout:
    <output>/<model_name>/view<index>.<ext>   (index 0..5, one per view)
"""

import argparse
import math
import os
import sys

import bpy
import mathutils

# Named views -> render index. Matches the convention used across the pipeline:
#   0: -Y (front)   1: +Y (back)
#   2: -X (left)    3: +X (right)
#   4: -Z (bottom)  5: +Z (top)
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
    """Delete the default startup mesh objects (e.g. the Cube) so they never
    appear in a render. Cameras and lights are left untouched."""
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
    children = [child for child in obj.children_recursive]
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

    # fallback when nothing renderable was found
    if any(math.isinf(v) for v in min_coord) or any(math.isinf(v) for v in max_coord):
        min_coord = obj.location.copy()
        max_coord = obj.location.copy()

    size = max_coord - min_coord
    center = (max_coord + min_coord) / 2.0
    return size.x, size.y, size.z, center, min_coord, max_coord


def position_camera_for_object(camera, obj, distance_factor=2.0, render_index=0):
    """Place the camera to frame obj from the given axis view.
    render_index: 0=-Y, 1=+Y, 2=-X, 3=+X, 4=-Z, 5=+Z."""
    camera.data.clip_end = 500000

    xdiff, ydiff, zdiff, center, min_coord, max_coord = get_total_bounds(obj)

    if any(math.isnan(v) or math.isinf(v) for v in center):
        center = obj.location.copy()

    if render_index in (0, 1):  # looking along Y
        camera.data.sensor_fit = "HORIZONTAL" if xdiff > zdiff else "VERTICAL"
        xFOV = camera.data.angle_x
        zFOV = camera.data.angle_y
        xdistance = (xdiff / 2) / (math.tan(xFOV / 2) if xFOV else 1e-6)
        zdistance = (zdiff / 2) / (math.tan(zFOV / 2) if zFOV else 1e-6)
        distance = max(xdistance, zdistance) + abs(center.y - min_coord.y)

    elif render_index in (2, 3):  # looking along X
        camera.data.sensor_fit = "HORIZONTAL" if ydiff > zdiff else "VERTICAL"
        yFOV = camera.data.angle_y
        zFOV = camera.data.angle_x
        ydistance = (ydiff / 2) / (math.tan(yFOV / 2) if yFOV else 1e-6)
        zdistance = (zdiff / 2) / (math.tan(zFOV / 2) if zFOV else 1e-6)
        distance = max(ydistance, zdistance) + abs(center.x - min_coord.x)

    else:  # looking along Z
        camera.data.sensor_fit = "HORIZONTAL" if xdiff > ydiff else "VERTICAL"
        xFOV = camera.data.angle_x
        yFOV = camera.data.angle_y
        xdistance = (xdiff / 2) / (math.tan(xFOV / 2) if xFOV else 1e-6)
        ydistance = (ydiff / 2) / (math.tan(yFOV / 2) if yFOV else 1e-6)
        distance = max(xdistance, ydistance) + abs(center.z - min_coord.z)

    if math.isnan(distance) or math.isinf(distance) or distance <= 0:
        distance = 2.0
    distance *= float(distance_factor)

    positions = {
        0: (center.x, center.y - distance, center.z),
        1: (center.x, center.y + distance, center.z),
        2: (center.x - distance, center.y, center.z),
        3: (center.x + distance, center.y, center.z),
        4: (center.x, center.y, center.z - distance),
        5: (center.x, center.y, center.z + distance),
    }
    camera.location = mathutils.Vector(positions[render_index])
    direction = camera.location - center
    camera.rotation_euler = direction.to_track_quat("Z", "Y").to_euler()


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def render_image(output_path, camera, resolution):
    """Render a single still from the given camera to output_path."""
    scene = bpy.context.scene
    scene.render.resolution_x = resolution[0]
    scene.render.resolution_y = resolution[1]
    scene.camera = camera

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    scene.render.filepath = output_path
    bpy.ops.render.render(write_still=True)
    print(f"[ok] Rendered {output_path}")


def render_views(output_folder, camera, obj, view_indices, distance_factor,
                 resolution, ext):
    """Render each requested view of obj into output_folder/view<index>.<ext>."""
    os.makedirs(output_folder, exist_ok=True)
    for i in view_indices:
        position_camera_for_object(camera, obj, distance_factor, i)
        out_path = os.path.join(output_folder, f"view{i}.{ext}")
        render_image(out_path, camera, resolution)


# ---------------------------------------------------------------------------
# Input handling + main
# ---------------------------------------------------------------------------

def collect_inputs(input_path):
    """Return a sorted list of model files. A file yields itself; a folder
    yields all supported model files directly inside it."""
    if os.path.isfile(input_path):
        return [input_path]
    if os.path.isdir(input_path):
        files = [
            os.path.join(input_path, f)
            for f in sorted(os.listdir(input_path))
            if f.lower().endswith(SUPPORTED_EXTS)
        ]
        return files
    raise FileNotFoundError(f"Input path does not exist: {input_path}")


def parse_args(argv):
    parser = argparse.ArgumentParser(
        description="Render axis views of 3D models with Blender.")
    parser.add_argument("--input", required=True,
                        help="Path to a model file or a folder of models.")
    parser.add_argument("--output", required=True,
                        help="Output folder for renders.")
    parser.add_argument(
        "--views", default="front,back,left,right,bottom,top",
        help="Comma-separated views to render "
             "(front,back,left,right,bottom,top). Default: all six.")
    parser.add_argument("--resolution", default="1920x1080",
                        help="Render resolution as WIDTHxHEIGHT.")
    parser.add_argument("--distance_factor", type=float, default=2.0,
                        help="Camera distance multiplier.")
    parser.add_argument("--camera", default="Camera",
                        help="Name of the camera object to use.")
    parser.add_argument("--ext", default="jpg",
                        help="Output image extension (e.g. jpg, png).")
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

    view_indices = []
    for name in (v.strip().lower() for v in args.views.split(",")):
        if not name:
            continue
        if name not in VIEW_TO_INDEX:
            raise SystemExit(
                f"Unknown view '{name}'. Choose from: {', '.join(VIEW_TO_INDEX)}.")
        view_indices.append(VIEW_TO_INDEX[name])

    model_files = collect_inputs(args.input)
    if not model_files:
        raise SystemExit(f"No supported model files found in: {args.input}")

    remove_default_objects()
    camera = ensure_camera(args.camera)

    print(f"[info] {len(model_files)} model(s) to render, "
          f"views={sorted(view_indices)}, resolution={width}x{height}")

    for filepath in model_files:
        model_name = os.path.splitext(os.path.basename(filepath))[0]
        try:
            obj = load_and_rename_model(filepath, new_name="imported_model")
            if obj is None:
                continue
            out_dir = os.path.join(args.output, model_name)
            render_views(out_dir, camera, obj, view_indices,
                         args.distance_factor, resolution, args.ext)
            clear_scene(obj)
        except Exception as exc:  # keep going on a bad model
            print(f"[error] Failed to render '{filepath}': {exc}")
            continue

    print("[done] Rendering complete.")


if __name__ == "__main__":
    main()
