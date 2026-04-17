#!/usr/bin/env python3
"""
Run inside Blender:
  blender --background --python scripts/generate_synthetic_video_blender.py -- \
    --background-dir data/raw/kitchen_backgrounds \
    --rat-model assets/models/rat/rat_primary.glb \
    --mouse-model assets/models/mouse/mouse_primary.glb \
    --cockroach-model assets/models/cockroach/cockroach_primary.glb \
    --out-dir data/generated/video_smoke \
    --num-clips 3 \
    --frames-per-clip 120

Outputs:
- per-clip frame images in <out-dir>/clips/<clip_id>/frames
- per-clip YOLO-format bbox labels in <out-dir>/clips/<clip_id>/labels
- per-clip metadata in <out-dir>/clips/<clip_id>/clip_metadata.json
- clip summary rows in <out-dir>/manifest.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from pathlib import Path

import bpy
import mathutils
from bpy_extras.object_utils import world_to_camera_view


CLASS_TO_ID = {"rat": 0, "mouse": 1, "cockroach": 2}


def blender_args() -> list[str]:
    if "--" not in sys.argv:
        return []
    idx = sys.argv.index("--")
    return sys.argv[idx + 1 :]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--background-dir", type=Path, required=True)
    p.add_argument("--rat-model", type=Path, required=True)
    p.add_argument("--mouse-model", type=Path, required=True)
    p.add_argument("--cockroach-model", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--num-clips", type=int, default=5)
    p.add_argument("--frames-per-clip", type=int, default=180)
    p.add_argument("--fps", type=int, default=12)
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args(blender_args())


def reset_scene() -> None:
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    for block in bpy.data.meshes:
        if block.users == 0:
            bpy.data.meshes.remove(block)


def setup_scene(
    width: int, height: int
) -> tuple[bpy.types.Object, bpy.types.Object, bpy.types.ShaderNodeTexImage]:
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.render.resolution_x = width
    scene.render.resolution_y = height
    scene.render.film_transparent = False
    scene.render.image_settings.file_format = "PNG"
    scene.cycles.samples = 32

    cam_data = bpy.data.cameras.new("Camera")
    cam = bpy.data.objects.new("Camera", cam_data)
    bpy.context.collection.objects.link(cam)
    scene.camera = cam
    cam.location = (0.0, -3.8, 1.3)
    cam.rotation_euler = (math.radians(75), 0.0, 0.0)

    bpy.ops.mesh.primitive_plane_add(size=8, location=(0, 0, 0))
    floor = bpy.context.active_object
    floor.name = "Floor"

    light_data = bpy.data.lights.new(name="KeyLight", type="AREA")
    light_data.energy = 1200
    light = bpy.data.objects.new(name="KeyLight", object_data=light_data)
    bpy.context.collection.objects.link(light)
    light.location = (1.5, -2.0, 3.0)
    light.rotation_euler = (math.radians(-60), math.radians(20), math.radians(15))

    bg_distance = 20.0
    margin = 4.0
    cam_data = cam.data
    view_w = 2.0 * bg_distance * math.tan(cam_data.angle_x / 2.0)
    view_h = 2.0 * bg_distance * math.tan(cam_data.angle_y / 2.0)

    plane_loc = cam.matrix_world @ mathutils.Vector((0.0, 0.0, -bg_distance))
    bpy.ops.mesh.primitive_plane_add(size=1, location=plane_loc)
    bg_plane = bpy.context.active_object
    bg_plane.name = "Backdrop"
    bg_plane.parent = cam
    bg_plane.location = (0.0, 0.0, -bg_distance)
    bg_plane.rotation_euler = (0.0, 0.0, 0.0)
    bg_plane.scale = (0.5 * view_w * margin, 0.5 * view_h * margin, 1.0)

    mat = bpy.data.materials.new(name="BackdropMaterial")
    mat.use_nodes = True
    if hasattr(mat, "shadow_method"):
        mat.shadow_method = "NONE"
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()

    out = nodes.new("ShaderNodeOutputMaterial")
    emission = nodes.new("ShaderNodeEmission")
    tex = nodes.new("ShaderNodeTexImage")
    tex.name = "BGImageTex"
    emission.inputs["Strength"].default_value = 1.0
    links.new(tex.outputs["Color"], emission.inputs["Color"])
    links.new(emission.outputs["Emission"], out.inputs["Surface"])

    bg_plane.data.materials.clear()
    bg_plane.data.materials.append(mat)

    return cam, floor, tex


def import_model(path: Path) -> tuple[bpy.types.Object, list[bpy.types.Object], list[bpy.types.Object]]:
    ext = path.suffix.lower()
    if ext in {".glb", ".gltf"}:
        bpy.ops.import_scene.gltf(filepath=str(path))
    elif ext == ".fbx":
        bpy.ops.import_scene.fbx(filepath=str(path))
    elif ext == ".obj":
        bpy.ops.wm.obj_import(filepath=str(path))
    else:
        raise ValueError(f"Unsupported model format: {path}")

    imported = list(bpy.context.selected_objects)
    if not imported:
        raise RuntimeError(f"No object imported from {path}")

    meshes = [o for o in imported if o.type == "MESH"]
    if not meshes:
        raise RuntimeError(f"Imported model has no mesh objects: {path}")

    ctrl = bpy.data.objects.new(f"CTRL_{path.stem}", None)
    bpy.context.collection.objects.link(ctrl)
    for m in meshes:
        m.parent = ctrl

    recenter_and_ground(ctrl, meshes)
    return ctrl, meshes, imported


def recenter_and_ground(ctrl: bpy.types.Object, meshes: list[bpy.types.Object]) -> None:
    bpy.context.view_layer.update()
    mins = [float("inf"), float("inf"), float("inf")]
    maxs = [float("-inf"), float("-inf"), float("-inf")]
    for obj in meshes:
        for corner in obj.bound_box:
            w = obj.matrix_world @ mathutils.Vector(corner)
            mins[0] = min(mins[0], w.x)
            mins[1] = min(mins[1], w.y)
            mins[2] = min(mins[2], w.z)
            maxs[0] = max(maxs[0], w.x)
            maxs[1] = max(maxs[1], w.y)
            maxs[2] = max(maxs[2], w.z)

    center_x = 0.5 * (mins[0] + maxs[0])
    center_y = 0.5 * (mins[1] + maxs[1])
    min_z = mins[2]
    ctrl.location = (-center_x, -center_y, -min_z)
    bpy.context.view_layer.update()


def _combined_max_dim(meshes: list[bpy.types.Object]) -> float:
    mins = [float("inf"), float("inf"), float("inf")]
    maxs = [float("-inf"), float("-inf"), float("-inf")]
    for obj in meshes:
        for corner in obj.bound_box:
            w = obj.matrix_world @ mathutils.Vector(corner)
            mins[0] = min(mins[0], w.x)
            mins[1] = min(mins[1], w.y)
            mins[2] = min(mins[2], w.z)
            maxs[0] = max(maxs[0], w.x)
            maxs[1] = max(maxs[1], w.y)
            maxs[2] = max(maxs[2], w.z)
    dx = maxs[0] - mins[0]
    dy = maxs[1] - mins[1]
    dz = maxs[2] - mins[2]
    return max(dx, dy, dz, 1e-6)


def normalize_scale(ctrl: bpy.types.Object, meshes: list[bpy.types.Object], target_size: float) -> None:
    bpy.context.view_layer.update()
    max_dim = _combined_max_dim(meshes)
    s = target_size / max_dim
    ctrl.scale = (s, s, s)
    bpy.context.view_layer.update()


def class_scale_range(cls_name: str) -> tuple[float, float]:
    if cls_name == "cockroach":
        return (0.20, 0.35)
    if cls_name == "mouse":
        return (0.35, 0.65)
    return (0.45, 0.80)


def class_speed_range(cls_name: str) -> tuple[float, float]:
    if cls_name == "cockroach":
        return (0.012, 0.020)
    if cls_name == "mouse":
        return (0.009, 0.016)
    return (0.006, 0.012)


def get_bbox_normalized(meshes: list[bpy.types.Object], scene: bpy.types.Scene, cam: bpy.types.Object):
    bpy.context.view_layer.update()
    corners = []
    for obj in meshes:
        corners.extend([obj.matrix_world @ v.co for v in obj.data.vertices])
    points = [world_to_camera_view(scene, cam, c) for c in corners]

    xs = [p.x for p in points]
    ys = [p.y for p in points]

    x_min, x_max = max(min(xs), 0.0), min(max(xs), 1.0)
    y_min, y_max = max(min(ys), 0.0), min(max(ys), 1.0)
    if x_max <= x_min or y_max <= y_min:
        return None

    x_center = (x_min + x_max) / 2.0
    y_center = 1.0 - (y_min + y_max) / 2.0
    w = x_max - x_min
    h = y_max - y_min
    return x_center, y_center, w, h


def list_backgrounds(path: Path) -> list[Path]:
    exts = {".jpg", ".jpeg", ".png"}
    return [p for p in path.rglob("*") if p.suffix.lower() in exts]


def sample_initial_state(
    ctrl: bpy.types.Object,
    meshes: list[bpy.types.Object],
    cls_name: str,
    scene: bpy.types.Scene,
    cam: bpy.types.Object,
) -> dict[str, float]:
    scale_min, scale_max = class_scale_range(cls_name)
    for _ in range(30):
        target_size = random.uniform(scale_min, scale_max)
        normalize_scale(ctrl, meshes, target_size=target_size)
        x = random.uniform(-0.9, 0.9)
        y = random.uniform(-0.4, 0.8)
        heading = random.uniform(0.0, math.pi * 2.0)
        ctrl.location = (x, y, 0.02)
        ctrl.rotation_euler = (0.0, 0.0, heading)
        bbox = get_bbox_normalized(meshes, scene, cam)
        if bbox is not None and (bbox[2] * bbox[3]) >= 0.01:
            speed_min, speed_max = class_speed_range(cls_name)
            return {
                "x": x,
                "y": y,
                "heading": heading,
                "speed": random.uniform(speed_min, speed_max),
                "target_size": target_size,
            }
    raise RuntimeError(f"Could not place visible starting pose for class {cls_name}")


def clamp_with_bounce(value: float, lo: float, hi: float, heading: float, axis: str) -> tuple[float, float]:
    if lo <= value <= hi:
        return value, heading
    if value < lo:
        value = lo + (lo - value)
    elif value > hi:
        value = hi - (value - hi)

    if axis == "x":
        heading = math.pi - heading
    else:
        heading = -heading
    return value, heading


def step_motion(state: dict[str, float], cls_name: str) -> dict[str, float]:
    heading = state["heading"] + random.uniform(-0.18, 0.18)
    speed = state["speed"]
    if random.random() < 0.08:
        speed *= random.uniform(0.3, 0.8)
    else:
        speed *= random.uniform(0.95, 1.05)

    speed_min, speed_max = class_speed_range(cls_name)
    speed = max(speed_min, min(speed, speed_max))
    x = state["x"] + math.cos(heading) * speed
    y = state["y"] + math.sin(heading) * speed
    x, heading = clamp_with_bounce(x, -0.95, 0.95, heading, "x")
    y, heading = clamp_with_bounce(y, -0.45, 0.85, heading, "y")
    return {
        **state,
        "x": x,
        "y": y,
        "heading": heading,
        "speed": speed,
    }


def main() -> None:
    args = parse_args()
    args.background_dir = args.background_dir.resolve()
    args.rat_model = args.rat_model.resolve()
    args.mouse_model = args.mouse_model.resolve()
    args.cockroach_model = args.cockroach_model.resolve()
    args.out_dir = args.out_dir.resolve()
    random.seed(args.seed)

    backgrounds = list_backgrounds(args.background_dir)
    if not backgrounds:
        raise RuntimeError(f"No backgrounds found in {args.background_dir}")

    models = {
        "rat": args.rat_model,
        "mouse": args.mouse_model,
        "cockroach": args.cockroach_model,
    }
    for cls_name, model_path in models.items():
        if not model_path.exists():
            raise FileNotFoundError(f"{cls_name} model missing: {model_path}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.out_dir / "manifest.csv"

    reset_scene()
    cam, _, bg_tex_node = setup_scene(args.width, args.height)
    scene = bpy.context.scene
    scene.render.fps = args.fps

    with manifest_path.open("w", newline="", encoding="utf-8") as manifest_file:
        writer = csv.writer(manifest_file)
        writer.writerow(
            [
                "clip_id",
                "class_name",
                "model_path",
                "background_path",
                "frames_per_clip",
                "fps",
                "duration_seconds",
                "frames_dir",
                "labels_dir",
                "metadata_path",
            ]
        )

        for clip_idx in range(args.num_clips):
            clip_id = f"clip_{clip_idx:06d}"
            clip_dir = args.out_dir / "clips" / clip_id
            frames_dir = clip_dir / "frames"
            labels_dir = clip_dir / "labels"
            frames_dir.mkdir(parents=True, exist_ok=True)
            labels_dir.mkdir(parents=True, exist_ok=True)

            cls_name = random.choice(list(models.keys()))
            model_path = models[cls_name]
            background_path = random.choice(backgrounds).resolve()
            bg_tex_node.image = bpy.data.images.load(str(background_path), check_existing=True)

            ctrl, meshes, imported_objs = import_model(model_path)
            state = sample_initial_state(ctrl, meshes, cls_name, scene, cam)

            metadata = {
                "clip_id": clip_id,
                "class_name": cls_name,
                "class_id": CLASS_TO_ID[cls_name],
                "model_path": str(model_path),
                "background_path": str(background_path),
                "frames_per_clip": args.frames_per_clip,
                "fps": args.fps,
                "duration_seconds": args.frames_per_clip / args.fps,
                "seed": args.seed + clip_idx,
                "width": args.width,
                "height": args.height,
                "initial_state": state.copy(),
            }

            for frame_idx in range(args.frames_per_clip):
                if frame_idx > 0:
                    state = step_motion(state, cls_name)

                ctrl.location = (state["x"], state["y"], 0.02)
                ctrl.rotation_euler = (0.0, 0.0, state["heading"])
                bbox = get_bbox_normalized(meshes, scene, cam)

                frame_name = f"frame_{frame_idx:06d}.png"
                label_name = f"frame_{frame_idx:06d}.txt"
                frame_path = frames_dir / frame_name
                label_path = labels_dir / label_name

                scene.render.filepath = str(frame_path)
                bpy.ops.render.render(write_still=True)

                with label_path.open("w", encoding="utf-8") as label_file:
                    if bbox is not None:
                        label_file.write(
                            f"{CLASS_TO_ID[cls_name]} {bbox[0]:.6f} {bbox[1]:.6f} {bbox[2]:.6f} {bbox[3]:.6f}\n"
                        )

                print(
                    f"[{clip_idx + 1}/{args.num_clips}] {clip_id} "
                    f"frame {frame_idx + 1}/{args.frames_per_clip} {cls_name}"
                )

            metadata_path = clip_dir / "clip_metadata.json"
            metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
            writer.writerow(
                [
                    clip_id,
                    cls_name,
                    str(model_path),
                    str(background_path),
                    args.frames_per_clip,
                    args.fps,
                    args.frames_per_clip / args.fps,
                    str(frames_dir),
                    str(labels_dir),
                    str(metadata_path),
                ]
            )

            for obj in imported_objs:
                if obj.name in bpy.data.objects:
                    bpy.data.objects.remove(obj, do_unlink=True)
            if ctrl.name in bpy.data.objects:
                bpy.data.objects.remove(ctrl, do_unlink=True)

    print(f"Done. Video-style outputs in: {args.out_dir}")


if __name__ == "__main__":
    main()
