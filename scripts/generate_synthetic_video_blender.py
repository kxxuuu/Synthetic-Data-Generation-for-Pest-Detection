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
    --duration-seconds 10 \
    --fps 12

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
    p.add_argument("--rat-model-dir", type=Path, default=None)
    p.add_argument("--mouse-model-dir", type=Path, default=None)
    p.add_argument("--cockroach-model-dir", type=Path, default=None)
    p.add_argument("--rat-model-exclude", default=None)
    p.add_argument("--mouse-model-exclude", default=None)
    p.add_argument("--cockroach-model-exclude", default=None)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--num-clips", type=int, default=5)
    p.add_argument("--start-clip", type=int, default=0)
    p.add_argument("--frames-per-clip", type=int, default=180)
    p.add_argument("--duration-seconds", type=float, default=None)
    p.add_argument("--fps", type=int, default=12)
    p.add_argument("--samples", type=int, default=32)
    p.add_argument("--min-bbox-area", type=float, default=0.01)
    p.add_argument("--negative-clip-prob", type=float, default=0.15)
    p.add_argument("--negative-frame-prob", type=float, default=0.10)
    p.add_argument("--pause-prob", type=float, default=0.10)
    p.add_argument(
        "--clip-class-sequence",
        default=None,
        help="Optional comma-separated class sequence, e.g. rat,mouse,cockroach",
    )
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
    width: int, height: int, samples: int
) -> tuple[bpy.types.Object, bpy.types.Object, bpy.types.ShaderNodeTexImage]:
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.render.resolution_x = width
    scene.render.resolution_y = height
    scene.render.film_transparent = False
    scene.render.image_settings.file_format = "PNG"
    scene.cycles.samples = samples

    cam_data = bpy.data.cameras.new("Camera")
    cam = bpy.data.objects.new("Camera", cam_data)
    bpy.context.collection.objects.link(cam)
    scene.camera = cam
    cam.location = (0.0, -3.8, 1.3)
    cam.rotation_euler = (math.radians(75), 0.0, 0.0)

    bpy.ops.mesh.primitive_plane_add(size=8, location=(0, 0, 0))
    floor = bpy.context.active_object
    floor.name = "Floor"
    # Keep a hidden floor guide for placement only. Rendering it in front of
    # the 2D kitchen backdrop produces a large opaque slab that obscures most
    # of the scene.
    floor.hide_render = True
    floor.hide_viewport = True

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


def import_model(
    path: Path,
) -> tuple[bpy.types.Object, bpy.types.Object, list[bpy.types.Object], list[bpy.types.Object]]:
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
    pose = bpy.data.objects.new(f"POSE_{path.stem}", None)
    bpy.context.collection.objects.link(pose)
    pose.parent = ctrl
    for m in meshes:
        m.parent = pose

    recenter_and_ground(ctrl, meshes)
    return ctrl, pose, meshes, imported


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


def class_pose_rotation(cls_name: str) -> tuple[float, float, float]:
    # The rat asset imports standing upright in this scene setup. Lay it down
    # so its long axis stays on the floor plane before applying clip motion.
    if cls_name == "rat":
        return (0.0, math.radians(90), 0.0)
    return (0.0, 0.0, 0.0)


def apply_class_pose(pose: bpy.types.Object, cls_name: str) -> None:
    pose.rotation_euler = class_pose_rotation(cls_name)
    bpy.context.view_layer.update()


def class_scale_range(cls_name: str) -> tuple[float, float]:
    if cls_name == "cockroach":
        return (0.08, 0.14)
    if cls_name == "mouse":
        return (0.10, 0.18)
    return (0.12, 0.22)


def class_bbox_area_range(cls_name: str) -> tuple[float, float]:
    if cls_name == "cockroach":
        return (0.0030, 0.0120)
    if cls_name == "mouse":
        return (0.0040, 0.0150)
    return (0.0060, 0.0200)


def class_speed_range(cls_name: str) -> tuple[float, float]:
    if cls_name == "cockroach":
        return (0.012, 0.020)
    if cls_name == "mouse":
        return (0.009, 0.016)
    return (0.006, 0.012)


def resolve_frames_per_clip(args: argparse.Namespace) -> int:
    if args.duration_seconds is not None:
        return max(1, int(round(args.duration_seconds * args.fps)))
    return args.frames_per_clip


def validate_probability(name: str, value: float) -> None:
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be in [0, 1], got {value}")


def parse_class_sequence(raw: str | None, valid_classes: set[str]) -> list[str] | None:
    if raw is None:
        return None
    items = [part.strip() for part in raw.split(",") if part.strip()]
    if not items:
        return None
    invalid = [item for item in items if item not in valid_classes]
    if invalid:
        raise ValueError(
            f"--clip-class-sequence contains invalid classes: {invalid}. "
            f"Valid classes: {sorted(valid_classes)}"
        )
    return items


def parse_name_list(raw: str | None) -> set[str]:
    if raw is None:
        return set()
    return {part.strip() for part in raw.split(",") if part.strip()}


def model_file_candidates(model_dir: Path) -> list[Path]:
    exts = {".glb", ".gltf", ".fbx", ".obj"}
    return sorted(
        [p for p in model_dir.iterdir() if p.is_file() and p.suffix.lower() in exts]
    )


def resolve_model_pool(
    single_model: Path,
    model_dir: Path | None,
    exclude_names: set[str],
    cls_name: str,
) -> list[Path]:
    if model_dir is not None:
        model_dir = model_dir.resolve()
        if not model_dir.exists():
            raise FileNotFoundError(f"{cls_name} model dir missing: {model_dir}")
        pool = [p.resolve() for p in model_file_candidates(model_dir) if p.name not in exclude_names]
        if not pool:
            raise RuntimeError(
                f"No usable {cls_name} models remain in {model_dir} after exclusions: "
                f"{sorted(exclude_names)}"
            )
        return pool

    single_model = single_model.resolve()
    if not single_model.exists():
        raise FileNotFoundError(f"{cls_name} model missing: {single_model}")
    return [single_model]


def set_objects_visibility(objects: list[bpy.types.Object], visible: bool) -> None:
    for obj in objects:
        obj.hide_render = not visible
        obj.hide_viewport = not visible


def get_bbox_normalized(meshes: list[bpy.types.Object], scene: bpy.types.Scene, cam: bpy.types.Object):
    bpy.context.view_layer.update()
    corners = []
    for obj in meshes:
        corners.extend([obj.matrix_world @ v.co for v in obj.data.vertices])
    points = [world_to_camera_view(scene, cam, c) for c in corners]
    visible_points = [
        p for p in points if p.z > 0.0 and 0.0 <= p.x <= 1.0 and 0.0 <= p.y <= 1.0
    ]
    min_visible_points = max(6, int(len(points) * 0.08))
    if len(visible_points) < min_visible_points:
        return None

    xs = [p.x for p in visible_points]
    ys = [p.y for p in visible_points]

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
    rng: random.Random,
    min_bbox_area: float,
) -> dict[str, float]:
    scale_min, scale_max = class_scale_range(cls_name)
    area_min, area_max = class_bbox_area_range(cls_name)
    area_min = max(area_min, min_bbox_area)
    for _ in range(50):
        initial_target_size = rng.uniform(scale_min, scale_max)
        normalize_scale(ctrl, meshes, target_size=initial_target_size)
        x = rng.uniform(-0.9, 0.9)
        y = rng.uniform(-0.4, 0.8)
        heading = rng.uniform(0.0, math.pi * 2.0)
        ctrl.location = (x, y, 0.02)
        ctrl.rotation_euler = (0.0, 0.0, heading)
        bbox = get_bbox_normalized(meshes, scene, cam)
        if bbox is None:
            continue

        current_area = bbox[2] * bbox[3]
        if current_area <= 0:
            continue

        target_area = rng.uniform(area_min, area_max)
        scale_adjust = math.sqrt(target_area / current_area)
        ctrl.scale = tuple(v * scale_adjust for v in ctrl.scale)
        bpy.context.view_layer.update()
        bbox = get_bbox_normalized(meshes, scene, cam)
        if bbox is not None and area_min <= (bbox[2] * bbox[3]) <= area_max:
            speed_min, speed_max = class_speed_range(cls_name)
            return {
                "x": x,
                "y": y,
                "heading": heading,
                "speed": rng.uniform(speed_min, speed_max),
                "target_size": ctrl.scale[0],
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


def step_motion(
    state: dict[str, float],
    cls_name: str,
    rng: random.Random,
    pause_prob: float,
) -> dict[str, float]:
    heading = state["heading"] + rng.uniform(-0.12, 0.12)
    speed = state["speed"]
    if rng.random() < pause_prob:
        speed *= rng.uniform(0.15, 0.55)
    else:
        speed *= rng.uniform(0.96, 1.04)

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


def sample_negative_frame_indices(
    frame_count: int,
    negative_frame_prob: float,
    rng: random.Random,
) -> set[int]:
    if frame_count <= 1 or negative_frame_prob <= 0.0:
        return set()
    negative_count = int(round(frame_count * negative_frame_prob))
    negative_count = max(0, min(negative_count, frame_count - 1))
    if negative_count == 0:
        return set()
    return set(rng.sample(range(frame_count), negative_count))


def main() -> None:
    args = parse_args()
    args.background_dir = args.background_dir.resolve()
    args.rat_model = args.rat_model.resolve()
    args.mouse_model = args.mouse_model.resolve()
    args.cockroach_model = args.cockroach_model.resolve()
    if args.rat_model_dir is not None:
        args.rat_model_dir = args.rat_model_dir.resolve()
    if args.mouse_model_dir is not None:
        args.mouse_model_dir = args.mouse_model_dir.resolve()
    if args.cockroach_model_dir is not None:
        args.cockroach_model_dir = args.cockroach_model_dir.resolve()
    args.out_dir = args.out_dir.resolve()
    validate_probability("--negative-clip-prob", args.negative_clip_prob)
    validate_probability("--negative-frame-prob", args.negative_frame_prob)
    validate_probability("--pause-prob", args.pause_prob)
    if args.start_clip < 0:
        raise ValueError("--start-clip must be >= 0")
    if args.num_clips <= 0:
        raise ValueError("--num-clips must be >= 1")
    if args.duration_seconds is not None and args.duration_seconds <= 0:
        raise ValueError("--duration-seconds must be > 0 when provided")
    if args.min_bbox_area <= 0:
        raise ValueError("--min-bbox-area must be > 0")

    frame_count = resolve_frames_per_clip(args)

    backgrounds = list_backgrounds(args.background_dir)
    if not backgrounds:
        raise RuntimeError(f"No backgrounds found in {args.background_dir}")

    model_pools = {
        "rat": resolve_model_pool(
            args.rat_model,
            args.rat_model_dir,
            parse_name_list(args.rat_model_exclude),
            "rat",
        ),
        "mouse": resolve_model_pool(
            args.mouse_model,
            args.mouse_model_dir,
            parse_name_list(args.mouse_model_exclude),
            "mouse",
        ),
        "cockroach": resolve_model_pool(
            args.cockroach_model,
            args.cockroach_model_dir,
            parse_name_list(args.cockroach_model_exclude),
            "cockroach",
        ),
    }
    clip_class_sequence = parse_class_sequence(args.clip_class_sequence, set(model_pools.keys()))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.out_dir / "manifest.csv"

    reset_scene()
    if args.samples <= 0:
        raise ValueError("--samples must be >= 1")

    cam, _, bg_tex_node = setup_scene(args.width, args.height, args.samples)
    scene = bpy.context.scene
    scene.render.fps = args.fps

    with manifest_path.open("w", newline="", encoding="utf-8") as manifest_file:
        writer = csv.writer(manifest_file)
        writer.writerow(
            [
                "clip_id",
                "class_name",
                "class_id",
                "clip_seed",
                "is_negative_clip",
                "negative_frame_count",
                "visibility_filtered_frame_count",
                "model_pool_size",
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
            clip_number = args.start_clip + clip_idx
            clip_id = f"clip_{clip_number:06d}"
            clip_seed = args.seed + clip_number
            clip_rng = random.Random(clip_seed)
            clip_dir = args.out_dir / "clips" / clip_id
            frames_dir = clip_dir / "frames"
            labels_dir = clip_dir / "labels"
            frames_dir.mkdir(parents=True, exist_ok=True)
            labels_dir.mkdir(parents=True, exist_ok=True)

            background_path = clip_rng.choice(backgrounds).resolve()
            bg_tex_node.image = bpy.data.images.load(str(background_path), check_existing=True)
            is_negative_clip = clip_rng.random() < args.negative_clip_prob
            chosen_class = (
                clip_class_sequence[clip_idx % len(clip_class_sequence)]
                if clip_class_sequence is not None
                else clip_rng.choice(list(model_pools.keys()))
            )
            cls_name = "none" if is_negative_clip else chosen_class
            class_id = -1 if is_negative_clip else CLASS_TO_ID[cls_name]
            model_pool = [] if is_negative_clip else model_pools[cls_name]
            model_path = None if is_negative_clip else clip_rng.choice(model_pool)
            ctrl = None
            pose = None
            meshes: list[bpy.types.Object] = []
            imported_objs: list[bpy.types.Object] = []
            state = None
            visibility_filtered_frame_count = 0
            negative_frame_indices = set(range(frame_count)) if is_negative_clip else sample_negative_frame_indices(
                frame_count,
                args.negative_frame_prob,
                clip_rng,
            )
            if not is_negative_clip and model_path is not None:
                ctrl, pose, meshes, imported_objs = import_model(model_path)
                apply_class_pose(pose, cls_name)
                recenter_and_ground(ctrl, meshes)
                state = sample_initial_state(
                    ctrl,
                    meshes,
                    cls_name,
                    scene,
                    cam,
                    clip_rng,
                    args.min_bbox_area,
                )

            metadata = {
                "clip_id": clip_id,
                "class_name": cls_name,
                "class_id": class_id,
                "clip_seed": clip_seed,
                "is_negative_clip": is_negative_clip,
                "negative_frame_count": len(negative_frame_indices),
                "visibility_filtered_frame_count": 0,
                "model_pool_size": len(model_pool),
                "negative_frame_probability": args.negative_frame_prob,
                "model_path": str(model_path) if model_path is not None else None,
                "background_path": str(background_path),
                "frames_per_clip": frame_count,
                "fps": args.fps,
                "duration_seconds": frame_count / args.fps,
                "seed": clip_seed,
                "width": args.width,
                "height": args.height,
                "samples": args.samples,
                "min_bbox_area": args.min_bbox_area,
                "initial_state": state.copy() if state is not None else None,
            }

            for frame_idx in range(frame_count):
                if state is not None and frame_idx > 0:
                    state = step_motion(state, cls_name, clip_rng, args.pause_prob)

                frame_is_negative = is_negative_clip or frame_idx in negative_frame_indices
                bbox = None
                if state is not None and ctrl is not None:
                    ctrl.location = (state["x"], state["y"], 0.02)
                    ctrl.rotation_euler = (0.0, 0.0, state["heading"])
                    set_objects_visibility(imported_objs, not frame_is_negative)
                    if not frame_is_negative:
                        candidate_bbox = get_bbox_normalized(meshes, scene, cam)
                        if candidate_bbox is not None and (candidate_bbox[2] * candidate_bbox[3]) >= args.min_bbox_area:
                            bbox = candidate_bbox
                        else:
                            frame_is_negative = True
                            visibility_filtered_frame_count += 1
                            set_objects_visibility(imported_objs, False)

                frame_name = f"frame_{frame_idx:06d}.png"
                label_name = f"frame_{frame_idx:06d}.txt"
                frame_path = frames_dir / frame_name
                label_path = labels_dir / label_name

                scene.render.filepath = str(frame_path)
                bpy.ops.render.render(write_still=True)

                with label_path.open("w", encoding="utf-8") as label_file:
                    if bbox is not None:
                        label_file.write(f"{class_id} {bbox[0]:.6f} {bbox[1]:.6f} {bbox[2]:.6f} {bbox[3]:.6f}\n")

                print(
                    f"[{clip_idx + 1}/{args.num_clips}] {clip_id} "
                    f"frame {frame_idx + 1}/{frame_count} {cls_name}"
                    f"{' NEG' if frame_is_negative else ''}"
                )

            metadata_path = clip_dir / "clip_metadata.json"
            metadata["visibility_filtered_frame_count"] = visibility_filtered_frame_count
            metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
            writer.writerow(
                [
                    clip_id,
                    cls_name,
                    class_id,
                    clip_seed,
                    is_negative_clip,
                    len(negative_frame_indices),
                    visibility_filtered_frame_count,
                    len(model_pool),
                    str(model_path) if model_path is not None else "",
                    str(background_path),
                    frame_count,
                    args.fps,
                    frame_count / args.fps,
                    str(frames_dir),
                    str(labels_dir),
                    str(metadata_path),
                ]
            )

            if imported_objs:
                set_objects_visibility(imported_objs, True)
                for obj in imported_objs:
                    if obj.name in bpy.data.objects:
                        bpy.data.objects.remove(obj, do_unlink=True)
            if pose is not None and pose.name in bpy.data.objects:
                bpy.data.objects.remove(pose, do_unlink=True)
            if ctrl is not None and ctrl.name in bpy.data.objects:
                bpy.data.objects.remove(ctrl, do_unlink=True)

    print(f"Done. Video-style outputs in: {args.out_dir}")


if __name__ == "__main__":
    main()
