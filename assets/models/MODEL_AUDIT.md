# Model Audit

Audit date: April 19, 2026

This audit was run against the current Blender video generator and scene setup.
Status labels below mean "usable in the current generator pipeline", not "universally good 3D asset".

## Current decision

- Keep:
  - `assets/models/rat/rat_candidate_converted_1k.glb`
  - `assets/models/mouse/mouse_primary.glb`
  - `assets/models/mouse/mouse_candidate_house_1k.glb`
  - `assets/models/cockroach/cockroach_candidate_american_1k.glb`
  - `assets/models/cockroach/cockroach_candidate_classic_1k.glb`
- Exclude for now:
  - `assets/models/rat/rat_primary.glb`
  - `assets/models/rat/rat_variant_animated.glb`
  - `assets/models/rat/rat_variant_black.glb`
  - `assets/models/mouse/mouse_variant_field.glb`
  - `assets/models/mouse/mouse_variant_lab.glb`
  - `assets/models/cockroach/cockroach_primary.glb`
  - `assets/models/cockroach/cockroach_variant_arachnid.glb`
  - `assets/models/cockroach/cockroach_variant_giant.glb`

## Standardized accepted candidates from `assets/models/raw`

- `assets/models/rat/rat_candidate_converted_1k.glb`
  - copied from `assets/models/raw/rat-Converted-format1k.glb`
- `assets/models/mouse/mouse_candidate_house_1k.glb`
  - copied from `assets/models/raw/house_mouse-Converted-format1k.glb`
- `assets/models/cockroach/cockroach_candidate_american_1k.glb`
  - copied from `assets/models/raw/american_cockroach_-_insect-Converted-format1k.glb`
- `assets/models/cockroach/cockroach_candidate_classic_1k.glb`
  - copied from `assets/models/raw/cockroach-Converted-format1k.glb`

## Audit results

### Rat

- `rat_primary.glb`: unusable
  - renders in a bat-like / suspended pose rather than a plausible floor pose
- `rat_variant_animated.glb`: unusable
  - produces oversized occluding geometry that dominates the frame
- `rat_variant_black.glb`: unusable
  - renders as a thin edge-on silhouette with degenerate appearance

### Mouse

- `mouse_primary.glb`: usable
  - renders with a stable silhouette and reasonable size under the current generator
- `mouse_candidate_house_1k.glb`: usable
  - renders with a stable silhouette and no obvious attached base plane
- `mouse_variant_field.glb`: unusable
  - mesh explodes into large broken polygons
- `mouse_variant_lab.glb`: unusable
  - orientation is incorrect and the mouse appears vertically suspended

### Cockroach

- `cockroach_primary.glb`: unusable
  - renders as a thin long-legged insect-like silhouette, not a convincing cockroach
- `cockroach_candidate_american_1k.glb`: usable
  - silhouette is small but plausible and stable under the current generator
- `cockroach_candidate_classic_1k.glb`: usable
  - silhouette is plausible and more top-down than the existing primary asset
- `cockroach_variant_arachnid.glb`: unusable
  - geometry appears fragmented/exploded
- `cockroach_variant_giant.glb`: unusable
  - fails the generator visibility/placement check and does not render a valid sample

## Raw downloads audit

### Rat raw files

- Keep:
  - `assets/models/raw/rat-Converted-format1k.glb`
    - copied into `assets/models/rat/rat_candidate_converted_1k.glb`
- Duplicates of the same usable asset:
  - `assets/models/raw/rat-Converted-format4k.glb`
- Exclude:
  - `assets/models/raw/rat-Converted-format1k (2).glb`
  - `assets/models/raw/rat-Converted-format2k.glb`
  - `assets/models/raw/rat.glb`

### Mouse raw files

- Keep:
  - `assets/models/raw/house_mouse-Converted-format1k.glb`
    - copied into `assets/models/mouse/mouse_candidate_house_1k.glb`
- Duplicates of the same usable asset:
  - `assets/models/raw/house_mouse-Converted-format4k.glb`
  - `assets/models/raw/house_mouse.glb`
- Exclude:
  - `assets/models/raw/22_23 House Mouse Original format.glb`
    - includes an attached base plane
  - `assets/models/raw/rato_domestico__house_mouse-Converted-format1k.glb`
    - includes an attached base plane
  - `assets/models/raw/rato_domestico__house_mouse.glb`
    - includes an attached base plane

### Cockroach raw files

- Keep:
  - `assets/models/raw/american_cockroach_-_insect-Converted-format1k.glb`
    - copied into `assets/models/cockroach/cockroach_candidate_american_1k.glb`
  - `assets/models/raw/cockroach-Converted-format1k.glb`
    - copied into `assets/models/cockroach/cockroach_candidate_classic_1k.glb`
- Duplicates of the same usable asset:
  - `assets/models/raw/american_cockroach_-_insect-Converted-format2k.glb`

## Validation outputs

The one-frame audit renders are stored under:

- `data/generated/model_audit/`

These were generated with the current `generate_synthetic_video_blender.py` logic after the recent visibility and scale fixes.

## New multi-model workflow

The video generator now supports:

- per-class model directories
- per-class exclusion lists
- random selection from the remaining model pool

New arguments:

- `--rat-model-dir`
- `--mouse-model-dir`
- `--cockroach-model-dir`
- `--rat-model-exclude`
- `--mouse-model-exclude`
- `--cockroach-model-exclude`

Example:

```powershell
$blender = Join-Path $env:ProgramFiles "Blender Foundation\Blender 5.1\blender.exe"
& $blender --background --python scripts\generate_synthetic_video_blender.py -- `
  --background-dir data\raw\kitchen_backgrounds `
  --rat-model assets\models\rat\rat_primary.glb `
  --mouse-model assets\models\mouse\mouse_primary.glb `
  --cockroach-model assets\models\cockroach\cockroach_primary.glb `
  --mouse-model-dir assets\models\mouse `
  --mouse-model-exclude mouse_variant_field.glb,mouse_variant_lab.glb `
  --out-dir data\generated\model_pool_smoke `
  --num-clips 1 `
  --duration-seconds 1 `
  --fps 1 `
  --clip-class-sequence mouse `
  --negative-clip-prob 0 `
  --negative-frame-prob 0
```

## Recommended next step

- Replace the rat and cockroach assets with new candidates.
- Re-run the same audit process.
- Once each class has at least one or two visually valid assets, enable directory-based random sampling for training runs.
