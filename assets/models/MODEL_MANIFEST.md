# Model Manifest

## Current standardized structure

- `assets/models/rat/rat_primary.glb`
- `assets/models/rat/rat_variant_animated.glb`
- `assets/models/rat/rat_variant_black.glb`
- `assets/models/mouse/mouse_primary.glb`
- `assets/models/mouse/mouse_variant_field.glb`
- `assets/models/mouse/mouse_variant_lab.glb`
- `assets/models/cockroach/cockroach_primary.glb`
- `assets/models/cockroach/cockroach_variant_arachnid.glb`
- `assets/models/cockroach/cockroach_variant_giant.glb`

## Recommended run command

```bash
bash scripts/run_generate_synthetic.sh \
  blender \
  data/raw/kitchen_backgrounds \
  assets/models/rat/rat_primary.glb \
  assets/models/mouse/mouse_primary.glb \
  assets/models/cockroach/cockroach_primary.glb \
  data/generated/synth_v1 \
  200
```
