# Windows Run Troubleshooting

This file records the Windows-specific issues encountered while setting up, generating data, and training this project, along with the fixes that worked in practice.

## Scope

These notes were collected while running the repository on Windows with:

- Conda environment: `pest-synth`
- GPU: NVIDIA RTX 4070
- Blender installed via `winget`
- PowerShell shell

## 1. Conda Environment Solve Failure

### Symptom

`conda env create -f environment.yml` failed while solving dependencies, with an error around `pytorch-cuda=12.1` and missing `cuda-nvtx`.

### Cause

The environment file used `pytorch-cuda=12.1` but did not include the `nvidia` Conda channel.

### Fix

Add `nvidia` to the `channels:` section of `environment.yml`.

Current working channel order:

```yaml
channels:
  - pytorch
  - nvidia
  - conda-forge
  - defaults
```

## 2. Conda PyTorch Build Failed to Import on Windows

### Symptom

`import torch` failed with a DLL error involving `fbgemm.dll`.

### Cause

The Conda-installed PyTorch/CUDA stack was not stable in this Windows environment.

### Fix

Remove the Conda PyTorch packages and install the official Windows GPU wheels from the PyTorch index.

Commands:

```powershell
conda remove -n pest-synth -y pytorch torchvision torchaudio pytorch-cuda
conda run -n pest-synth python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
conda run -n pest-synth python -m pip install timm
```

## 3. Pillow DLL Import Failure

### Symptom

`from PIL import Image` failed with a DLL load error.

### Cause

Mixed Conda and pip binary packages caused a Windows imaging DLL mismatch.

### Fix

Reinstall `Pillow` with pip inside the project environment.

Command:

```powershell
conda run -n pest-synth python -m pip install --force-reinstall --no-cache-dir pillow
```

## 4. Blender Not Found in PowerShell

### Symptom

Running `blender --version` failed because PowerShell could not find `blender`.

### Cause

Blender was installed, but not available on `PATH`.

### Fix

Use the installed executable path explicitly instead of assuming `blender` is on `PATH`.

Recommended PowerShell pattern:

```powershell
$blender = Join-Path $env:ProgramFiles "Blender Foundation\Blender 5.1\blender.exe"
& $blender --version
```

Install command that worked:

```powershell
winget install -e --id BlenderFoundation.Blender
```

## 5. Places365 Kitchen Background Collection Returned Zero Images

### Symptom

Running `collect_places365_kitchen.py` after downloading Places365 `val` produced:

- `Collected 0 images`

### Cause

The original script expected category names to appear in directory paths. That assumption works better for some extracted directory layouts, but the tested `val` split stores image filenames and labels separately.

### Fix

The script was updated to support the `val` split by:

- reading `categories_places365.txt`
- reading `places365_val.txt`
- mapping category IDs to class names
- selecting only `kitchen` and `restaurant_kitchen`

Working command on Windows:

```powershell
conda run -n pest-synth python scripts\download_places365.py --root data\raw\places365 --split val --small true
conda run -n pest-synth python scripts\collect_places365_kitchen.py --places-root data\raw\places365 --out-dir data\raw\kitchen_backgrounds --mode copy --max-per-class 500 --kitchen-keywords kitchen restaurant_kitchen
```

### Windows note

Use `--mode copy`, not `--mode symlink`, unless your machine is configured for symlink support.

## 6. Blender Failed to Load Background Images

### Symptom

Synthetic generation started, then failed with an error similar to:

`Cannot read 'data\raw\kitchen_backgrounds\kitchen\000008.jpg'`

### Cause

Blender on Windows was being handed a relative path that did not resolve correctly at runtime.

### Fix

`generate_synthetic_blender.py` was updated to resolve all major input paths to absolute paths before loading files.

This includes:

- `background_dir`
- `rat_model`
- `mouse_model`
- `cockroach_model`
- `out_dir`
- selected background image path

## 7. Training Failed on Windows Because DataLoader Could Not Pickle `collate_fn`

### Symptom

Training crashed with an error like:

`AttributeError: Can't pickle local object 'build_loader.<locals>.collate_fn'`

### Cause

Windows multiprocessing requires worker-callable functions to be picklable. A nested `collate_fn` inside `build_loader()` is not picklable in that context.

### Fix

Move the collate function to module scope and pass it into `DataLoader` with `functools.partial`.

This change is already applied in `scripts/train_detr.py`.

## 8. Training Failed Because `DetrImageProcessor.pad()` API Changed

### Symptom

Training crashed with:

`TypeError: DetrImageProcessor.pad() got an unexpected keyword argument 'return_tensors'`

### Cause

The installed `transformers` version uses a newer `DetrImageProcessor.pad()` signature that no longer matches the original batching code.

### Fix

Because the generated smoke dataset uses a fixed image size, the collate function was simplified to:

- `torch.stack(...)` all image tensors directly
- create an all-ones `pixel_mask`

This avoids reliance on the changed `pad()` batch API.

This change is already applied in `scripts/train_detr.py`.

## 9. Hugging Face Cache Warning About Symlinks on Windows

### Symptom

Training printed a warning about Hugging Face cache symlinks on Windows.

### Cause

Windows often disables symlink support unless Developer Mode is enabled or Python is run as administrator.

### Fix

No code change was required.

This warning is safe to ignore unless you specifically want:

- lower disk duplication in the HF cache
- cleaner symlink-based caching behavior

Optional improvement:

- enable Windows Developer Mode

## 10. Model Load Warnings About `num_labels`, `UNEXPECTED`, and `MISMATCH`

### Symptom

Training prints warnings when loading `facebook/detr-resnet-50`, including:

- `num_labels=3` vs pretrained label map length
- `UNEXPECTED`
- `MISMATCH`

### Cause

The pretrained checkpoint was trained on COCO and includes a classifier head for a different label space.

### Fix

No change required for this project.

The detection head is intentionally reinitialized for the 3 custom pest classes:

- `rat`
- `mouse`
- `cockroach`

These warnings are expected during transfer learning.

## 11. Known Working Commands

### Verify environment

```powershell
conda run -n pest-synth python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

### Generate smoke dataset

```powershell
$blender = Join-Path $env:ProgramFiles "Blender Foundation\Blender 5.1\blender.exe"
& $blender --background --python scripts\generate_synthetic_blender.py -- --background-dir data\raw\kitchen_backgrounds --rat-model assets\models\rat\rat_primary.glb --mouse-model assets\models\mouse\mouse_primary.glb --cockroach-model assets\models\cockroach\cockroach_primary.glb --out-dir data\generated\synth_v1_smoke60 --num-images 60
```

### Split dataset

```powershell
conda run -n pest-synth python scripts\split_detection_dataset.py --data-dir data\generated\synth_v1_smoke60 --out-dir data\splits\synth_v1_smoke60 --train-ratio 0.7 --val-ratio 0.15 --test-ratio 0.15 --seed 42
```

### Train smoke model

```powershell
conda run -n pest-synth python scripts\train_detr.py --data-dir data\generated\synth_v1_smoke60 --split-json data\splits\synth_v1_smoke60\split.json --output-dir outputs\detr_smoke60 --epochs 5 --batch-size 4 --lr 1e-4
```

## 12. Recommendation for Future Re-runs

If you rerun the full pipeline on Windows, use this order:

1. Create and verify the Conda environment.
2. Confirm Blender is installed.
3. Download Places365 `val`.
4. Collect kitchen backgrounds with `--mode copy`.
5. Confirm the three `*_primary.glb` model files exist.
6. Generate a small smoke dataset first.
7. Split the dataset.
8. Run smoke training before attempting a larger generation/training job.
