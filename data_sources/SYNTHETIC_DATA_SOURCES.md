# Synthetic Data Sources

This document tracks the sources for:

- Places365 kitchen backgrounds
- Sketchfab pest 3D models
- Blender synthetic generation reference

## 1) Places365 (kitchen backgrounds)

Primary sources:

- Places project page: http://places2.csail.mit.edu/
- Places devkit (official class list, split files): https://github.com/zhoubolei/places_devkit
- Torchvision Places365 docs (download helper): https://docs.pytorch.org/vision/stable/generated/torchvision.datasets.Places365.html

Notes:

- Use only scene classes containing `kitchen`.
- Respect Places terms: non-commercial research/education usage.

## 2) Sketchfab 3D models (rat / mouse / cockroach)

Curated starter links (verify license again before final dataset release):

- Rat:
  - https://sketchfab.com/3d-models/rat-801d26d47ebf41baa17055287a82da08
  - https://sketchfab.com/3d-models/realistic-rat-3d-model-16df1518d7944a929116113041a75e0b
- Mouse:
  - https://sketchfab.com/3d-models/house-mouse-b2737ba987794081bfb9cebc6aa6b55e
  - https://sketchfab.com/3d-models/rato-domestico-house-mouse-b05c5a51278645d99d40debb8d6ff6bd
- Cockroach:
  - https://sketchfab.com/3d-models/cockroach-202640f19aac4253872b0b1457c5dcb2
  - https://sketchfab.com/3d-models/american-cockroach-insect-0f59da04b8cb4129920263f05d1a235d

Warnings:

- Some Sketchfab assets include `NoAI` restrictions. Do not use those for this pipeline.
- Keep model-level metadata: URL, author, license, download date.

## 3) Blender synthetic generation reference

- Assignment-recommended repo: https://github.com/sean-halpin/synthetic_dataset_creation_blender

Use it mainly as reference for:

- scene randomization
- headless rendering
- annotation export workflow
