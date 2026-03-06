# GlowUp

You, but on your best day. A local, privacy-first face enhancement app. No cloud uploads, no model training. Uses MediaPipe FaceMesh for landmark detection and OpenCV for image processing.

## Quick Start

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Features

| Effect | What it does |
|---|---|
| Skin Smoothing | Bilateral filter within a feathered face-oval mask |
| Spot & Pore Removal | Difference-of-Gaussians detection + median correction |
| Skin Glow | Highlight-aware screen-blended bloom on skin |
| Eye Whitening | Brightens sclera in LAB color space |
| Eye Elongation | Stretches eyes horizontally for an almond shape |
| Eye Enlargement | Outward warps from eye center with iris protection |
| Smize | Smile with your eyes — subtle lower lid push |
| Eyelash Darkening | Tapered lash line with perspective-aware balancing |
| Lower Lash Line | Subtle lower lid darkening |
| Eyebrow Darkening | Pixel-based hair detection, not geometric masks |
| Brow Lift | Smooth uniform lift with outer-tail emphasis |
| Brow Tail Lift | Extra lift for the outer 25% of each brow |
| Lip Flip | Upper lip flip with teeth protection |
| Lip Liner | Smooth line along upper lip vermilion border |
| Nose Slimming | Side inward warp + tip lift |
| Nostril Left/Right | Independent nostril wing narrowing |
| Jaw Sharpening | Inward warp + micro-contrast along jawline |
| Marionette Lines | LAB lightness lift on detected creases |

## Presets

- **Subtle** — Very light touch, barely noticeable
- **Normal** — Balanced defaults
- **Strong** — More dramatic edits

## Saveable Filters

Save your custom slider settings as named filters. Filters appear as clickable buttons below the photo for quick switching.

## Project Structure

```
GlowUp/
  app.py            # Streamlit UI entry point
  effects.py        # All image processing functions
  face.py           # MediaPipe face detection, landmarks, masks
  requirements.txt  # Python dependencies
  README.md         # This file
```
