# FaceApp MVP

A local, privacy-first face editing app. No cloud uploads, no model training. Uses MediaPipe FaceMesh for landmark detection and OpenCV for image processing.

## Quick Start

```bash
cd faceapp_mvp
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## Features

| Effect | What it does |
|---|---|
| Skin Smoothing | Bilateral filter within a feathered face-oval mask. Smooths skin texture while preserving edges (eyes, lips, hairline). |
| Eye Whitening | Brightens the sclera (whites) of both eyes in LAB color space. Iris and pupil are excluded via MediaPipe iris landmarks. |
| Eyebrow Darkening | Darkens the outer third of each eyebrow for a more defined, groomed look. |
| Nose Slimming | Applies local "liquify" warps pushing the nose wings inward. Uses `cv2.remap` for smooth distortion. |
| Jaw Sharpening | Warps jaw-area pixels inward + applies micro-contrast along the jawline for a more sculpted look. |

## Presets

- **Subtle** - Very light touch, barely noticeable
- **Normal** - Balanced defaults
- **Strong** - More dramatic edits

## Debug Overlay

Check the "Debug overlay" checkbox in the sidebar to see:
- All 478 face landmarks (green dots)
- Face oval boundary (blue)
- Eye regions (cyan) and iris circles (magenta)
- Eyebrow paths (yellow)
- Nose landmarks (red)
- Jaw contour (green)

This is invaluable for checking whether masks align correctly with facial features.

## Troubleshooting

| Problem | Fix |
|---|---|
| "No face detected" | Use a well-lit, front-facing portrait. Ensure the face is clearly visible. |
| Eyebrow darkening in wrong spot | Adjust `LEFT_EYEBROW_INDICES` / `RIGHT_EYEBROW_INDICES` in `face.py` (lines ~34-37). |
| Eye whitening hits the iris | Tweak `LEFT_IRIS_INDICES` / `RIGHT_IRIS_INDICES` in `face.py` (lines ~27-28) or adjust the `iris_radius` multiplier in `make_eye_white_masks`. |
| Nose warp looks unnatural | Lower the intensity slider, or edit `NOSE_SIDE_INDICES_LEFT/RIGHT` in `face.py` (lines ~42-43). |
| Slow on large images | Resize your input to ~1080p before uploading. Effects scale with pixel count. |
| MediaPipe install fails | Try `pip install mediapipe --no-cache-dir`. On Apple Silicon, ensure you're using Python 3.11+. |

## Project Structure

```
faceapp_mvp/
  app.py            # Streamlit UI entry point
  effects.py        # All image processing functions
  face.py           # MediaPipe face detection, landmarks, masks
  requirements.txt  # Python dependencies
  README.md         # This file
```

## Where to Tweak Landmark Indices

All landmark index constants are at the top of `face.py`:
- **Face oval**: `FACE_OVAL_INDICES` - controls the skin smoothing boundary
- **Eyes**: `LEFT_EYE_INDICES`, `RIGHT_EYE_INDICES` - sclera region for whitening
- **Iris**: `LEFT_IRIS_INDICES`, `RIGHT_IRIS_INDICES` - excluded from whitening
- **Eyebrows**: `LEFT_EYEBROW_INDICES`, `RIGHT_EYEBROW_INDICES` - ordered inner to outer
- **Nose**: `NOSE_SIDE_INDICES_LEFT`, `NOSE_SIDE_INDICES_RIGHT` - warp anchor points
- **Jaw**: `JAW_INDICES` - contour for sharpening warps

Reference: [MediaPipe Face Mesh map](https://github.com/google/mediapipe/blob/master/mediapipe/modules/face_geometry/data/canonical_face_model_uv_visualization.png)
