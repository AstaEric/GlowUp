"""
FaceApp MVP - Streamlit entry point.
Local face editing with toggles and sliders.
"""

import streamlit as st
import cv2
import numpy as np
import logging
import time
import base64
import json
import os

from face import FaceDetector, draw_debug_overlay
from effects import apply_effects

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# --- Presets ---
PRESETS = {
    "Subtle": {
        "skin_smoothing": 0.25,
        "eye_whitening": 0.15,
        "eye_elongation": 0.10,
        "eye_enlargement": 0.10,
        "smize": 0.10,
        "eyelash_darkening": 0.15,
        "lower_lash_darkening": 0.10,
        "eyebrow_darkening": 0.15,
        "brow_lift": 0.15,
        "brow_tail_lift": 0.15,
        "lip_flip": 0.15,
        "lip_liner": 0.10,
        "nose_slimming": 0.10,
        "nostril_narrowing_left": 0.10,
        "nostril_narrowing_right": 0.10,
        "jaw_sharpening": 0.10,
        "marionette_lines": 0.15,
        "spot_removal": 0.15,
        "skin_glow": 0.15,
    },
    "Normal": {
        "skin_smoothing": 0.50,
        "eye_whitening": 0.55,
        "eye_elongation": 0.20,
        "eye_enlargement": 0.50,
        "smize": 0.20,
        "eyelash_darkening": 0.50,
        "lower_lash_darkening": 0.25,
        "eyebrow_darkening": 0.30,
        "brow_lift": 0.25,
        "brow_tail_lift": 0.25,
        "lip_flip": 0.30,
        "lip_liner": 0.20,
        "nose_slimming": 0.25,
        "nostril_narrowing_left": 0.20,
        "nostril_narrowing_right": 0.20,
        "jaw_sharpening": 0.25,
        "marionette_lines": 0.25,
        "spot_removal": 0.30,
        "skin_glow": 0.25,
    },
    "Strong": {
        "skin_smoothing": 0.80,
        "eye_whitening": 0.55,
        "eye_elongation": 0.35,
        "eye_enlargement": 0.40,
        "smize": 0.35,
        "eyelash_darkening": 0.55,
        "lower_lash_darkening": 0.40,
        "eyebrow_darkening": 0.55,
        "brow_lift": 0.40,
        "brow_tail_lift": 0.40,
        "lip_flip": 0.50,
        "lip_liner": 0.35,
        "nose_slimming": 0.50,
        "nostril_narrowing_left": 0.35,
        "nostril_narrowing_right": 0.35,
        "jaw_sharpening": 0.50,
        "marionette_lines": 0.40,
        "spot_removal": 0.50,
        "skin_glow": 0.40,
    },
}

DEFAULTS = PRESETS["Normal"]

EFFECTS_CONFIG = [
    ("skin_smoothing", "Skin Smoothing"),
    ("spot_removal", "Spot & Pore Removal"),
    ("eye_whitening", "Eye Whitening"),
    ("eye_elongation", "Eye Elongation"),
    ("eye_enlargement", "Eye Enlargement"),
    ("smize", "Smize"),
    ("eyelash_darkening", "Eyelash Darkening"),
    ("lower_lash_darkening", "Lower Lash Line"),
    ("eyebrow_darkening", "Eyebrow Darkening"),
    ("brow_lift", "Brow Lift"),
    ("brow_tail_lift", "Brow Tail Lift"),
    ("lip_flip", "Lip Flip"),
    ("lip_liner", "Lip Liner"),
    ("nose_slimming", "Nose Slimming"),
    ("nostril_narrowing_left", "Nostril Left"),
    ("nostril_narrowing_right", "Nostril Right"),
    ("jaw_sharpening", "Jaw Sharpening"),
    ("marionette_lines", "Marionette Lines"),
    ("skin_glow", "Skin Glow"),
]

# ---------------------------------------------------------------------------
# Filter storage — multiple numbered filters in a single JSON file
# ---------------------------------------------------------------------------

FILTERS_PATH = os.path.join(os.path.dirname(__file__), "my_filters.json")


def _load_filters_file():
    """Load the filters file. Returns a list of filter dicts."""
    if not os.path.exists(FILTERS_PATH):
        return []
    try:
        with open(FILTERS_PATH, "r") as f:
            data = json.load(f)
        if isinstance(data, list):
            return data
    except (json.JSONDecodeError, KeyError):
        pass
    return []


def _save_filters_file(filters):
    """Write the filters list to disk."""
    with open(FILTERS_PATH, "w") as f:
        json.dump(filters, f, indent=2)


def _current_settings_snapshot():
    """Capture current slider/toggle state as a dict."""
    snap = {}
    for key, _ in EFFECTS_CONFIG:
        snap[key] = {
            "enabled": st.session_state.get(f"{key}_enabled", True),
            "intensity": st.session_state.get(f"{key}_intensity", 0.5),
        }
    return snap


def save_as_new_filter():
    """Save current settings as a new numbered filter."""
    filters = _load_filters_file()
    num = len(filters) + 1
    name = f"Filter {num}"
    filters.append({"name": name, "settings": _current_settings_snapshot()})
    _save_filters_file(filters)
    return name


def save_over_filter(index):
    """Overwrite an existing filter with current settings."""
    filters = _load_filters_file()
    if 0 <= index < len(filters):
        filters[index]["settings"] = _current_settings_snapshot()
        _save_filters_file(filters)


def rename_filter(index, new_name):
    """Rename an existing filter."""
    filters = _load_filters_file()
    if 0 <= index < len(filters):
        filters[index]["name"] = new_name
        _save_filters_file(filters)


def delete_filter(index):
    """Delete a filter by index."""
    filters = _load_filters_file()
    if 0 <= index < len(filters):
        filters.pop(index)
        _save_filters_file(filters)


def load_filter(index):
    """Apply a saved filter's settings to session state."""
    filters = _load_filters_file()
    if 0 <= index < len(filters):
        data = filters[index]["settings"]
        for key, _ in EFFECTS_CONFIG:
            if key in data:
                st.session_state[f"{key}_enabled"] = data[key].get("enabled", True)
                st.session_state[f"{key}_intensity"] = data[key].get("intensity", 0.5)
        st.session_state["_active_filter_idx"] = index


def get_filter_names():
    """Return list of (index, name) for all saved filters."""
    filters = _load_filters_file()
    return [(i, f.get("name", f"Filter {i+1}")) for i, f in enumerate(filters)]


# ---------------------------------------------------------------------------
# Session state helpers
# ---------------------------------------------------------------------------

def init_session_state():
    """Initialize session state with Normal preset defaults."""
    for key, val in DEFAULTS.items():
        if f"{key}_enabled" not in st.session_state:
            st.session_state[f"{key}_enabled"] = True
        if f"{key}_intensity" not in st.session_state:
            st.session_state[f"{key}_intensity"] = val
    if "debug_overlay" not in st.session_state:
        st.session_state["debug_overlay"] = False
    if "_active_filter_idx" not in st.session_state:
        st.session_state["_active_filter_idx"] = None


def apply_preset(preset_name):
    """Apply a named preset to all sliders."""
    preset = PRESETS[preset_name]
    for key, val in preset.items():
        st.session_state[f"{key}_intensity"] = val
        st.session_state[f"{key}_enabled"] = True
    st.session_state["_active_filter_idx"] = None


def reset_defaults():
    """Reset all controls to Normal preset defaults."""
    apply_preset("Normal")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    st.set_page_config(page_title="FaceApp MVP", layout="wide")
    st.title("FaceApp MVP")
    st.caption("Local face editing - no cloud, no training. Powered by MediaPipe + OpenCV.")

    init_session_state()

    # --- Sidebar controls ---
    with st.sidebar:
        st.header("Controls")

        # Presets
        st.subheader("Presets")
        cols = st.columns(3)
        for i, name in enumerate(PRESETS):
            if cols[i].button(name, use_container_width=True):
                apply_preset(name)
                st.rerun()

        if st.button("Reset to defaults", use_container_width=True):
            reset_defaults()
            st.rerun()

        st.divider()

        # Effect toggles + sliders
        st.subheader("Effects")

        for key, label in EFFECTS_CONFIG:
            st.session_state[f"{key}_enabled"] = st.toggle(
                label, value=st.session_state[f"{key}_enabled"], key=f"toggle_{key}"
            )
            if st.session_state[f"{key}_enabled"]:
                st.session_state[f"{key}_intensity"] = st.slider(
                    f"{label} intensity",
                    0.0, 1.0,
                    value=st.session_state[f"{key}_intensity"],
                    step=0.05,
                    key=f"slider_{key}",
                )

        st.divider()
        st.session_state["debug_overlay"] = st.checkbox(
            "Debug overlay (show landmarks & masks)",
            value=st.session_state["debug_overlay"],
        )

    # --- File uploader ---
    uploaded = st.file_uploader("Upload a portrait photo", type=["jpg", "jpeg", "png", "webp"])

    # Cache uploaded image bytes in session state so we can reprocess
    if uploaded is not None:
        st.session_state["cached_image_bytes"] = np.frombuffer(uploaded.read(), dtype=np.uint8)

    if "cached_image_bytes" not in st.session_state:
        st.info("Upload a portrait photo to get started.")
        return

    # Decode image from cache
    original = cv2.imdecode(st.session_state["cached_image_bytes"], cv2.IMREAD_COLOR)

    if original is None:
        st.error("Could not decode the uploaded image. Please try a different file.")
        return

    # Reprocess button — re-runs with current slider values, no re-upload needed
    st.button("🔄 Reprocess", use_container_width=True)

    h, w = original.shape[:2]
    st.caption(f"Original size: {w} x {h}")

    # Downscale large images for processing (keep original for final output)
    MAX_DIM = 1920
    scale = 1.0
    if max(h, w) > MAX_DIM:
        scale = MAX_DIM / max(h, w)
        proc_image = cv2.resize(original, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        st.caption(f"Processing at: {proc_image.shape[1]} x {proc_image.shape[0]} (scaled down for speed)")
    else:
        proc_image = original

    # Detect face
    detector = FaceDetector()
    with st.spinner("Detecting face..."):
        landmarks = detector.detect(proc_image)

    if landmarks is None:
        st.error("No face detected. Please upload a clear portrait photo with a visible face.")
        # Still show original
        st.image(cv2.cvtColor(original, cv2.COLOR_BGR2RGB), caption="Original (no face detected)")
        return

    # If we downscaled for detection, scale landmarks back to original size
    if scale < 1.0:
        landmarks = [(int(x / scale), int(y / scale)) for x, y in landmarks]

    # Build settings dict
    settings = {}
    for key, _ in EFFECTS_CONFIG:
        settings[key] = {
            "enabled": st.session_state[f"{key}_enabled"],
            "intensity": st.session_state[f"{key}_intensity"],
        }

    # Apply effects
    start = time.time()
    with st.spinner("Applying effects..."):
        result = apply_effects(original, landmarks, settings)
    elapsed = time.time() - start
    logger.info(f"Effects applied in {elapsed:.2f}s")

    # Debug overlay
    if st.session_state["debug_overlay"]:
        result = draw_debug_overlay(result, landmarks)

    # Encode images as base64 for the HTML component
    _, after_buf = cv2.imencode(".png", result)
    _, before_buf = cv2.imencode(".png", original)
    after_b64 = base64.b64encode(after_buf).decode()
    before_b64 = base64.b64encode(before_buf).decode()

    # Display with click-to-reveal-original
    st.components.v1.html(f"""
    <div id="img-container" style="position:relative;cursor:pointer;user-select:none;-webkit-user-select:none;">
        <img id="after-img" src="data:image/png;base64,{after_b64}" style="width:100%;display:block;" draggable="false"/>
        <img id="before-img" src="data:image/png;base64,{before_b64}" style="width:100%;display:none;position:absolute;top:0;left:0;" draggable="false"/>
        <div id="label" style="position:absolute;top:12px;left:12px;background:rgba(0,0,0,0.55);color:#fff;padding:4px 10px;border-radius:4px;font:600 14px/1 sans-serif;pointer-events:none;">After</div>
    </div>
    <p style="text-align:center;color:#888;font:12px sans-serif;margin-top:6px;">Click and hold to see original</p>
    <script>
        const c = document.getElementById('img-container');
        const a = document.getElementById('after-img');
        const b = document.getElementById('before-img');
        const l = document.getElementById('label');
        function showBefore() {{ b.style.display='block'; a.style.display='none'; l.textContent='Before'; }}
        function showAfter() {{ a.style.display='block'; b.style.display='none'; l.textContent='After'; }}
        c.addEventListener('mousedown', showBefore);
        c.addEventListener('mouseup', showAfter);
        c.addEventListener('mouseleave', showAfter);
        c.addEventListener('touchstart', function(e) {{ e.preventDefault(); showBefore(); }});
        c.addEventListener('touchend', showAfter);
        c.addEventListener('touchcancel', showAfter);
    </script>
    """, height=int(h * 800 / w) + 40)

    st.caption(f"Processing time: {elapsed:.2f}s")

    # --- Saved-filter area below the photo ---
    st.markdown("---")

    # "Save as new filter" button — always visible
    if st.button("💾 Save current settings as new filter", key="save_new_main",
                 use_container_width=True):
        name = save_as_new_filter()
        st.session_state["_active_filter_idx"] = len(get_filter_names()) - 1
        st.rerun()

    # Clickable saved-filter buttons
    filter_names = get_filter_names()
    if filter_names:
        st.markdown("**My Filters** — click to apply")
        active_idx = st.session_state.get("_active_filter_idx", None)
        cols_per_row = 4
        for row_start in range(0, len(filter_names), cols_per_row):
            row_items = filter_names[row_start : row_start + cols_per_row]
            cols = st.columns(cols_per_row)
            for col_i, (f_idx, f_name) in enumerate(row_items):
                with cols[col_i]:
                    btn_type = "primary" if f_idx == active_idx else "secondary"
                    if st.button(f_name, key=f"flt_btn_{f_idx}",
                                 use_container_width=True, type=btn_type):
                        load_filter(f_idx)
                        st.rerun()

        # If a filter is active, show save-over / rename / delete right here
        if active_idx is not None and 0 <= active_idx < len(filter_names):
            active_name = filter_names[active_idx][1]
            c1, c2, c3 = st.columns(3)
            with c1:
                if st.button(f"💾 Update '{active_name}'", key="save_over_main",
                             use_container_width=True):
                    save_over_filter(active_idx)
                    st.rerun()
            with c2:
                if st.button("✏️ Rename", key="rename_main", use_container_width=True):
                    st.session_state["_renaming_main"] = True
            with c3:
                if st.button("🗑️ Delete", key="delete_main", use_container_width=True):
                    delete_filter(active_idx)
                    st.session_state["_active_filter_idx"] = None
                    st.rerun()
            if st.session_state.get("_renaming_main", False):
                new_name = st.text_input("New name", value=active_name, key="_rename_main_input")
                if st.button("OK", key="rename_main_ok"):
                    rename_filter(active_idx, new_name)
                    st.session_state["_renaming_main"] = False
                    st.rerun()

    st.markdown("---")

    # Download button
    _, buf = cv2.imencode(".png", result)
    st.download_button(
        label="Download edited photo",
        data=buf.tobytes(),
        file_name="faceapp_result.png",
        mime="image/png",
    )


if __name__ == "__main__":
    main()
