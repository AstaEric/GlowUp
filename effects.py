"""
Image processing effects for GlowUp.
Each effect takes an image, landmarks, and an intensity (0..1) and returns the modified image.
"""

import cv2
import numpy as np
import logging

from face import (
    make_face_oval_mask,
    make_skin_mask,
    make_eye_white_masks,
    make_outer_eyebrow_masks,
    make_full_eyebrow_masks,
    make_upper_eyelid_masks,
    make_lower_eyelid_masks,
    get_landmark_points,
    NOSE_BRIDGE_INDEX,
    NOSE_TIP_INDEX,
    LEFT_NOSE_WING,
    RIGHT_NOSE_WING,
    NOSE_SIDE_INDICES_LEFT,
    NOSE_SIDE_INDICES_RIGHT,
    JAW_INDICES,
    LEFT_JAW_CORNER,
    RIGHT_JAW_CORNER,
    CHIN_INDEX,
    LEFT_EYE_INDICES,
    RIGHT_EYE_INDICES,
    LEFT_IRIS_INDICES,
    RIGHT_IRIS_INDICES,
    UPPER_LIP_TOP_INDICES,
    UPPER_LIP_BOTTOM_INDICES,
    LEFT_LIP_CORNER,
    RIGHT_LIP_CORNER,
    LEFT_EYEBROW_UPPER,
    LEFT_EYEBROW_LOWER,
    RIGHT_EYEBROW_UPPER,
    RIGHT_EYEBROW_LOWER,
)

logger = logging.getLogger(__name__)


def _blend(original, modified, mask):
    """Alpha-blend modified onto original using a single-channel mask."""
    mask_3ch = np.stack([mask] * 3, axis=-1)
    return (original * (1 - mask_3ch) + modified * mask_3ch).astype(np.uint8)


# ---------------------------------------------------------------------------
# 1) Skin Smoothing
# ---------------------------------------------------------------------------

def skin_smoothing(image, landmarks, intensity=0.5):
    """
    Smooth skin texture only — excludes eyes, eyebrows, lips, and non-skin areas.
    Preserves edges and contrast while removing skin texture.

    intensity: 0 = no smoothing, 1 = maximum smoothing
    """
    if intensity <= 0:
        return image

    # Bilateral filter params scale with intensity
    d = int(5 + intensity * 10)  # diameter: 5..15
    sigma_color = 20 + intensity * 55  # 20..75
    sigma_space = 20 + intensity * 55

    smoothed = cv2.bilateralFilter(image, d, sigma_color, sigma_space)

    # Use skin-only mask (excludes eyes, brows, lips)
    mask = make_skin_mask(landmarks, image.shape, feather=21)
    mask = mask * intensity

    return _blend(image, smoothed, mask)


# ---------------------------------------------------------------------------
# 2) Eye Whitening
# ---------------------------------------------------------------------------

def eye_whitening(image, landmarks, intensity=0.3):
    """
    Brighten the whites of the eyes (sclera), avoiding the iris/pupil.

    intensity: 0 = no change, 1 = maximum brightening
    """
    if intensity <= 0:
        return image

    left_mask, right_mask = make_eye_white_masks(landmarks, image.shape, feather=5)
    combined_mask = np.clip(left_mask + right_mask, 0, 1)

    # Brighten in LAB space (L channel only) for natural results
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
    boost = intensity * 40  # up to +40 lightness
    lab[:, :, 0] = np.clip(lab[:, :, 0] + boost, 0, 255)
    brightened = cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR)

    # Also slightly desaturate for whiter appearance
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] = hsv[:, :, 1] * (1 - intensity * 0.5)
    desaturated = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    # Blend brightened + desaturated
    white_enhanced = cv2.addWeighted(brightened, 0.6, desaturated, 0.4, 0)

    combined_mask *= intensity
    return _blend(image, white_enhanced, combined_mask)


# ---------------------------------------------------------------------------
# 2a-ii) Eye Elongation — stretch eyes horizontally for an almond shape
# ---------------------------------------------------------------------------

def eye_elongation(image, landmarks, intensity=0.3):
    """
    Elongate the eyes horizontally by pushing the outer corner outward
    and the inner corner very slightly inward.  The iris is protected
    so it stays round.

    Uses a single-pass accumulated displacement field for smoothness.

    intensity: 0 = no change, 1 = maximum elongation
    """
    if intensity <= 0:
        return image

    h, w = image.shape[:2]
    face_width = abs(landmarks[234][0] - landmarks[454][0])

    map_x, map_y = np.meshgrid(
        np.arange(w, dtype=np.float32),
        np.arange(h, dtype=np.float32),
    )
    total_dx = np.zeros((h, w), dtype=np.float32)

    # Left eye: inner=263, outer=362   (left in MediaPipe = right side of face)
    # Right eye: inner=33, outer=133
    eye_configs = [
        {
            "outer_idx": 362, "inner_idx": 263,
            "iris": LEFT_IRIS_INDICES,
            "eye": LEFT_EYE_INDICES,
        },
        {
            "outer_idx": 133, "inner_idx": 33,
            "iris": RIGHT_IRIS_INDICES,
            "eye": RIGHT_EYE_INDICES,
        },
    ]

    for cfg in eye_configs:
        outer = np.array(landmarks[cfg["outer_idx"]], dtype=float)
        inner = np.array(landmarks[cfg["inner_idx"]], dtype=float)
        iris_pts = np.array([landmarks[i] for i in cfg["iris"]], dtype=float)
        eye_pts = np.array([landmarks[i] for i in cfg["eye"]], dtype=float)

        eye_w = np.linalg.norm(outer - inner)
        if eye_w < 1:
            continue

        # Direction from eye centre outward (horizontal component dominates)
        eye_cx = (outer[0] + inner[0]) / 2.0
        eye_cy = (outer[1] + inner[1]) / 2.0

        # Push strength proportional to eye width
        max_push = eye_w * 0.04 * intensity

        # Iris protection circle
        iris_cx = float(iris_pts[:, 0].mean())
        iris_cy = float(iris_pts[:, 1].mean())
        iris_r = float(np.linalg.norm(iris_pts[0] - iris_pts[2]) * 0.55)
        protect_r = iris_r * 1.5

        # --- Outer corner push (outward, away from nose) ---
        out_dir = 1.0 if outer[0] > inner[0] else -1.0
        rx_out = eye_w * 0.45
        ry_out = eye_w * 0.35
        dx_out = (map_x - outer[0]) / max(rx_out, 1)
        dy_out = (map_y - outer[1]) / max(ry_out, 1)
        dist_out = np.sqrt(dx_out ** 2 + dy_out ** 2)
        falloff_out = np.clip(1.0 - dist_out, 0, 1) ** 2

        total_dx += out_dir * max_push * falloff_out

        # --- Inner corner push (inward, toward nose) — much gentler ---
        rx_in = eye_w * 0.30
        ry_in = eye_w * 0.25
        dx_in = (map_x - inner[0]) / max(rx_in, 1)
        dy_in = (map_y - inner[1]) / max(ry_in, 1)
        dist_in = np.sqrt(dx_in ** 2 + dy_in ** 2)
        falloff_in = np.clip(1.0 - dist_in, 0, 1) ** 2

        total_dx -= out_dir * max_push * 0.35 * falloff_in

        # --- Iris protection: fade displacement to zero inside iris ---
        iris_dist = np.sqrt((map_x - iris_cx) ** 2 + (map_y - iris_cy) ** 2)
        iris_shield = np.clip((iris_dist - iris_r * 0.6) / (protect_r - iris_r * 0.6), 0, 1)

        # Zero-out displacement inside iris for this eye's contribution
        # We apply it to total_dx but only the portion this eye added
        eye_contribution = out_dir * max_push * falloff_out - out_dir * max_push * 0.35 * falloff_in
        iris_correction = eye_contribution * (1.0 - iris_shield)
        total_dx -= iris_correction

    new_map_x = (map_x - total_dx).astype(np.float32)
    result = cv2.remap(image, new_map_x, map_y, cv2.INTER_LINEAR,
                       borderMode=cv2.BORDER_REFLECT)
    return result


# ---------------------------------------------------------------------------
# 2b) Eye Enlargement
# ---------------------------------------------------------------------------

def eye_enlargement(image, landmarks, intensity=0.3):
    """
    Subtly enlarge eyes using outward warps from the eye center.

    intensity: 0 = no change, 1 = maximum enlargement
    """
    if intensity <= 0:
        return image

    result = image.copy()

    for eye_indices, iris_indices in [
        (LEFT_EYE_INDICES, LEFT_IRIS_INDICES),
        (RIGHT_EYE_INDICES, RIGHT_IRIS_INDICES),
    ]:
        eye_pts = get_landmark_points(landmarks, eye_indices)
        iris_pts = get_landmark_points(landmarks, iris_indices)

        # Eye center from iris
        center = iris_pts.mean(axis=0).astype(int)

        # Compute iris circle for protection
        iris_center_x = float(iris_pts.mean(axis=0)[0])
        iris_center_y = float(iris_pts.mean(axis=0)[1])
        iris_radius = float(np.linalg.norm(iris_pts[0].astype(float) - iris_pts[2].astype(float)) * 0.55)
        iris_protect = (iris_center_x, iris_center_y, iris_radius)

        # Eye size for radius
        eye_width = np.linalg.norm(eye_pts[0].astype(float) - eye_pts[len(eye_pts)//2].astype(float))
        radius = int(eye_width * 0.8)
        strength = intensity * eye_width * 0.08

        # Warp upper eyelid points upward, lower eyelid points downward
        # Iris is protected — displacement fades to zero inside the iris circle
        upper = eye_pts[len(eye_pts)//2:]  # upper half
        lower = eye_pts[:len(eye_pts)//2]  # lower half

        for pt in upper:
            result = _local_warp(result, tuple(pt), (0, -1), radius, strength / len(upper), iris_protect=iris_protect)
        for pt in lower:
            result = _local_warp(result, tuple(pt), (0, 1), radius, strength / len(lower), iris_protect=iris_protect)

    # Re-sharpen the eye regions to restore detail lost by warping
    h, w = image.shape[:2]
    for eye_indices, iris_indices in [
        (LEFT_EYE_INDICES, LEFT_IRIS_INDICES),
        (RIGHT_EYE_INDICES, RIGHT_IRIS_INDICES),
    ]:
        eye_pts = get_landmark_points(landmarks, eye_indices)
        iris_pts = get_landmark_points(landmarks, iris_indices)

        # Build a feathered mask around the eye area
        eye_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(eye_mask, [eye_pts], 255)
        # Dilate to cover lashes and surrounding area affected by warp
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
        eye_mask = cv2.dilate(eye_mask, kernel)
        eye_mask = cv2.GaussianBlur(eye_mask.astype(np.float32), (15, 15), 0) / 255.0

        # Unsharp mask sharpening — restore crispness
        sharpen_strength = 0.5 + intensity * 1.0  # stronger sharpening at higher enlargement
        blurred = cv2.GaussianBlur(result, (0, 0), 2.0)
        sharpened = cv2.addWeighted(result, 1.0 + sharpen_strength, blurred, -sharpen_strength, 0)

        # Blend sharpened version only in the eye region
        result = _blend(result, sharpened, eye_mask * min(intensity * 1.5, 1.0))

    return result


# ---------------------------------------------------------------------------
# 2c) Eyelash Darkening
# ---------------------------------------------------------------------------

def _draw_outer_lash_strokes(mask, outer_pt, lid_direction, lash_length,
                             num_lashes=5, base_thickness=1):
    """
    Draw individual curved lash strokes fanning out from the outer eye corner.

    outer_pt:      (x, y) the outer corner of the eye
    lid_direction: (dx, dy) unit vector along the upper lid pointing outward
    lash_length:   how long each lash stroke is in pixels
    num_lashes:    how many individual strokes to draw
    base_thickness: line thickness in pixels
    """
    ox, oy = float(outer_pt[0]), float(outer_pt[1])
    dx, dy = float(lid_direction[0]), float(lid_direction[1])

    # Perpendicular to lid direction (pointing away from the eye = upward-ish)
    # The lashes curl away from the eye — perpendicular to the lid edge
    perp_x, perp_y = -dy, dx  # rotate 90° CCW

    # Make sure perp points upward (away from eye center)
    if perp_y > 0:
        perp_x, perp_y = -perp_x, -perp_y

    for i in range(num_lashes):
        # Spread lashes along the lid direction at the outer corner
        t = i / max(num_lashes - 1, 1)  # 0..1

        # Starting point: spread along the last bit of the lid edge
        spread = lash_length * 0.8
        start_x = ox + dx * t * spread * 0.3
        start_y = oy + dy * t * spread * 0.3

        # Each lash has a slightly different angle — fan outward
        # Base angle follows the perpendicular, fanning from lid-tangent toward perpendicular
        fan_angle = 0.3 + t * 0.7  # inner lashes more along lid, outer more perpendicular
        lash_dx = dx * (1 - fan_angle) + perp_x * fan_angle
        lash_dy = dy * (1 - fan_angle) + perp_y * fan_angle
        # Normalize
        mag = max(np.sqrt(lash_dx ** 2 + lash_dy ** 2), 1e-6)
        lash_dx /= mag
        lash_dy /= mag

        # Length varies: longer in the middle, shorter at edges
        length_factor = 0.6 + 0.4 * (1.0 - abs(t - 0.4) / 0.6)
        this_length = lash_length * length_factor

        # Draw a curved stroke using a quadratic bezier (3 points)
        # Start → control point (curving outward) → end
        mid_x = start_x + lash_dx * this_length * 0.5
        mid_y = start_y + lash_dy * this_length * 0.5
        # Add curvature: bend the tip further away from the eye
        curve_strength = this_length * 0.25
        mid_x += perp_x * curve_strength * 0.5
        mid_y += perp_y * curve_strength * 0.5

        end_x = start_x + lash_dx * this_length
        end_y = start_y + lash_dy * this_length
        end_x += perp_x * curve_strength
        end_y += perp_y * curve_strength

        # Sample points along the quadratic bezier
        n_steps = 8
        pts = []
        for s in range(n_steps + 1):
            u = s / n_steps
            # Quadratic bezier: (1-u)^2 * P0 + 2(1-u)u * P1 + u^2 * P2
            bx = (1 - u) ** 2 * start_x + 2 * (1 - u) * u * mid_x + u ** 2 * end_x
            by = (1 - u) ** 2 * start_y + 2 * (1 - u) * u * mid_y + u ** 2 * end_y
            pts.append((int(round(bx)), int(round(by))))

        # Opacity: full at base, fading at tip
        for j in range(len(pts) - 1):
            tip_fade = 1.0 - (j / max(len(pts) - 1, 1)) * 0.6
            opacity = int(255 * tip_fade)
            cv2.line(mask, pts[j], pts[j + 1], opacity, base_thickness)


def eyelash_darkening(image, landmarks, intensity=0.3):
    """
    Darken and sharpen the upper eyelid lash line for crisp, defined lashes.

    Automatically detects which eye is further from the camera (appears
    narrower) and boosts the outer-corner lash mask on that eye so both
    eyes look balanced.

    intensity: 0 = no change, 1 = maximum effect
    """
    if intensity <= 0:
        return image

    # Scale line thickness with image size
    h, w = image.shape[:2]
    thickness = max(2, int(min(h, w) / 400))

    # Use minimal feathering for a crisp lash line
    left_mask, right_mask = make_upper_eyelid_masks(
        landmarks, image.shape, thickness=thickness, feather=3
    )

    # --- Draw individual lash strokes at the outer corner of each eye ---
    left_inner = np.array(landmarks[263], dtype=float)
    left_outer = np.array(landmarks[362], dtype=float)
    left_w = np.linalg.norm(left_outer - left_inner)

    right_inner = np.array(landmarks[33], dtype=float)
    right_outer = np.array(landmarks[133], dtype=float)
    right_w = np.linalg.norm(right_outer - right_inner)

    # Left eye: lid direction at outer corner from second-to-last → last landmark
    # LEFT_UPPER_EYELID = [362, 398, ...] — 362 is outer, 398 is next inward
    left_lid_dir = left_outer - np.array(landmarks[398], dtype=float)
    mag = np.linalg.norm(left_lid_dir)
    if mag > 0:
        left_lid_dir /= mag
    lash_len = left_w * 0.12  # lash length proportional to eye
    num_lashes = max(3, int(intensity * 7))
    # Draw on a temp uint8 mask, then merge
    left_lash_strokes = np.zeros((h, w), dtype=np.uint8)
    _draw_outer_lash_strokes(left_lash_strokes, left_outer, left_lid_dir,
                             lash_len, num_lashes=num_lashes, base_thickness=1)
    left_stroke_f = cv2.GaussianBlur(left_lash_strokes.astype(np.float32),
                                      (3, 3), 0) / 255.0
    left_mask = np.clip(left_mask + left_stroke_f * intensity, 0, 1)

    # Right eye: lid direction from second-to-last → outer
    # RIGHT_UPPER_EYELID = [..., 173, 133] — 133 is outer, 173 is next inward
    right_lid_dir = right_outer - np.array(landmarks[173], dtype=float)
    mag = np.linalg.norm(right_lid_dir)
    if mag > 0:
        right_lid_dir /= mag
    lash_len_r = right_w * 0.12
    right_lash_strokes = np.zeros((h, w), dtype=np.uint8)
    _draw_outer_lash_strokes(right_lash_strokes, right_outer, right_lid_dir,
                             lash_len_r, num_lashes=num_lashes, base_thickness=1)
    right_stroke_f = cv2.GaussianBlur(right_lash_strokes.astype(np.float32),
                                       (3, 3), 0) / 255.0
    right_mask = np.clip(right_mask + right_stroke_f * intensity, 0, 1)

    # --- Boost the far eye's outer corner ---
    # The narrower eye is further from camera — give it extra lash at outer corner
    if left_w > 0 and right_w > 0:
        ratio = min(left_w, right_w) / max(left_w, right_w)
        # Only boost if there's a noticeable difference (ratio < 0.95)
        if ratio < 0.95:
            # Boost strength: bigger difference → more boost (up to 60% extra)
            boost = min(0.6, (1.0 - ratio) * 3.0)

            # Pick the far eye and its outer corner point
            if left_w < right_w:
                far_mask = left_mask
                outer_pt = left_outer
            else:
                far_mask = right_mask
                outer_pt = right_outer

            far_eye_w = min(left_w, right_w)

            # Paint a soft radial boost centered on the outer corner
            # covering the outer ~40% of the eye
            boost_radius = far_eye_w * 0.45
            ys_arr = np.arange(h, dtype=np.float32)
            xs_arr = np.arange(w, dtype=np.float32)
            yy, xx = np.meshgrid(ys_arr, xs_arr, indexing='ij')
            dist_from_outer = np.sqrt(
                (xx - outer_pt[0]) ** 2 + (yy - outer_pt[1]) ** 2
            )
            boost_falloff = np.clip(1.0 - dist_from_outer / boost_radius, 0, 1) ** 2

            # Thicken: add extra mask density at the outer corner of the far eye
            # Only where the existing mask already has some value (don't create new line)
            existing = far_mask > 0.01
            extra = boost_falloff * boost * existing.astype(np.float32)
            far_mask[:] = np.clip(far_mask + extra, 0, 1)

    combined_mask = np.clip(left_mask + right_mask, 0, 1)

    # Square the mask so faint inner-corner values become even more transparent
    combined_mask = combined_mask ** 2

    # Sharpen the lash area first with unsharp mask
    blur = cv2.GaussianBlur(image, (0, 0), 1.5)
    sharpened = cv2.addWeighted(image, 1.0 + intensity * 0.8, blur, -intensity * 0.8, 0)

    # Darken toward the existing lash color (not blue/gray)
    # Sample the average color in the lash mask region to preserve hue
    mask_bool = combined_mask > 0.1
    if mask_bool.any():
        lash_pixels = sharpened[mask_bool]
        avg_color = lash_pixels.mean(axis=0)
    else:
        avg_color = np.array([30, 20, 15], dtype=np.float32)  # fallback dark brown

    # Create a darkened version that pushes toward a darker version of the lash color
    dark_target = (avg_color * 0.2).astype(np.uint8)  # very dark version of same hue
    dark_layer = np.full_like(sharpened, dark_target)
    blend_strength = intensity * 0.7
    darkened = cv2.addWeighted(sharpened, 1.0 - blend_strength, dark_layer, blend_strength, 0)

    combined_mask *= intensity
    result = _blend(image, darkened, combined_mask)

    # --- Extra sharpness (nitidezza) on the far eye's outer corner ---
    # The eye further from the camera loses detail; restore it with targeted
    # unsharp mask on that eye region
    if left_w > 0 and right_w > 0:
        ratio = min(left_w, right_w) / max(left_w, right_w)
        if ratio < 0.97:
            if left_w < right_w:
                far_eye_indices = LEFT_EYE_INDICES
                far_outer = left_outer
                far_eye_w = left_w
            else:
                far_eye_indices = RIGHT_EYE_INDICES
                far_outer = right_outer
                far_eye_w = right_w

            # Build a soft mask around the far eye, weighted toward outer corner
            far_eye_pts = get_landmark_points(landmarks, far_eye_indices)
            eye_mask = np.zeros((h, w), dtype=np.uint8)
            cv2.fillPoly(eye_mask, [far_eye_pts], 255)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
            eye_mask = cv2.dilate(eye_mask, kernel)
            eye_mask_f = cv2.GaussianBlur(eye_mask.astype(np.float32),
                                           (11, 11), 0) / 255.0

            # Weight toward outer corner
            sharp_radius = far_eye_w * 0.6
            dist_outer = np.sqrt(
                (np.arange(w, dtype=np.float32)[np.newaxis, :] - far_outer[0]) ** 2 +
                (np.arange(h, dtype=np.float32)[:, np.newaxis] - far_outer[1]) ** 2
            )
            outer_weight = np.clip(1.0 - dist_outer / sharp_radius, 0.3, 1.0)
            eye_mask_f = eye_mask_f * outer_weight

            # Unsharp mask — boost proportional to asymmetry
            sharp_amount = 0.4 + (1.0 - ratio) * 2.5
            blur_s = cv2.GaussianBlur(result, (0, 0), 1.2)
            sharpened_far = cv2.addWeighted(result, 1.0 + sharp_amount,
                                             blur_s, -sharp_amount, 0)

            eye_mask_f = eye_mask_f * min(intensity, 0.8)
            result = _blend(result, sharpened_far, eye_mask_f)

    return result


# ---------------------------------------------------------------------------
# 2b) Lower Lash Line — very subtle, transparent darkening below the eye
# ---------------------------------------------------------------------------

def lower_lash_darkening(image, landmarks, intensity=0.3):
    """
    Very subtle darkening along the lower lash line.
    Much more transparent than the upper lashes — just a hint of definition.
    Tapers thinner toward the inner corner (near nose).

    intensity: 0 = no change, 1 = maximum (still quite subtle)
    """
    if intensity <= 0:
        return image

    h, w = image.shape[:2]
    # Thinner line than upper lashes
    thickness = max(1, int(min(h, w) / 600))

    left_mask, right_mask = make_lower_eyelid_masks(
        landmarks, image.shape, thickness=thickness, feather=3
    )
    combined_mask = np.clip(left_mask + right_mask, 0, 1)

    # Sample existing color along the lower lid to keep it natural
    mask_bool = combined_mask > 0.1
    if mask_bool.any():
        lid_pixels = image[mask_bool]
        avg_color = lid_pixels.mean(axis=0)
    else:
        avg_color = np.array([40, 30, 25], dtype=np.float32)

    # Dark target — darker version of existing lid colour
    dark_target = (avg_color * 0.25).astype(np.uint8)
    dark_layer = np.full_like(image, dark_target)

    # Very low blend strength — this should be barely noticeable
    blend_strength = intensity * 0.35
    darkened = cv2.addWeighted(image, 1.0 - blend_strength, dark_layer, blend_strength, 0)

    # Scale mask down further for extra transparency
    combined_mask *= intensity * 0.5
    return _blend(image, darkened, combined_mask)


# ---------------------------------------------------------------------------
# 3) Eyebrow Darkening — pixel-based hair detection, no geometric masks
# ---------------------------------------------------------------------------

def eyebrow_darkening(image, landmarks, intensity=0.3):
    """
    Darken existing brow hairs by detecting them from the image itself.
    No geometric polygon masks — works by finding pixels that are darker
    than surrounding skin (i.e. actual hair), then deepening their color.

    Strategy:
    1. Use landmarks only as a rough search region (with generous padding)
    2. Within that region, detect brow hair pixels by comparing local
       brightness against a local skin-tone baseline
    3. Build a soft mask from the detected hair pixels
    4. Darken only those pixels, proportional to how much they already
       look like brow hair

    This follows the actual shape of the brows, not a geometric polygon.

    intensity: 0 = no change, 1 = maximum darkening
    """
    if intensity <= 0:
        return image

    h, w = image.shape[:2]
    result_f = image.astype(np.float32).copy()

    for upper_idx, lower_idx in [
        (LEFT_EYEBROW_UPPER, LEFT_EYEBROW_LOWER),
        (RIGHT_EYEBROW_UPPER, RIGHT_EYEBROW_LOWER),
    ]:
        upper_pts = get_landmark_points(landmarks, upper_idx)
        lower_pts = get_landmark_points(landmarks, lower_idx)
        all_pts = np.concatenate([upper_pts, lower_pts], axis=0)

        # Bounding box of the brow landmarks with generous padding
        x_min = int(all_pts[:, 0].min())
        x_max = int(all_pts[:, 0].max())
        y_min = int(all_pts[:, 1].min())
        y_max = int(all_pts[:, 1].max())

        brow_w = x_max - x_min
        brow_h = y_max - y_min

        # Pad generously — we want to include any stray hairs but let
        # the pixel detection decide what's brow vs skin
        pad_x = int(brow_w * 0.15)
        pad_y = int(brow_h * 0.5)

        rx1 = max(0, x_min - pad_x)
        rx2 = min(w, x_max + pad_x)
        ry1 = max(0, y_min - pad_y)
        ry2 = min(h, y_max + pad_y)

        if rx2 - rx1 < 5 or ry2 - ry1 < 5:
            continue

        # Extract the region
        region = result_f[ry1:ry2, rx1:rx2]
        rh, rw = region.shape[:2]

        # Convert region to grayscale
        region_gray = (region[:, :, 0] * 0.114 +
                       region[:, :, 1] * 0.587 +
                       region[:, :, 2] * 0.299)

        # Estimate local skin tone: heavily blurred version = skin baseline
        # Brow hairs are darker than this baseline
        skin_baseline = cv2.GaussianBlur(region_gray, (0, 0),
                                          sigmaX=max(3, brow_w * 0.15))

        # Hair detection: pixels darker than skin baseline by a threshold
        # The difference tells us "how much darker than surrounding skin"
        darkness_diff = skin_baseline - region_gray  # positive = darker than skin

        # Normalize: find the range of darkness in this region
        max_diff = np.percentile(darkness_diff[darkness_diff > 0], 95) if (darkness_diff > 0).any() else 1
        if max_diff < 3:
            continue  # no significant dark pixels found — skip

        # Hair confidence: 0 = same as skin, 1 = as dark as the darkest hair
        hair_conf = np.clip(darkness_diff / max_diff, 0, 1)

        # Threshold: only pixels with meaningful darkness difference
        # This eliminates skin pixels even if they're inside the landmark bbox
        hair_conf[darkness_diff < max_diff * 0.15] = 0

        # Smooth the hair mask slightly so individual pixels don't look speckled
        hair_mask = cv2.GaussianBlur(hair_conf.astype(np.float32), (3, 3), 0)

        # Sample the brow's own dark color from the strongest hair pixels
        strong_hair = hair_mask > 0.5
        if strong_hair.sum() < 3:
            continue

        brow_color = region[strong_hair].mean(axis=0)

        # Create darkened version: blend toward a darker version of brow color
        dark_target = brow_color * (0.4 + 0.4 * (1 - intensity))  # darker at higher intensity
        dark_layer = np.full_like(region, dark_target)

        # Blend amount: hair pixels get darkened, skin pixels untouched
        blend = hair_mask * intensity * 0.65

        blend_3ch = np.stack([blend] * 3, axis=-1)
        darkened_region = region * (1 - blend_3ch) + dark_layer * blend_3ch

        result_f[ry1:ry2, rx1:rx2] = darkened_region

    return np.clip(result_f, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# 3b) Brow Lift — subtle upward warp of the eyebrow arch
# ---------------------------------------------------------------------------

def brow_lift(image, landmarks, intensity=0.3):
    """
    Lift the entire eyebrow region upward in one smooth pass.
    The whole brow (upper edge, lower edge, hair, everything) moves as a
    rigid unit so it keeps its original thickness — no squishing.

    Strongest at the outer tail, lighter at the inner end.
    Uses a single accumulated displacement field (like lip_flip) instead
    of per-point warps, so the profile stays smooth with no waviness.

    intensity: 0 = no change, 1 = maximum lift
    """
    if intensity <= 0:
        return image

    h, w = image.shape[:2]
    face_width = abs(landmarks[234][0] - landmarks[454][0])
    max_lift = face_width * 0.015 * intensity

    map_x, map_y = np.meshgrid(
        np.arange(w, dtype=np.float32),
        np.arange(h, dtype=np.float32),
    )
    total_dy = np.zeros((h, w), dtype=np.float32)

    for upper_idx, lower_idx in [
        (LEFT_EYEBROW_UPPER, LEFT_EYEBROW_LOWER),
        (RIGHT_EYEBROW_UPPER, RIGHT_EYEBROW_LOWER),
    ]:
        upper_pts = np.array([landmarks[i] for i in upper_idx], dtype=float)
        lower_pts = np.array([landmarks[i] for i in lower_idx], dtype=float)
        all_pts = np.concatenate([upper_pts, lower_pts], axis=0)

        # Brow bounding box
        brow_cx = all_pts[:, 0].mean()
        brow_cy = all_pts[:, 1].mean()
        brow_w = all_pts[:, 0].max() - all_pts[:, 0].min()
        brow_h = all_pts[:, 1].max() - all_pts[:, 1].min()

        # Generous elliptical falloff around the whole brow region
        # so the entire brow + surrounding skin lifts as one piece
        rx = brow_w * 0.75  # horizontal radius
        ry = brow_h * 2.0   # vertical radius — tall so skin above/below moves too

        # Elliptical distance from brow center
        dx = (map_x - brow_cx) / max(rx, 1)
        dy = (map_y - brow_cy) / max(ry, 1)
        ellipse_dist = np.sqrt(dx ** 2 + dy ** 2)

        # Smooth falloff: 1 inside the ellipse, fading to 0 outside
        falloff = np.clip(1.0 - ellipse_dist, 0, 1) ** 2

        # Inner-to-outer weight: stronger lift at the outer tail
        # Find inner and outer x extents
        inner_x = upper_pts[0, 0]   # first point = inner end
        outer_x = upper_pts[-1, 0]  # last point = outer end
        brow_span = abs(outer_x - inner_x)
        if brow_span < 1:
            brow_span = 1

        # For each pixel, compute how "outer" it is (0 = inner, 1 = outer)
        if outer_x > inner_x:
            outer_frac = np.clip((map_x - inner_x) / brow_span, 0, 1)
        else:
            outer_frac = np.clip((inner_x - map_x) / brow_span, 0, 1)

        # Weight: 0.15 at inner end, ramps steeply toward outer tail
        # Power curve concentrates the lift at the outer end
        weight = 0.15 + 0.85 * (outer_frac ** 0.6)

        # Upward displacement (positive total_dy → source pixel is below → lifts)
        total_dy += max_lift * weight * falloff

    # Apply the combined warp — shift source y downward to pull image up
    new_map_y = (map_y + total_dy).astype(np.float32)
    result = cv2.remap(image, map_x, new_map_y, cv2.INTER_LINEAR,
                       borderMode=cv2.BORDER_REFLECT)

    return result


# ---------------------------------------------------------------------------
# 3b-ii) Brow tail lift — extra lift for the outer 25 % of each brow
# ---------------------------------------------------------------------------

def brow_tail_lift(image, landmarks, intensity=0.3):
    """
    Additional lift targeting only the outer tail (~25 %) of each eyebrow.
    Uses a small, focused elliptical region centred on the outer brow points
    so it doesn't interfere with the mid-brow.

    intensity: 0 = no change, 1 = maximum tail lift
    """
    if intensity <= 0:
        return image

    h, w = image.shape[:2]
    face_width = abs(landmarks[234][0] - landmarks[454][0])
    max_lift = face_width * 0.018 * intensity   # slightly more aggressive per-pixel

    map_x, map_y = np.meshgrid(
        np.arange(w, dtype=np.float32),
        np.arange(h, dtype=np.float32),
    )
    total_dy = np.zeros((h, w), dtype=np.float32)

    for upper_idx, lower_idx in [
        (LEFT_EYEBROW_UPPER, LEFT_EYEBROW_LOWER),
        (RIGHT_EYEBROW_UPPER, RIGHT_EYEBROW_LOWER),
    ]:
        upper_pts = np.array([landmarks[i] for i in upper_idx], dtype=float)
        lower_pts = np.array([landmarks[i] for i in lower_idx], dtype=float)

        # Full brow span
        inner_x = upper_pts[0, 0]
        outer_x = upper_pts[-1, 0]
        brow_span = abs(outer_x - inner_x)
        if brow_span < 1:
            continue

        # Centre the effect on the outer 25 %
        # Take the last 2 upper + last 2 lower points as the tail region
        tail_upper = upper_pts[-2:]   # last two upper points
        tail_lower = lower_pts[-2:]   # last two lower points
        tail_pts = np.concatenate([tail_upper, tail_lower], axis=0)

        tail_cx = tail_pts[:, 0].mean()
        tail_cy = tail_pts[:, 1].mean()

        # Focused ellipse — covers roughly the outer quarter
        rx = brow_span * 0.30
        ry = abs(upper_pts[:, 1].max() - lower_pts[:, 1].min()) * 2.5

        dx = (map_x - tail_cx) / max(rx, 1)
        dy = (map_y - tail_cy) / max(ry, 1)
        ellipse_dist = np.sqrt(dx ** 2 + dy ** 2)

        falloff = np.clip(1.0 - ellipse_dist, 0, 1) ** 2

        total_dy += max_lift * falloff

    new_map_y = (map_y + total_dy).astype(np.float32)
    result = cv2.remap(image, map_x, new_map_y, cv2.INTER_LINEAR,
                       borderMode=cv2.BORDER_REFLECT)
    return result


# ---------------------------------------------------------------------------
# 3c) Smize — smile with your eyes
# ---------------------------------------------------------------------------

def smize(image, landmarks, intensity=0.3):
    """
    'Smize' — smiling with your eyes.  Two push points per eye on
    the lower lid:
      1. At 20 % along the lower lid from the outer corner
      2. At 5 % from the inner corner
    Both push upward with a broad radius.

    intensity: 0 = no change, 1 = maximum smize
    """
    if intensity <= 0:
        return image

    result = image.copy()

    # Left eye: inner 263, outer 362
    # Lower lid landmarks outer→inner: 362,382,381,380,374,373,390,249,263
    left_lower = [362, 382, 381, 380, 374, 373, 390, 249, 263]
    # Right eye: inner 33, outer 133
    # Lower lid landmarks outer→inner: 133,155,154,153,145,144,163,7,33
    right_lower = [133, 155, 154, 153, 145, 144, 163, 7, 33]

    for lower_indices in [left_lower, right_lower]:
        pts = np.array([landmarks[i] for i in lower_indices], dtype=float)
        outer = pts[0]
        inner = pts[-1]
        eye_w = np.linalg.norm(outer - inner)
        if eye_w < 1:
            continue

        radius = int(eye_w * 0.45)
        max_push = eye_w * 0.018 * intensity

        # Compute cumulative arc length along the lower lid
        diffs = np.diff(pts, axis=0)
        seg_lens = np.linalg.norm(diffs, axis=1)
        cum = np.concatenate([[0], np.cumsum(seg_lens)])
        total_len = cum[-1]
        if total_len < 1:
            continue

        # Interpolate a point at a given fraction along the lid
        def point_at(frac):
            target = frac * total_len
            for j in range(len(cum) - 1):
                if cum[j + 1] >= target:
                    t = (target - cum[j]) / max(cum[j + 1] - cum[j], 1e-6)
                    return pts[j] + t * (pts[j + 1] - pts[j])
            return pts[-1]

        # Push point 1: 20 % from the outer corner
        p1 = point_at(0.20)
        # Push point 2: 5 % from the inner corner (= 95 % from outer)
        p2 = point_at(0.95)

        for pt, weight in [(p1, 1.0), (p2, 0.6)]:
            result = _local_warp(
                result,
                center=(float(pt[0]), float(pt[1])),
                direction=(0, -1),
                radius=radius,
                strength=max_push * weight,
            )

    return result


# ---------------------------------------------------------------------------
# 3d) Lip Flip (upper lip vermilion border lift)
# ---------------------------------------------------------------------------

def lip_flip(image, landmarks, intensity=0.3):
    """
    Push the upper lip vermilion border upward to reveal more of the upper lip.
    Mimics a 'lip flip' — the top edge of the upper lip lifts, making it look fuller.

    Strongest at the center (cupid's bow), fading toward lip corners so
    the corners stay natural.

    Teeth are protected: displacement is blocked below the upper lip bottom edge
    so the teeth/mouth area stays completely untouched.

    intensity: 0 = no change, 1 = maximum lift
    """
    if intensity <= 0:
        return image

    h, w = image.shape[:2]

    # Measure lip width to scale the effect proportionally
    left_corner = np.array(landmarks[LEFT_LIP_CORNER], dtype=float)
    right_corner = np.array(landmarks[RIGHT_LIP_CORNER], dtype=float)
    lip_width = np.linalg.norm(left_corner - right_corner)
    lip_center_x = (left_corner[0] + right_corner[0]) / 2.0

    # Build a teeth protection boundary from the upper lip bottom edge.
    # For each x-coordinate, find the y of the upper lip bottom edge.
    # Any pixel below this line should get zero displacement.
    bottom_pts = get_landmark_points(landmarks, UPPER_LIP_BOTTOM_INDICES).astype(float)

    # Sort by x so we can interpolate
    sorted_idx = np.argsort(bottom_pts[:, 0])
    bx = bottom_pts[sorted_idx, 0]
    by = bottom_pts[sorted_idx, 1]

    # Interpolate the lip bottom y for every x column in the image
    xs = np.arange(w, dtype=np.float32)
    lip_bottom_y = np.interp(xs, bx, by)

    # Create a per-pixel mask: 1.0 above the lip bottom, fading to 0 at and below it
    ys = np.arange(h, dtype=np.float32)
    yy, xx = np.meshgrid(ys, xs, indexing='ij')

    # Push protection boundary upward by a safety margin so warp
    # never reaches the teeth.  Margin scales with lip size.
    safety_margin = lip_width * 0.04          # ~4 % of lip width upward
    lip_bottom_2d = np.broadcast_to(
        (lip_bottom_y - safety_margin)[np.newaxis, :], (h, w)
    )

    # Very tight transition — almost a hard cut at the boundary
    transition = 2.0  # pixels (was 4)
    teeth_shield = np.clip((lip_bottom_2d - yy) / transition, 0, 1)

    # Warp radius and max strength proportional to lip size
    radius = int(lip_width * 0.25)
    max_strength = intensity * lip_width * 0.07

    # Accumulate all warp displacements in one pass for better quality
    total_dy = np.zeros((h, w), dtype=np.float32)
    map_x_base, map_y_base = np.meshgrid(
        np.arange(w, dtype=np.float32),
        np.arange(h, dtype=np.float32)
    )

    for idx in UPPER_LIP_TOP_INDICES:
        pt = np.array(landmarks[idx], dtype=float)

        # Fade toward corners: strongest at center, weakest at edges
        dist_from_center = abs(pt[0] - lip_center_x)
        max_dist = lip_width / 2.0
        center_weight = 1.0 - (min(dist_from_center / max_dist, 1.0)) ** 1.3
        center_weight = max(0.05, center_weight)

        pt_strength = max_strength * center_weight

        # Compute displacement for this warp point
        cx, cy = float(landmarks[idx][0]), float(landmarks[idx][1])
        dist = np.sqrt((map_x_base - cx) ** 2 + (map_y_base - cy) ** 2)
        falloff = np.clip(1.0 - dist / max(radius, 1), 0, 1) ** 2

        # Upward displacement (negative y)
        total_dy += pt_strength * falloff

    # Apply teeth shield: zero out displacement at and below the lip bottom edge
    total_dy = total_dy * teeth_shield

    # Apply the combined warp
    map_x = map_x_base.astype(np.float32)
    map_y = (map_y_base + total_dy).astype(np.float32)  # +dy because we're mapping source, pushing up

    result = cv2.remap(image, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

    return result


# ---------------------------------------------------------------------------
# 3e) Lip Liner — define the upper lip vermilion border
# ---------------------------------------------------------------------------

def lip_liner(image, landmarks, intensity=0.3):
    """
    Draw a smooth, defined line along the upper lip vermilion border to
    make the lip edge more solid and contrasted against the skin above it.

    Instead of following the raw landmarks point-by-point (which can look
    jagged), we fit a smooth spline through the lip top points and draw
    a single clean line, then darken and add contrast along it.

    intensity: 0 = no change, 1 = maximum definition
    """
    if intensity <= 0:
        return image

    h, w = image.shape[:2]

    # Get the upper lip top landmarks and sort by x
    top_pts = get_landmark_points(landmarks, UPPER_LIP_TOP_INDICES).astype(float)
    sorted_idx = np.argsort(top_pts[:, 0])
    tx = top_pts[sorted_idx, 0]
    ty = top_pts[sorted_idx, 1]

    # Lip width for scaling
    left_corner = np.array(landmarks[LEFT_LIP_CORNER], dtype=float)
    right_corner = np.array(landmarks[RIGHT_LIP_CORNER], dtype=float)
    lip_width = np.linalg.norm(left_corner - right_corner)

    # Interpolate a smooth curve with many more points than the landmarks
    # This removes the jaggedness of connecting landmarks directly
    num_smooth = max(50, int(lip_width))
    smooth_x = np.linspace(tx[0], tx[-1], num_smooth)
    smooth_y = np.interp(smooth_x, tx, ty)

    # Further smooth the y values with a small moving average to remove bumps
    kernel_size = max(3, int(num_smooth * 0.08)) | 1
    smooth_y_padded = np.pad(smooth_y, kernel_size // 2, mode='edge')
    kernel = np.ones(kernel_size) / kernel_size
    smooth_y = np.convolve(smooth_y_padded, kernel, mode='valid')[:num_smooth]

    # Build the smooth line as a polyline
    line_pts = np.stack([smooth_x, smooth_y], axis=1).astype(np.int32)

    # Line thickness proportional to lip size
    line_thickness = max(1, int(lip_width * 0.012))

    # Draw the line on a mask
    line_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.polylines(line_mask, [line_pts], isClosed=False,
                  color=255, thickness=line_thickness)

    # Feather slightly for natural blending
    k = max(3, line_thickness * 2) | 1
    line_mask_f = cv2.GaussianBlur(line_mask.astype(np.float32), (k, k), 0) / 255.0

    # Fade toward corners so the liner doesn't extend past the lip ends
    lip_cx = (left_corner[0] + right_corner[0]) / 2.0
    half_w = lip_width / 2.0
    xs_arr = np.arange(w, dtype=np.float32)
    corner_fade = 1.0 - np.clip(
        np.abs(xs_arr - lip_cx) / max(half_w * 0.9, 1), 0, 1
    ) ** 2
    line_mask_f = line_mask_f * corner_fade[np.newaxis, :]

    # Sample the lip edge colour for natural-looking darkening
    mask_bool = line_mask_f > 0.1
    if mask_bool.any():
        edge_pixels = image[mask_bool]
        avg_color = edge_pixels.mean(axis=0)
    else:
        avg_color = np.array([80, 60, 70], dtype=np.float32)

    # Dark target: darker and slightly more saturated version of lip edge
    dark_factor = 0.35 + 0.25 * (1 - intensity)  # darker at higher intensity
    dark_target = (avg_color * dark_factor).astype(np.uint8)
    dark_layer = np.full_like(image, dark_target)

    # Blend the darkened line onto the image
    blend_strength = intensity * 0.6
    darkened = cv2.addWeighted(image, 1.0 - blend_strength, dark_layer, blend_strength, 0)

    # Also add micro-contrast along the line (unsharp mask)
    blur = cv2.GaussianBlur(image, (0, 0), 1.0)
    contrast_boost = intensity * 0.5
    contrasted = cv2.addWeighted(darkened, 1.0 + contrast_boost, blur, -contrast_boost, 0)

    line_mask_f *= intensity
    return _blend(image, contrasted, line_mask_f)


# ---------------------------------------------------------------------------
# 4) Nose Slimming (local warp / liquify)
# ---------------------------------------------------------------------------

def _local_warp(image, center, direction, radius, strength, iris_protect=None):
    """
    Apply a local displacement (liquify-like) warp using cv2.remap.

    center: (x, y) center of the warp
    direction: (dx, dy) direction to push pixels
    radius: radius of effect in pixels
    strength: 0..1 warp strength
    iris_protect: optional (cx, cy, r) — zero out displacement inside this circle
                  to keep the iris round during eye enlargement
    """
    h, w = image.shape[:2]
    map_x, map_y = np.meshgrid(np.arange(w, dtype=np.float32),
                                np.arange(h, dtype=np.float32))

    cx, cy = center
    dx, dy = direction[0] * strength, direction[1] * strength

    # Distance from center for each pixel
    dist_x = map_x - cx
    dist_y = map_y - cy
    dist = np.sqrt(dist_x ** 2 + dist_y ** 2)

    # Smooth falloff within radius
    mask = np.clip(1.0 - dist / radius, 0, 1)
    mask = mask ** 2  # quadratic falloff for smoothness

    # Protect iris: smoothly fade displacement to zero inside the iris circle
    if iris_protect is not None:
        icx, icy, ir = iris_protect
        # Use a slightly larger radius for smooth transition
        protect_r = ir * 1.4
        iris_dist = np.sqrt((map_x - icx) ** 2 + (map_y - icy) ** 2)
        # 0 inside iris core, ramps up to 1 outside protect radius
        iris_shield = np.clip((iris_dist - ir * 0.6) / (protect_r - ir * 0.6), 0, 1)
        mask = mask * iris_shield

    map_x = (map_x - dx * mask).astype(np.float32)
    map_y = (map_y - dy * mask).astype(np.float32)

    return cv2.remap(image, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


def nose_slimming(image, landmarks, intensity=0.3):
    """
    Refine the nose: slim the sides inward + lift the tip upward.
    Keeps a natural shape by using proportional scaling and smooth falloff.

    intensity: 0 = no change, 1 = maximum refinement
    """
    if intensity <= 0:
        return image

    h, w = image.shape[:2]
    result = image.copy()

    nose_bridge = np.array(landmarks[NOSE_BRIDGE_INDEX], dtype=float)
    nose_tip = np.array(landmarks[NOSE_TIP_INDEX], dtype=float)
    left_wing = np.array(landmarks[LEFT_NOSE_WING], dtype=float)
    right_wing = np.array(landmarks[RIGHT_NOSE_WING], dtype=float)

    # Nose dimensions for proportional scaling
    nose_width = np.linalg.norm(left_wing - right_wing)
    nose_height = np.linalg.norm(nose_bridge - nose_tip)

    # --- 1) Slim the sides (inward warp) ---
    slim_radius = int(nose_width * 0.7)
    slim_strength = intensity * nose_width * 0.12

    # Push left wing inward (to the right)
    for idx in NOSE_SIDE_INDICES_LEFT:
        pt = landmarks[idx]
        result = _local_warp(result, pt, (1, 0), slim_radius, slim_strength / len(NOSE_SIDE_INDICES_LEFT))

    # Push right wing inward (to the left)
    for idx in NOSE_SIDE_INDICES_RIGHT:
        pt = landmarks[idx]
        result = _local_warp(result, pt, (-1, 0), slim_radius, slim_strength / len(NOSE_SIDE_INDICES_RIGHT))

    # --- 2) Lift the tip upward ---
    # Warp centered on the nose tip, pushing upward
    # Use a tighter radius so only the tip area moves, not the whole nose
    tip_radius = int(nose_height * 0.4)
    tip_strength = intensity * nose_height * 0.08

    # Main tip lift — straight up
    result = _local_warp(result, landmarks[NOSE_TIP_INDEX], (0, -1), tip_radius, tip_strength)

    # Also gently lift the nostrils (wings) upward to keep the shape balanced
    nostril_strength = tip_strength * 0.4  # gentler than the tip
    nostril_radius = int(nose_width * 0.35)
    result = _local_warp(result, landmarks[LEFT_NOSE_WING], (0, -1), nostril_radius, nostril_strength)
    result = _local_warp(result, landmarks[RIGHT_NOSE_WING], (0, -1), nostril_radius, nostril_strength)

    return result


# ---------------------------------------------------------------------------
# 4b) Nostril Narrowing — push nostrils inward toward the nose tip
# ---------------------------------------------------------------------------

def _nostril_narrow_one_side(image, landmarks, wing_index, intensity):
    """Push a single nostril wing toward the nose tip."""
    if intensity <= 0:
        return image

    nose_tip = np.array(landmarks[NOSE_TIP_INDEX], dtype=float)
    wing = np.array(landmarks[wing_index], dtype=float)
    other_wing = np.array(landmarks[
        RIGHT_NOSE_WING if wing_index == LEFT_NOSE_WING else LEFT_NOSE_WING
    ], dtype=float)

    nose_width = np.linalg.norm(wing - other_wing)
    radius = int(nose_width * 0.4)
    max_shift = intensity * nose_width * 0.08

    toward_tip = nose_tip - wing
    dist = np.linalg.norm(toward_tip)
    if dist < 1:
        return image
    direction = toward_tip / dist

    return _local_warp(
        image,
        center=(float(wing[0]), float(wing[1])),
        direction=(float(direction[0]), float(direction[1])),
        radius=radius,
        strength=max_shift,
    )


def nostril_narrowing_left(image, landmarks, intensity=0.3):
    """Push the left nostril inward toward the nose tip."""
    return _nostril_narrow_one_side(image, landmarks, LEFT_NOSE_WING, intensity)


def nostril_narrowing_right(image, landmarks, intensity=0.3):
    """Push the right nostril inward toward the nose tip."""
    return _nostril_narrow_one_side(image, landmarks, RIGHT_NOSE_WING, intensity)


# ---------------------------------------------------------------------------
# 5) Jaw Sharpening
# ---------------------------------------------------------------------------

def jaw_sharpening(image, landmarks, intensity=0.3):
    """
    Sharpen/slim the jawline with subtle inward warps + optional micro-contrast.

    intensity: 0 = no change, 1 = maximum sharpening
    """
    if intensity <= 0:
        return image

    h, w = image.shape[:2]
    result = image.copy()

    chin = np.array(landmarks[CHIN_INDEX])
    left_jaw = np.array(landmarks[LEFT_JAW_CORNER])
    right_jaw = np.array(landmarks[RIGHT_JAW_CORNER])

    jaw_width = np.linalg.norm(left_jaw - right_jaw)
    face_center_x = (left_jaw[0] + right_jaw[0]) / 2
    face_center_y = (left_jaw[1] + right_jaw[1]) / 2

    radius = int(jaw_width * 0.25)
    strength = intensity * jaw_width * 0.06

    # Warp jaw points inward and slightly upward
    for idx in JAW_INDICES:
        pt = np.array(landmarks[idx])
        # Direction: toward the vertical center line and slightly up
        dir_x = np.sign(face_center_x - pt[0])
        dir_y = -0.3  # slight upward pull
        norm = np.sqrt(dir_x ** 2 + dir_y ** 2)
        if norm > 0:
            dir_x /= norm
            dir_y /= norm
        result = _local_warp(result, tuple(pt), (dir_x, dir_y), radius, strength / len(JAW_INDICES))

    # Optional micro-contrast along the jawline
    if intensity > 0.3:
        # Unsharp mask for local contrast
        blur = cv2.GaussianBlur(result, (0, 0), 3)
        contrast = cv2.addWeighted(result, 1.0 + intensity * 0.3, blur, -intensity * 0.3, 0)

        # Apply only near jaw region
        jaw_pts = get_landmark_points(landmarks, JAW_INDICES)
        jaw_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(jaw_mask, [jaw_pts], 255)
        jaw_mask = cv2.GaussianBlur(jaw_mask.astype(np.float32), (31, 31), 0) / 255.0
        jaw_mask *= intensity * 0.5

        result = _blend(result, contrast, jaw_mask)

    return result


# ---------------------------------------------------------------------------
# 7) Skin Glow — luminous bloom on lighter skin areas
# ---------------------------------------------------------------------------

def skin_glow(image, landmarks, intensity=0.3):
    """
    Add a soft, dewy glow to the skin — strongest on naturally lighter
    areas (forehead, nose bridge, cheekbone highlights, chin) while
    leaving darker areas (shadows, contours) untouched.

    Works by:
    1. Converting to LAB and reading the L (lightness) channel
    2. Building a 'highlight mask' from pixels brighter than the median
       skin lightness — these are the natural highlight zones
    3. Creating a soft bloom (heavily blurred bright layer) and blending
       it into the highlights through the skin mask

    The result is a lit-from-within, dewy look without washing out shadows.

    intensity: 0 = no glow, 1 = maximum glow
    """
    if intensity <= 0:
        return image

    h, w = image.shape[:2]

    # Skin-only mask so glow stays on skin, not eyes/lips/hair
    skin_mask = make_skin_mask(landmarks, image.shape, feather=21)

    # Convert to LAB
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
    L = lab[:, :, 0]

    # Find the median lightness of skin pixels to split highlights vs shadows
    skin_pixels = L[skin_mask > 0.3]
    if len(skin_pixels) < 10:
        return image
    median_L = np.median(skin_pixels)

    # Highlight mask: pixels brighter than median get glow, darker ones don't
    # Smooth ramp so there's no hard edge
    highlight_range = max(15.0, np.std(skin_pixels) * 0.8)
    highlight_mask = np.clip((L - median_L) / highlight_range, 0, 1)

    # Combine: only glow on skin highlights
    glow_mask = highlight_mask * skin_mask

    # Soften the mask for smooth falloff
    glow_mask = cv2.GaussianBlur(glow_mask, (0, 0), sigmaX=max(5, min(h, w) * 0.02))

    # --- Create the bloom layer ---
    # Heavy Gaussian blur of the image = soft dreamy glow
    bloom_sigma = max(10, min(h, w) * 0.03)
    bloom = cv2.GaussianBlur(image, (0, 0), sigmaX=bloom_sigma)

    # Brighten the bloom slightly in LAB (lift lightness)
    bloom_lab = cv2.cvtColor(bloom, cv2.COLOR_BGR2LAB).astype(np.float32)
    lift = 8.0 + intensity * 15.0  # how much brighter
    bloom_lab[:, :, 0] = np.clip(bloom_lab[:, :, 0] + lift, 0, 255)
    # Slightly desaturate for that soft-focus look
    bloom_lab[:, :, 1] = bloom_lab[:, :, 1] * (1.0 - intensity * 0.15)
    bloom_lab[:, :, 2] = bloom_lab[:, :, 2] * (1.0 - intensity * 0.15)
    bloom = cv2.cvtColor(bloom_lab.astype(np.uint8), cv2.COLOR_LAB2BGR)

    # Blend: use screen-like blending for a natural glow
    # Screen blend: result = 1 - (1 - base) * (1 - bloom)
    base_f = image.astype(np.float32) / 255.0
    bloom_f = bloom.astype(np.float32) / 255.0
    screen = 1.0 - (1.0 - base_f) * (1.0 - bloom_f)
    screen = (screen * 255).astype(np.uint8)

    # Apply through the glow mask, scaled by intensity
    glow_mask = glow_mask * intensity * 0.7
    return _blend(image, screen, glow_mask)


# ---------------------------------------------------------------------------
# 8) Spot / Pore Removal — inpaint-style correction using surrounding colour
# ---------------------------------------------------------------------------

def spot_removal(image, landmarks, intensity=0.3):
    """
    Detect dark spots and pores on the skin and blend them away using
    the surrounding skin colour.  Uses a difference-of-Gaussians approach
    to find small dark anomalies, then inpaints them with a local median.

    intensity: 0 = no change, 1 = aggressive spot removal
    """
    if intensity <= 0:
        return image

    # Work on skin-only region
    skin_mask = make_skin_mask(landmarks, image.shape, feather=11)

    # Convert to LAB for lightness-based detection
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    L = lab[:, :, 0].astype(np.float32)

    # Difference-of-Gaussians to find small dark features (pores, spots)
    # Small sigma catches fine pores, larger sigma catches dark spots
    blur_small = cv2.GaussianBlur(L, (0, 0), 1.5)
    blur_large = cv2.GaussianBlur(L, (0, 0), 6.0)

    # Spots are where the local area is darker than the surrounding average
    diff = blur_large - blur_small
    # Threshold: only consider noticeable dark spots
    threshold = 2.0 + (1.0 - intensity) * 4.0  # lower threshold = more aggressive
    spot_mask = np.clip((diff - threshold) / 6.0, 0, 1)

    # Restrict to skin only
    spot_mask = spot_mask * skin_mask

    # Create corrected version using a median filter (preserves edges,
    # replaces spots with surrounding colour)
    ksize = max(3, int(5 + intensity * 6)) | 1  # odd kernel: 5..11
    corrected = cv2.medianBlur(image, ksize)

    # Also try bilateral for smoother result
    corrected2 = cv2.bilateralFilter(image, 7, 40, 40)
    # Blend: median for strong spots, bilateral for fine pores
    corrected = cv2.addWeighted(corrected, 0.5, corrected2, 0.5, 0)

    spot_mask *= intensity
    return _blend(image, corrected, spot_mask)


# ---------------------------------------------------------------------------
# 8) Marionette / Nasolabial Line Lightening
# ---------------------------------------------------------------------------

# Marionette lines run from the nose wings / lip corners downward
# We use landmark geometry to define approximate line paths
def marionette_line_lightening(image, landmarks, intensity=0.3):
    """
    Subtly lighten the nasolabial folds and marionette lines to reduce
    the appearance of deep creases.  Works by detecting the dark crease
    in the expected region and gently raising its lightness toward the
    surrounding skin tone.

    intensity: 0 = no change, 1 = maximum lightening
    """
    if intensity <= 0:
        return image

    h, w = image.shape[:2]
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
    L = lab[:, :, 0]

    # Define the marionette / nasolabial line region using landmarks.
    # Left line: from left nose wing (129) down to below left lip corner (61),
    #            continuing toward left jaw area (177)
    # Right line: from right nose wing (358) down to below right lip corner (291),
    #             continuing toward right jaw area (401)
    left_line_pts = []
    right_line_pts = []

    # Build paths from nose wing → lip corner area → below chin
    # Left side
    left_top = np.array(landmarks[129], dtype=float)     # left nose wing
    left_mid = np.array(landmarks[61], dtype=float)      # left lip corner
    left_bot = np.array(landmarks[172], dtype=float)     # left jaw corner
    # Add a point between lip corner and jaw
    left_mid2 = (left_mid + left_bot) / 2.0

    # Right side
    right_top = np.array(landmarks[358], dtype=float)    # right nose wing
    right_mid = np.array(landmarks[291], dtype=float)    # right lip corner
    right_bot = np.array(landmarks[397], dtype=float)    # right jaw corner
    right_mid2 = (right_mid + right_bot) / 2.0

    # Create a soft mask along each crease line
    line_mask = np.zeros((h, w), dtype=np.float32)

    # Width of the line region scales with face size
    face_width = abs(landmarks[234][0] - landmarks[454][0])  # ear to ear approx
    line_width = max(4, int(face_width * 0.035))

    for pts_list in [
        [left_top, left_mid, left_mid2, left_bot],
        [right_top, right_mid, right_mid2, right_bot],
    ]:
        pts_arr = np.array(pts_list, dtype=np.int32)
        cv2.polylines(line_mask, [pts_arr], isClosed=False,
                      color=1.0, thickness=line_width)

    # Feather the mask
    k = max(line_width * 2, 5) | 1
    line_mask = cv2.GaussianBlur(line_mask, (k, k), 0)
    line_mask = np.clip(line_mask, 0, 1)

    # Also restrict to skin
    skin_mask = make_skin_mask(landmarks, image.shape, feather=11)
    line_mask = line_mask * skin_mask

    # Within the masked region, detect where the crease actually is
    # by finding locally darker pixels (the fold is a shadow)
    local_avg = cv2.GaussianBlur(L, (0, 0), sigmaX=line_width * 1.5)
    darkness = local_avg - L  # positive where pixel is darker than surroundings
    crease_conf = np.clip(darkness / 12.0, 0, 1)

    # Combine: only lighten where we see a crease in the expected region
    final_mask = line_mask * crease_conf * intensity * 0.7

    # Lighten: raise L channel toward surrounding average
    lift_amount = 8.0 + intensity * 12.0  # subtle lift
    lab_out = lab.copy()
    lab_out[:, :, 0] = np.clip(L + final_mask * lift_amount, 0, 255)

    result = cv2.cvtColor(lab_out.astype(np.uint8), cv2.COLOR_LAB2BGR)
    return result


# ---------------------------------------------------------------------------
# Apply all effects
# ---------------------------------------------------------------------------

def apply_effects(image, landmarks, settings):
    """
    Apply all enabled effects to the image.

    settings: dict with keys like:
        {
            "skin_smoothing": {"enabled": True, "intensity": 0.5},
            "eye_whitening": {"enabled": True, "intensity": 0.3},
            "eyebrow_darkening": {"enabled": True, "intensity": 0.3},
            "nose_slimming": {"enabled": True, "intensity": 0.3},
            "jaw_sharpening": {"enabled": True, "intensity": 0.3},
        }
    """
    result = image.copy()

    # Order matters: geometry warps first, then detail effects, skin smoothing last
    effect_funcs = [
        ("eye_elongation", eye_elongation),
        ("eye_enlargement", eye_enlargement),
        ("smize", smize),
        ("lip_flip", lip_flip),
        ("lip_liner", lip_liner),
        ("eyelash_darkening", eyelash_darkening),
        ("lower_lash_darkening", lower_lash_darkening),
        ("eyebrow_darkening", eyebrow_darkening),
        ("brow_lift", brow_lift),
        ("brow_tail_lift", brow_tail_lift),
        ("eye_whitening", eye_whitening),
        ("nose_slimming", nose_slimming),
        ("nostril_narrowing_left", nostril_narrowing_left),
        ("nostril_narrowing_right", nostril_narrowing_right),
        ("jaw_sharpening", jaw_sharpening),
        ("marionette_lines", marionette_line_lightening),
        ("spot_removal", spot_removal),
        ("skin_smoothing", skin_smoothing),
        ("skin_glow", skin_glow),
    ]

    for name, func in effect_funcs:
        cfg = settings.get(name, {})
        if cfg.get("enabled", False):
            intensity = cfg.get("intensity", 0.5)
            logger.info(f"Applying {name} with intensity {intensity:.2f}")
            result = func(result, landmarks, intensity)

    return result
