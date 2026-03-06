"""
Face detection, landmark extraction, and mask generation using MediaPipe FaceMesh.
"""

import os
import cv2
import numpy as np
import mediapipe as mp
import logging

logger = logging.getLogger(__name__)

MODEL_PATH = os.path.join(os.path.dirname(__file__), "face_landmarker_v2_with_blendshapes.task")

# MediaPipe FaceMesh landmark indices for the face oval (outer boundary)
FACE_OVAL_INDICES = [
    10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
    397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
    172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109
]

# Eye white (sclera) landmark indices
# Left eye upper/lower boundary
LEFT_EYE_INDICES = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
# Right eye upper/lower boundary
RIGHT_EYE_INDICES = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]

# Iris indices (to exclude from whitening)
LEFT_IRIS_INDICES = [474, 475, 476, 477]
RIGHT_IRIS_INDICES = [469, 470, 471, 472]

# Eyebrow landmark indices
# Left eyebrow upper edge (from inner to outer)
LEFT_EYEBROW_UPPER = [276, 283, 282, 295, 285]
# Left eyebrow lower edge (from inner to outer)
LEFT_EYEBROW_LOWER = [300, 293, 334, 296, 336]
# Right eyebrow upper edge (from inner to outer)
RIGHT_EYEBROW_UPPER = [46, 53, 52, 65, 55]
# Right eyebrow lower edge (from inner to outer)
RIGHT_EYEBROW_LOWER = [70, 63, 105, 66, 107]

# Lip outline indices (to exclude from skin smoothing)
LIPS_OUTER_INDICES = [
    61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291,
    409, 270, 269, 267, 0, 37, 39, 40, 185,
]

# Upper lip — top edge (vermilion border, the line you see)
# Ordered left corner → cupid's bow → right corner
UPPER_LIP_TOP_INDICES = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291]

# Upper lip — bottom edge (where upper lip meets teeth/lower lip)
# This is the boundary below which we must NOT warp (teeth protection)
UPPER_LIP_BOTTOM_INDICES = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291]

# Lip corner indices (to fade the warp toward edges)
LEFT_LIP_CORNER = 61
RIGHT_LIP_CORNER = 291

# Upper eyelid edge indices (for eyelash darkening)
LEFT_UPPER_EYELID = [362, 398, 384, 385, 386, 387, 388, 466, 263]
RIGHT_UPPER_EYELID = [33, 246, 161, 160, 159, 158, 157, 173, 133]

# Lower eyelid edge indices (for subtle lower lash line)
LEFT_LOWER_EYELID = [263, 249, 390, 373, 374, 380, 381, 382, 362]
RIGHT_LOWER_EYELID = [33, 7, 163, 144, 145, 153, 154, 155, 133]

# Nose landmarks for slimming
NOSE_BRIDGE_INDEX = 6
NOSE_TIP_INDEX = 1
LEFT_NOSE_WING = 129
RIGHT_NOSE_WING = 358
NOSE_SIDE_INDICES_LEFT = [48, 115, 220, 45, 4]
NOSE_SIDE_INDICES_RIGHT = [278, 344, 440, 275, 4]

# Jaw landmarks
LEFT_JAW_CORNER = 172
RIGHT_JAW_CORNER = 397
JAW_INDICES = [172, 136, 150, 149, 176, 148, 152, 377, 400, 378, 379, 365, 397]
CHIN_INDEX = 152


class FaceDetector:
    """Handles face detection and landmark extraction using MediaPipe FaceLandmarker (tasks API)."""

    def __init__(self):
        base_options = mp.tasks.BaseOptions(model_asset_path=MODEL_PATH)
        options = mp.tasks.vision.FaceLandmarkerOptions(
            base_options=base_options,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
        )
        self.landmarker = mp.tasks.vision.FaceLandmarker.create_from_options(options)

    def detect(self, image: np.ndarray):
        """
        Detect face landmarks in an image.

        Args:
            image: BGR image (OpenCV format)

        Returns:
            List of (x, y) landmark coordinates in pixel space, or None if no face found.
        """
        h, w = image.shape[:2]
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        results = self.landmarker.detect(mp_image)

        if not results.face_landmarks:
            logger.warning("No face detected in the image.")
            return None

        face = results.face_landmarks[0]
        landmarks = []
        for lm in face:
            landmarks.append((int(lm.x * w), int(lm.y * h)))

        logger.info(f"Detected face with {len(landmarks)} landmarks.")
        return landmarks


def get_landmark_points(landmarks, indices):
    """Extract specific landmark points by indices."""
    return np.array([landmarks[i] for i in indices], dtype=np.int32)


def make_polygon_mask(shape, points, feather=15):
    """
    Create a feathered mask from a polygon defined by points.

    Args:
        shape: (h, w) of the output mask
        points: Nx2 array of polygon vertices
        feather: Gaussian blur kernel size for feathering

    Returns:
        Float32 mask in [0, 1]
    """
    mask = np.zeros(shape[:2], dtype=np.uint8)
    cv2.fillConvexPoly(mask, points, 255)
    if feather > 0:
        k = feather | 1  # ensure odd
        mask = cv2.GaussianBlur(mask.astype(np.float32), (k, k), 0)
    else:
        mask = mask.astype(np.float32)
    return mask / 255.0


def make_face_oval_mask(landmarks, shape, feather=31):
    """Create a feathered face oval mask."""
    pts = get_landmark_points(landmarks, FACE_OVAL_INDICES)
    return make_polygon_mask(shape, pts, feather)


def make_skin_mask(landmarks, shape, feather=31):
    """
    Create a mask for skin-only areas: face oval minus eyes, eyebrows, lips.
    Also uses HSV skin color detection to avoid smoothing non-skin areas.
    """
    h, w = shape[:2]

    # Start with face oval
    mask = make_face_oval_mask(landmarks, shape, feather=feather)

    # Subtract eye regions (with padding)
    for indices in [LEFT_EYE_INDICES, RIGHT_EYE_INDICES]:
        pts = get_landmark_points(landmarks, indices)
        eye_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(eye_mask, [pts], 255)
        # Dilate to add padding around eyes
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        eye_mask = cv2.dilate(eye_mask, kernel)
        mask = mask * (1.0 - eye_mask.astype(np.float32) / 255.0)

    # Subtract upper eyelid / lash line area
    for indices in [LEFT_UPPER_EYELID, RIGHT_UPPER_EYELID]:
        pts = get_landmark_points(landmarks, indices)
        lash_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.polylines(lash_mask, [pts], False, 255, max(4, int(min(h, w) / 200)))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        lash_mask = cv2.dilate(lash_mask, kernel)
        mask = mask * (1.0 - lash_mask.astype(np.float32) / 255.0)

    # Subtract eyebrow regions
    for upper, lower in [(LEFT_EYEBROW_UPPER, LEFT_EYEBROW_LOWER),
                         (RIGHT_EYEBROW_UPPER, RIGHT_EYEBROW_LOWER)]:
        upper_pts = get_landmark_points(landmarks, upper)
        lower_pts = get_landmark_points(landmarks, lower)
        brow_poly = np.concatenate([upper_pts, lower_pts[::-1]], axis=0)
        brow_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(brow_mask, [brow_poly], 255)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        brow_mask = cv2.dilate(brow_mask, kernel)
        mask = mask * (1.0 - brow_mask.astype(np.float32) / 255.0)

    # Subtract lips
    lip_pts = get_landmark_points(landmarks, LIPS_OUTER_INDICES)
    lip_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(lip_mask, [lip_pts], 255)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    lip_mask = cv2.dilate(lip_mask, kernel)
    mask = mask * (1.0 - lip_mask.astype(np.float32) / 255.0)

    # Feather the final mask
    k = feather | 1
    mask = cv2.GaussianBlur(mask.astype(np.float32), (k, k), 0)
    return np.clip(mask, 0, 1)


def make_eye_white_masks(landmarks, shape, feather=7):
    """
    Create masks for the white areas of both eyes (excluding iris).

    Returns:
        (left_mask, right_mask) as float32 in [0, 1]
    """
    h, w = shape[:2]

    # Left eye region
    left_pts = get_landmark_points(landmarks, LEFT_EYE_INDICES)
    left_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(left_mask, [left_pts], 255)

    # Exclude left iris
    left_iris_pts = get_landmark_points(landmarks, LEFT_IRIS_INDICES)
    iris_center = left_iris_pts.mean(axis=0).astype(int)
    iris_radius = int(np.linalg.norm(left_iris_pts[0] - left_iris_pts[2]) * 0.6)
    cv2.circle(left_mask, tuple(iris_center), iris_radius, 0, -1)

    # Right eye region
    right_pts = get_landmark_points(landmarks, RIGHT_EYE_INDICES)
    right_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(right_mask, [right_pts], 255)

    # Exclude right iris
    right_iris_pts = get_landmark_points(landmarks, RIGHT_IRIS_INDICES)
    iris_center = right_iris_pts.mean(axis=0).astype(int)
    iris_radius = int(np.linalg.norm(right_iris_pts[0] - right_iris_pts[2]) * 0.6)
    cv2.circle(right_mask, tuple(iris_center), iris_radius, 0, -1)

    k = feather | 1
    left_mask = cv2.GaussianBlur(left_mask.astype(np.float32), (k, k), 0) / 255.0
    right_mask = cv2.GaussianBlur(right_mask.astype(np.float32), (k, k), 0) / 255.0

    return left_mask, right_mask


def _make_full_brow_polygon(landmarks, upper_indices, lower_indices):
    """Build a closed polygon for the full eyebrow using upper and lower edges."""
    upper_pts = get_landmark_points(landmarks, upper_indices)
    lower_pts = get_landmark_points(landmarks, lower_indices)
    return np.concatenate([upper_pts, lower_pts[::-1]], axis=0)


def make_full_eyebrow_masks(landmarks, shape, feather=0):
    """
    Create masks covering the full eyebrow area.
    Hard-edged by default (feather=0) so darkening stays strictly inside the brow shape.

    Returns:
        (left_mask, right_mask) as float32 in [0, 1]
    """
    h, w = shape[:2]

    left_poly = _make_full_brow_polygon(landmarks, LEFT_EYEBROW_UPPER, LEFT_EYEBROW_LOWER)
    left_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(left_mask, [left_poly], 255)

    right_poly = _make_full_brow_polygon(landmarks, RIGHT_EYEBROW_UPPER, RIGHT_EYEBROW_LOWER)
    right_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(right_mask, [right_poly], 255)

    if feather > 0:
        k = feather | 1
        # Blur then re-clip to the hard polygon so nothing leaks outside
        left_blurred = cv2.GaussianBlur(left_mask.astype(np.float32), (k, k), 0) / 255.0
        right_blurred = cv2.GaussianBlur(right_mask.astype(np.float32), (k, k), 0) / 255.0
        # Hard clip: zero out anything outside the original polygon
        left_hard = left_mask.astype(np.float32) / 255.0
        right_hard = right_mask.astype(np.float32) / 255.0
        left_mask = np.minimum(left_blurred, left_hard)
        right_mask = np.minimum(right_blurred, right_hard)
    else:
        left_mask = left_mask.astype(np.float32) / 255.0
        right_mask = right_mask.astype(np.float32) / 255.0

    return left_mask, right_mask


def _make_brow_outer_polygon(landmarks, upper_indices, lower_indices):
    """
    Build a closed polygon for the outer third of an eyebrow
    using upper and lower edge landmarks.
    """
    # Take the outer 2 points (last 2 in each list)
    upper_pts = get_landmark_points(landmarks, upper_indices[-2:])
    lower_pts = get_landmark_points(landmarks, lower_indices[-2:])
    # Form polygon: upper outer points forward, lower outer points reversed
    return np.concatenate([upper_pts, lower_pts[::-1]], axis=0)


def make_outer_eyebrow_masks(landmarks, shape, feather=9):
    """
    Create masks for the outer third of each eyebrow.

    Returns:
        (left_mask, right_mask) as float32 in [0, 1]
    """
    h, w = shape[:2]

    left_poly = _make_brow_outer_polygon(landmarks, LEFT_EYEBROW_UPPER, LEFT_EYEBROW_LOWER)
    left_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(left_mask, [left_poly], 255)

    right_poly = _make_brow_outer_polygon(landmarks, RIGHT_EYEBROW_UPPER, RIGHT_EYEBROW_LOWER)
    right_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(right_mask, [right_poly], 255)

    k = feather | 1
    left_mask = cv2.GaussianBlur(left_mask.astype(np.float32), (k, k), 0) / 255.0
    right_mask = cv2.GaussianBlur(right_mask.astype(np.float32), (k, k), 0) / 255.0

    return left_mask, right_mask


def _draw_tapered_lash_line(mask, pts, max_thickness, inner_is_first):
    """
    Draw a lash line that tapers: very thin / faded at the inner corner
    (near nose, where lashes are sparse), full thickness at the outer corner.

    inner_is_first: True if pts[0] is the inner corner (near nose).
                    For the left eye the inner corner is the LAST point,
                    for the right eye it's the FIRST point.

    The inner ~35 % of the line fades in opacity (brightness) rather than
    just shrinking to 1 px, so it becomes nearly invisible at the very
    inner corner — matching real lash density.
    """
    n = len(pts)
    for i in range(n - 1):
        # t goes 0..1 from inner corner to outer corner
        if inner_is_first:
            t = i / max(n - 1, 1)
        else:
            t = 1.0 - i / max(n - 1, 1)

        # Inner 40 %: skip entirely (no real lashes there)
        if t < 0.40:
            continue

        # Thickness ramp: 1 px until 55 %, then ramps up to max
        if t < 0.55:
            seg_thickness = 1
        else:
            ramp = (t - 0.55) / 0.45          # 0..1 over the outer 45 %
            seg_thickness = max(1, int(1 + ramp * (max_thickness - 1)))

        # Opacity ramp: 40-60 % zone fades in gently from 0 → 255
        if t < 0.60:
            opacity = int(((t - 0.40) / 0.20) ** 2 * 255)  # quadratic fade-in
        else:
            opacity = 255

        p1 = tuple(pts[i])
        p2 = tuple(pts[i + 1])
        cv2.line(mask, p1, p2, opacity, seg_thickness)


def make_upper_eyelid_masks(landmarks, shape, thickness=3, feather=5):
    """
    Create masks along the upper eyelid edge for eyelash darkening.
    Tapered: thin at inner corner (near nose), thick at outer corner.

    Thickness is scaled per-eye based on that eye's apparent width,
    so a face angled to the camera gets balanced eyeliner on both sides.

    Returns:
        (left_mask, right_mask) as float32 in [0, 1]
    """
    h, w = shape[:2]

    # Measure each eye's width to scale thickness independently.
    # Left eye: inner corner 263, outer corner 362
    left_inner = np.array(landmarks[263], dtype=float)
    left_outer = np.array(landmarks[362], dtype=float)
    left_width = np.linalg.norm(left_outer - left_inner)

    # Right eye: inner corner 33, outer corner 133
    right_inner = np.array(landmarks[33], dtype=float)
    right_outer = np.array(landmarks[133], dtype=float)
    right_width = np.linalg.norm(right_outer - right_inner)

    # Use the average eye width as the reference; scale each eye relative to it
    avg_width = (left_width + right_width) / 2.0
    if avg_width < 1:
        avg_width = 1

    left_thickness = max(1, int(round(thickness * (left_width / avg_width))))
    right_thickness = max(1, int(round(thickness * (right_width / avg_width))))

    # Left eye: LEFT_UPPER_EYELID goes from outer (362) to inner (263)
    # so inner_is_first=False (inner is last)
    left_pts = get_landmark_points(landmarks, LEFT_UPPER_EYELID)
    left_mask = np.zeros((h, w), dtype=np.uint8)
    _draw_tapered_lash_line(left_mask, left_pts, left_thickness, inner_is_first=False)

    # Right eye: RIGHT_UPPER_EYELID goes from inner (33) to outer (133)
    # so inner_is_first=True
    right_pts = get_landmark_points(landmarks, RIGHT_UPPER_EYELID)
    right_mask = np.zeros((h, w), dtype=np.uint8)
    _draw_tapered_lash_line(right_mask, right_pts, right_thickness, inner_is_first=True)

    k = feather | 1
    left_mask = cv2.GaussianBlur(left_mask.astype(np.float32), (k, k), 0) / 255.0
    right_mask = cv2.GaussianBlur(right_mask.astype(np.float32), (k, k), 0) / 255.0

    return left_mask, right_mask


def make_lower_eyelid_masks(landmarks, shape, thickness=2, feather=3):
    """
    Create masks along the lower eyelid edge for subtle lower lash line.
    Tapered: thinnest at inner corner (near nose), slightly thicker at outer.
    Thickness is scaled per-eye to match perspective.

    Returns:
        (left_mask, right_mask) as float32 in [0, 1]
    """
    h, w = shape[:2]

    # Measure each eye's width for per-eye scaling
    left_width = np.linalg.norm(
        np.array(landmarks[362], dtype=float) - np.array(landmarks[263], dtype=float))
    right_width = np.linalg.norm(
        np.array(landmarks[133], dtype=float) - np.array(landmarks[33], dtype=float))
    avg_width = max((left_width + right_width) / 2.0, 1)

    left_thickness = max(1, int(round(thickness * (left_width / avg_width))))
    right_thickness = max(1, int(round(thickness * (right_width / avg_width))))

    # Left eye lower lid: goes from inner (263) to outer (362)
    left_pts = get_landmark_points(landmarks, LEFT_LOWER_EYELID)
    left_mask = np.zeros((h, w), dtype=np.uint8)
    _draw_tapered_lash_line(left_mask, left_pts, left_thickness, inner_is_first=True)

    # Right eye lower lid: goes from inner (33) to outer (133)
    right_pts = get_landmark_points(landmarks, RIGHT_LOWER_EYELID)
    right_mask = np.zeros((h, w), dtype=np.uint8)
    _draw_tapered_lash_line(right_mask, right_pts, right_thickness, inner_is_first=True)

    k = feather | 1
    left_mask = cv2.GaussianBlur(left_mask.astype(np.float32), (k, k), 0) / 255.0
    right_mask = cv2.GaussianBlur(right_mask.astype(np.float32), (k, k), 0) / 255.0

    return left_mask, right_mask


def draw_debug_overlay(image, landmarks):
    """
    Draw landmark points, face oval, eye regions, and brow regions on the image.
    Useful for debugging mask alignment.
    """
    overlay = image.copy()
    h, w = image.shape[:2]

    # Draw all landmarks as tiny dots
    for i, (x, y) in enumerate(landmarks):
        cv2.circle(overlay, (x, y), 1, (0, 255, 0), -1)

    # Face oval in blue
    oval_pts = get_landmark_points(landmarks, FACE_OVAL_INDICES)
    cv2.polylines(overlay, [oval_pts], True, (255, 0, 0), 1)

    # Eyes in cyan
    for indices in [LEFT_EYE_INDICES, RIGHT_EYE_INDICES]:
        pts = get_landmark_points(landmarks, indices)
        cv2.polylines(overlay, [pts], True, (255, 255, 0), 1)

    # Iris in magenta
    for indices in [LEFT_IRIS_INDICES, RIGHT_IRIS_INDICES]:
        pts = get_landmark_points(landmarks, indices)
        center = pts.mean(axis=0).astype(int)
        radius = int(np.linalg.norm(pts[0] - pts[2]) * 0.6)
        cv2.circle(overlay, tuple(center), radius, (255, 0, 255), 1)

    # Eyebrows in yellow
    for indices in [LEFT_EYEBROW_UPPER, RIGHT_EYEBROW_UPPER]:
        pts = get_landmark_points(landmarks, indices)
        cv2.polylines(overlay, [pts], False, (0, 255, 255), 2)

    # Nose landmarks in red
    for idx in NOSE_SIDE_INDICES_LEFT + NOSE_SIDE_INDICES_RIGHT:
        x, y = landmarks[idx]
        cv2.circle(overlay, (x, y), 3, (0, 0, 255), -1)

    # Jaw in green
    jaw_pts = get_landmark_points(landmarks, JAW_INDICES)
    cv2.polylines(overlay, [jaw_pts], False, (0, 200, 0), 2)

    # Label key landmark indices
    for idx in [NOSE_TIP_INDEX, NOSE_BRIDGE_INDEX, LEFT_NOSE_WING, RIGHT_NOSE_WING,
                LEFT_JAW_CORNER, RIGHT_JAW_CORNER, CHIN_INDEX]:
        x, y = landmarks[idx]
        cv2.putText(overlay, str(idx), (x + 3, y - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)

    return overlay
