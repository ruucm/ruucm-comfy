import numpy as np
import torch
from insightface.app import FaceAnalysis

# Singleton face analyzer to avoid reloading on every call
_app = None

def _get_app():
    global _app
    if _app is None:
        _app = FaceAnalysis(name="buffalo_l", providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
        _app.prepare(ctx_id=0, det_size=(640, 640))
    return _app


def _measure_gaze(img_np):
    """
    Measure gaze offset for a BGR numpy image.
    Returns normalized horizontal offset per eye:
      positive = iris shifted right relative to eye socket center
      negative = iris shifted left
    Normalized by eye socket width so it's resolution-independent.
    """
    app = _get_app()
    faces = app.get(img_np)
    if not faces:
        return 0.0, 0.0

    face = faces[0]
    lm = face.landmark_2d_106

    # Eye socket horizontal bounds from 106-point landmarks
    # Left eye: outer corner ~ index 35, inner corner ~ index 39
    # Right eye: inner corner ~ index 89, outer corner ~ index 93
    left_outer_x = lm[35][0]
    left_inner_x = lm[39][0]
    left_center_x = (left_outer_x + left_inner_x) / 2.0
    left_width = left_inner_x - left_outer_x

    right_inner_x = lm[89][0]
    right_outer_x = lm[93][0]
    right_center_x = (right_inner_x + right_outer_x) / 2.0
    right_width = right_outer_x - right_inner_x

    # 5-point landmarks: [left_eye, right_eye, nose, left_mouth, right_mouth]
    left_iris_x = face.kps[0][0]
    right_iris_x = face.kps[1][0]

    # Normalized offset (iris position relative to eye center / eye width)
    left_offset = (left_iris_x - left_center_x) / left_width if left_width > 0 else 0.0
    right_offset = (right_iris_x - right_center_x) / right_width if right_width > 0 else 0.0

    return float(left_offset), float(right_offset)


def _comfy_image_to_cv2(image_tensor):
    """Convert ComfyUI IMAGE tensor (B,H,W,C float 0-1 RGB) to cv2 BGR numpy."""
    img = image_tensor[0].cpu().numpy()  # (H, W, C), float 0-1, RGB
    img = (img * 255).clip(0, 255).astype(np.uint8)
    return img[:, :, ::-1].copy()  # RGB -> BGR


class GazeOffsetCalculator:
    """
    Calculates the pupil_x / pupil_y offset needed by ExpressionEditor
    to match the gaze direction of a source image.

    Connect outputs directly to ExpressionEditor's pupil_x and pupil_y inputs.
    """

    # ExpressionEditor pupil_x has a non-linear, saturating response.
    # Empirical calibration from measurements:
    #   pupil_x=-5  -> ~0.0106 normalized shift
    #   pupil_x=-15 -> ~0.0056 normalized shift (saturates)
    # We use a moderate scale factor and let users fine-tune with the multiplier.
    SCALE_FACTOR = 250.0

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "sample_image": ("IMAGE",),
                "source_image": ("IMAGE",),
                "strength": ("FLOAT", {
                    "default": 1.5,
                    "min": 0.0,
                    "max": 5.0,
                    "step": 0.1,
                    "tooltip": "Multiplier for the calculated offset. "
                               "1.0 = exact match attempt. "
                               "1.5 = overshoot to compensate for ExpressionEditor saturation. "
                               "Increase if gaze change is too weak.",
                }),
            },
        }

    RETURN_TYPES = ("FLOAT", "FLOAT")
    RETURN_NAMES = ("pupil_x", "pupil_y")
    FUNCTION = "calculate"
    CATEGORY = "face/gaze"
    DESCRIPTION = (
        "Analyzes gaze direction in both images using InsightFace, "
        "then calculates the pupil_x/pupil_y values needed by ExpressionEditor "
        "to shift the sample's gaze to match the source."
    )

    def calculate(self, sample_image, source_image, strength):
        sample_cv2 = _comfy_image_to_cv2(sample_image)
        source_cv2 = _comfy_image_to_cv2(source_image)

        sample_left, sample_right = _measure_gaze(sample_cv2)
        source_left, source_right = _measure_gaze(source_cv2)

        # Horizontal delta needed (average of both eyes)
        delta_left = source_left - sample_left
        delta_right = source_right - sample_right
        delta_x = (delta_left + delta_right) / 2.0

        # Convert normalized offset to ExpressionEditor pupil_x scale
        pupil_x = delta_x * self.SCALE_FACTOR * strength

        # Clamp to ExpressionEditor range
        pupil_x = max(-15.0, min(15.0, pupil_x))

        # pupil_y: not enough data to calibrate, output 0 for now
        pupil_y = 0.0

        print(f"[GazeOffset] sample L={sample_left:+.4f} R={sample_right:+.4f}")
        print(f"[GazeOffset] source L={source_left:+.4f} R={source_right:+.4f}")
        print(f"[GazeOffset] delta_x={delta_x:+.4f} -> pupil_x={pupil_x:+.1f} (strength={strength})")

        return (pupil_x, pupil_y)


NODE_CLASS_MAPPINGS = {
    "GazeOffsetCalculator": GazeOffsetCalculator,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GazeOffsetCalculator": "Gaze Offset Calculator",
}
