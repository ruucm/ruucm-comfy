import base64
import io
import json
import os
import urllib.request
from datetime import datetime
import numpy as np
import torch
from PIL import Image, ImageFilter

_DEBUG_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "debug_output")

MODELS = {
    "Nano Banana 2 (Flash)": "gemini-3.1-flash-image-preview",
    "Nano Banana Pro": "gemini-3-pro-image-preview",
}

QUALITY_OPTIONS = {
    "original": None,
    "512": 512,
    "1024": 1024,
    "2048": 2048,
}

_face_app = None


def _get_face_app():
    global _face_app
    if _face_app is None:
        from insightface.app import FaceAnalysis
        _face_app = FaceAnalysis(
            name="buffalo_l",
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )
        _face_app.prepare(ctx_id=0, det_size=(640, 640))
    return _face_app


def _tensor_to_pil(image_tensor):
    img = image_tensor[0].cpu().numpy()
    img = (img * 255).clip(0, 255).astype(np.uint8)
    return Image.fromarray(img, "RGB")


def _pil_to_tensor(pil_image):
    img = np.array(pil_image.convert("RGB")).astype(np.float32) / 255.0
    return torch.from_numpy(img).unsqueeze(0)


def _pil_to_base64_full(pil_image, fmt="PNG"):
    buf = io.BytesIO()
    pil_image.save(buf, format=fmt)
    return base64.b64encode(buf.getvalue()).decode()


def _crop_face_for_edit(pil_image):
    """Crop face/eye region for Nano Banana editing."""
    import cv2

    app = _get_face_app()
    img_cv2 = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    faces = app.get(img_cv2)

    if not faces:
        print("[EyeMatch] No face detected, using full image")
        return pil_image, None

    face = faces[0]
    lm = face.landmark_2d_106
    w, h = pil_image.size

    left_eye_pts = lm[33:43]
    right_eye_pts = lm[87:97]
    all_eye_pts = np.vstack([left_eye_pts, right_eye_pts])
    x_min, y_min = all_eye_pts.min(axis=0)
    x_max, y_max = all_eye_pts.max(axis=0)

    eye_w = x_max - x_min
    eye_h = y_max - y_min
    cx = (x_min + x_max) / 2
    cy = (y_min + y_max) / 2
    size = max(eye_w, eye_h) * 2.5
    half = size / 2

    crop_x1 = max(0, int(cx - half * 1.2))
    crop_y1 = max(0, int(cy - half))
    crop_x2 = min(w, int(cx + half * 1.2))
    crop_y2 = min(h, int(cy + half))

    crop = pil_image.crop((crop_x1, crop_y1, crop_x2, crop_y2))
    return crop, (crop_x1, crop_y1, crop_x2, crop_y2)


def _paste_crop_back(original_pil, edited_crop, crop_coords, blur_radius=30):
    """Paste edited crop back onto original with elliptical soft mask."""
    x1, y1, x2, y2 = crop_coords
    crop_w, crop_h = x2 - x1, y2 - y1

    edited_crop = edited_crop.resize((crop_w, crop_h), Image.LANCZOS)

    mask_np = np.zeros((crop_h, crop_w), dtype=np.float32)
    cy_m, cx_m = crop_h / 2, crop_w / 2
    ry, rx = crop_h * 0.35, crop_w * 0.35

    Y, X = np.ogrid[:crop_h, :crop_w]
    dist = ((X - cx_m) / rx) ** 2 + ((Y - cy_m) / ry) ** 2
    mask_np[dist <= 1.0] = 255.0
    falloff = (dist > 1.0) & (dist < 1.8)
    mask_np[falloff] = 255.0 * (1.0 - (dist[falloff] - 1.0) / 0.8)

    mask = Image.fromarray(mask_np.clip(0, 255).astype(np.uint8), 'L')
    mask = mask.filter(ImageFilter.GaussianBlur(blur_radius))

    result = original_pil.copy()
    result.paste(edited_crop, (x1, y1), mask)
    return result, mask


def _detect_gaze_direction(pil_image):
    """Detect gaze direction using insightface landmarks. Returns human-readable description."""
    import cv2

    app = _get_face_app()
    img_cv2 = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    faces = app.get(img_cv2)

    if not faces:
        return "directly at the camera"

    face = faces[0]
    lm = face.landmark_2d_106

    left_outer_x = lm[35][0]
    left_inner_x = lm[39][0]
    left_center_x = (left_outer_x + left_inner_x) / 2.0
    left_width = left_inner_x - left_outer_x

    right_inner_x = lm[89][0]
    right_outer_x = lm[93][0]
    right_center_x = (right_inner_x + right_outer_x) / 2.0
    right_width = right_outer_x - right_inner_x

    left_iris_x = face.kps[0][0]
    right_iris_x = face.kps[1][0]

    h_left = (left_iris_x - left_center_x) / left_width if left_width > 0 else 0.0
    h_right = (right_iris_x - right_center_x) / right_width if right_width > 0 else 0.0
    h_avg = (h_left + h_right) / 2.0

    if h_avg < -0.08:
        return "to their left"
    elif h_avg < -0.03:
        return "slightly to their left"
    elif h_avg > 0.08:
        return "to their right"
    elif h_avg > 0.03:
        return "slightly to their right"
    else:
        return "directly at the camera"


class NanoBananaEyeMatch:
    """
    Always edits the source image's eyes to match the target image's gaze direction.
    No comparison - always edits. Crops face region only for Nano Banana.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "target_image": ("IMAGE",),
                "api_key": ("STRING", {
                    "default": "",
                    "tooltip": "Google AI API key (from aistudio.google.com)",
                }),
                "model": (list(MODELS.keys()), {
                    "default": "Nano Banana 2 (Flash)",
                }),
                "output_quality": (list(QUALITY_OPTIONS.keys()), {
                    "default": "original",
                }),
                "blend_radius": ("INT", {
                    "default": 30,
                    "min": 1,
                    "max": 100,
                    "step": 1,
                    "tooltip": "Gaussian blur radius for blending.",
                }),
                "save_debug": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Save intermediate images to ruucm-comfy/debug_output/",
                }),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING", "STRING",)
    RETURN_NAMES = ("edited_image", "used_prompt", "debug_info",)
    FUNCTION = "edit"
    CATEGORY = "image/ai"
    DESCRIPTION = (
        "Always edits the source image's eye gaze to match the target image. "
        "No comparison step - always applies the edit. "
        "Only the face/eye region is sent to Nano Banana to prevent other changes."
    )

    def _save_debug(self, name, pil_img=None, text=None):
        if not self._save_debug_enabled:
            return
        os.makedirs(self._debug_session_dir, exist_ok=True)
        if pil_img is not None:
            path = os.path.join(self._debug_session_dir, f"{name}.png")
            pil_img.save(path)
        if text is not None:
            path = os.path.join(self._debug_session_dir, f"{name}.txt")
            with open(path, "w", encoding="utf-8") as f:
                f.write(text)

    def edit(self, image, target_image, api_key, model, output_quality, blend_radius, save_debug):
        if not api_key.strip():
            raise ValueError("API key is required. Get one from aistudio.google.com")

        self._save_debug_enabled = save_debug
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._debug_session_dir = os.path.join(_DEBUG_DIR, timestamp)

        pil_img = _tensor_to_pil(image)
        target_pil = _tensor_to_pil(target_image)
        original_size = pil_img.size
        debug_lines = []

        self._save_debug("00_input", pil_img)
        self._save_debug("00_target", target_pil)

        # Detect target gaze direction
        target_gaze = _detect_gaze_direction(target_pil)
        debug_lines.append(f"target_gaze: {target_gaze}")

        final_prompt = (
            f"Edit only the eyes in this photo so that both pupils look {target_gaze}. "
            f"Do NOT change anything else - keep the face, hair, skin, clothing, "
            f"background, lighting, and all other details exactly the same. "
            f"Only adjust the iris/pupil position."
        )
        debug_lines.append(f"prompt: {final_prompt}")

        # Crop face region
        face_crop, crop_coords = _crop_face_for_edit(pil_img)
        self._save_debug("00_face_crop", face_crop)
        if crop_coords:
            debug_lines.append(f"face_crop: ({crop_coords[0]},{crop_coords[1]})->({crop_coords[2]},{crop_coords[3]})")

        # Send to Nano Banana
        model_id = MODELS[model]
        img_b64 = _pil_to_base64_full(face_crop)

        payload = {
            "contents": [{
                "parts": [
                    {"text": final_prompt},
                    {"inline_data": {"mime_type": "image/png", "data": img_b64}},
                ]
            }],
            "generationConfig": {"responseModalities": ["TEXT", "IMAGE"]},
        }

        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_id}:generateContent?key={api_key}"
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        print(f"[EyeMatch] Sending face crop to {model_id}...")
        resp = urllib.request.urlopen(req, timeout=180)
        result = json.loads(resp.read())

        edited_crop = None
        for candidate in result.get("candidates", []):
            for part in candidate.get("content", {}).get("parts", []):
                if "inlineData" in part:
                    img_data = base64.b64decode(part["inlineData"]["data"])
                    edited_crop = Image.open(io.BytesIO(img_data)).convert("RGB")

        if edited_crop is None:
            raise RuntimeError("No image in API response.")

        self._save_debug("01_edited_crop", edited_crop)

        # Paste back
        if crop_coords:
            result_pil, mask = _paste_crop_back(pil_img, edited_crop, crop_coords, blur_radius=blend_radius)
            self._save_debug("01_mask", mask)
            self._save_debug("01_composited", result_pil)
        else:
            result_pil = edited_crop
            if result_pil.size != original_size:
                result_pil = result_pil.resize(original_size, Image.LANCZOS)

        # Output quality
        target_size = QUALITY_OPTIONS.get(output_quality)
        if target_size is not None:
            w, h = result_pil.size
            if w >= h:
                new_w = target_size
                new_h = int(h * target_size / w)
            else:
                new_h = target_size
                new_w = int(w * target_size / h)
            result_pil = result_pil.resize((new_w, new_h), Image.LANCZOS)

        debug_lines.append(f"edit_model: {model_id}")
        debug_lines.append(f"output_size: {result_pil.size}")
        debug_info = "\n".join(debug_lines)
        self._save_debug("99_final", result_pil)
        self._save_debug("debug_info", text=debug_info)
        print(f"[EyeMatch] Done. {debug_info}")
        return (_pil_to_tensor(result_pil), final_prompt, debug_info,)


NODE_CLASS_MAPPINGS = {
    "NanoBananaEyeMatch": NanoBananaEyeMatch,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "NanoBananaEyeMatch": "Nano Banana Eye Match",
}
