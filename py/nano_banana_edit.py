import base64
import io
import json
import os
import urllib.request
from datetime import datetime
import numpy as np
import torch
from PIL import Image, ImageFilter

# Debug output directory (inside this custom node package)
_DEBUG_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "debug_output")

MODELS = {
    "Nano Banana 2 (Flash)": "gemini-3.1-flash-image-preview",
    "Nano Banana Pro": "gemini-3-pro-image-preview",
}

COMPARE_MODELS = {
    "gemini-2.5-flash (fast, ~1 won)": "gemini-2.5-flash",
    "gemini-2.5-pro (accurate, ~10 won)": "gemini-2.5-pro",
    "gemini-3-pro (best, ~15 won)": "gemini-3-pro",
}

QUALITY_OPTIONS = {
    "original": None,
    "512": 512,
    "1024": 1024,
    "2048": 2048,
}

COMPOSITE_MODES = {
    "full": "Return the full edited image as-is",
    "eyes_only": "Composite only the eye region onto the original",
}

DETECT_MODES = {
    "gemini": "Use Gemini vision to compare gaze (more accurate, ~1 won per call)",
    "insightface": "Use InsightFace landmarks to compare gaze (free, local, less accurate)",
}

# InsightFace singleton
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


def _pil_to_base64(pil_image, fmt="JPEG", max_size=512):
    """Encode PIL image to base64. Downscale for API efficiency."""
    img = pil_image.copy()
    w, h = img.size
    if max(w, h) > max_size:
        scale = max_size / max(w, h)
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return base64.b64encode(buf.getvalue()).decode()


def _pil_to_base64_full(pil_image, fmt="PNG"):
    """Encode PIL image to base64 at full resolution."""
    buf = io.BytesIO()
    pil_image.save(buf, format=fmt)
    return base64.b64encode(buf.getvalue()).decode()


def _measure_gaze_insightface(pil_image):
    """Measure gaze direction using insightface. Returns (h_offset, v_offset) normalized."""
    import cv2

    app = _get_face_app()
    img_cv2 = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    faces = app.get(img_cv2)

    if not faces:
        return 0.0, 0.0

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

    left_top_y = lm[37][1]
    left_bottom_y = lm[41][1]
    left_center_y = (left_top_y + left_bottom_y) / 2.0
    left_height = left_bottom_y - left_top_y

    right_top_y = lm[91][1]
    right_bottom_y = lm[95][1]
    right_center_y = (right_top_y + right_bottom_y) / 2.0
    right_height = right_bottom_y - right_top_y

    left_iris_x, left_iris_y = face.kps[0]
    right_iris_x, right_iris_y = face.kps[1]

    h_left = (left_iris_x - left_center_x) / left_width if left_width > 0 else 0.0
    h_right = (right_iris_x - right_center_x) / right_width if right_width > 0 else 0.0
    v_left = (left_iris_y - left_center_y) / left_height if left_height > 0 else 0.0
    v_right = (right_iris_y - right_center_y) / right_height if right_height > 0 else 0.0

    h_avg = (h_left + h_right) / 2.0
    v_avg = (v_left + v_right) / 2.0

    return float(h_avg), float(v_avg)


def _gaze_to_description(h_offset, v_offset):
    """Convert normalized gaze offset to human-readable direction."""
    if h_offset < -0.05:
        h_dir = "to their left"
    elif h_offset > 0.05:
        h_dir = "to their right"
    else:
        h_dir = "straight ahead"

    if v_offset < -0.05:
        v_dir = "slightly upward"
    elif v_offset > 0.05:
        v_dir = "slightly downward"
    else:
        v_dir = None

    if h_dir == "straight ahead" and v_dir is None:
        return "directly at the camera"
    elif v_dir:
        return f"{h_dir} and {v_dir}"
    else:
        return h_dir


def _insightface_compare_gaze(source_pil, target_pil, gaze_threshold, debug_lines):
    """
    Compare gaze using insightface landmarks.
    Returns (needs_edit: bool, target_gaze_desc: str)
    """
    source_h, source_v = _measure_gaze_insightface(source_pil)
    target_h, target_v = _measure_gaze_insightface(target_pil)

    diff_h = abs(target_h - source_h)
    diff_v = abs(target_v - source_v)
    diff_total = max(diff_h, diff_v)

    target_desc = _gaze_to_description(target_h, target_v)

    debug_lines.append(f"detect_mode: insightface")
    debug_lines.append(f"source_gaze: h={source_h:+.4f} v={source_v:+.4f}")
    debug_lines.append(f"target_gaze: h={target_h:+.4f} v={target_v:+.4f}")
    debug_lines.append(f"diff: h={diff_h:.4f} v={diff_v:.4f} max={diff_total:.4f}")
    debug_lines.append(f"threshold: {gaze_threshold}")
    debug_lines.append(f"target_description: {target_desc}")

    needs_edit = diff_total >= gaze_threshold
    debug_lines.append(f"needs_edit: {needs_edit}")

    return needs_edit, target_desc


def _crop_eye_region(pil_image, pad_mult=2.0):
    """Crop the eye region from an image using insightface landmarks. Returns cropped PIL or original if detection fails."""
    import cv2

    app = _get_face_app()
    img_cv2 = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    faces = app.get(img_cv2)

    if not faces:
        return pil_image

    face = faces[0]
    lm = face.landmark_2d_106
    w, h = pil_image.size

    left_eye_pts = lm[33:43]
    right_eye_pts = lm[87:97]
    all_eye_pts = np.vstack([left_eye_pts, right_eye_pts])

    x_min, y_min = all_eye_pts.min(axis=0)
    x_max, y_max = all_eye_pts.max(axis=0)

    pad_x = (x_max - x_min) * pad_mult
    pad_y = (y_max - y_min) * pad_mult
    x_min = max(0, int(x_min - pad_x))
    y_min = max(0, int(y_min - pad_y))
    x_max = min(w, int(x_max + pad_x))
    y_max = min(h, int(y_max + pad_y))

    return pil_image.crop((x_min, y_min, x_max, y_max))


def _gemini_compare_gaze(source_pil, target_pil, api_key, debug_lines, prefix="",
                         save_debug_fn=None, compare_model_id="gemini-2.5-flash"):
    """
    Use Gemini text model to compare gaze direction between two images.
    Crops eye region first for better accuracy.
    Returns (needs_edit: bool, target_gaze_desc: str)
    """
    source_eyes = _crop_eye_region(source_pil)
    target_eyes = _crop_eye_region(target_pil)

    if save_debug_fn:
        save_debug_fn(f"{prefix}source_eyes_crop", source_eyes)
        save_debug_fn(f"{prefix}target_eyes_crop", target_eyes)

    source_b64 = _pil_to_base64(source_eyes)
    target_b64 = _pil_to_base64(target_eyes)

    compare_prompt = (
        "Look at the eye gaze direction (where the pupils/irises are pointing) in these two photos. "
        "Image 1 is the source, Image 2 is the target.\n\n"
        "Respond in this exact JSON format only, no other text:\n"
        '{"source_gaze": "<direction>", "target_gaze": "<direction>", '
        '"same_direction": true/false, "explanation": "<brief reason>"}\n\n'
        "For direction, use terms like: "
        '"straight at camera", "slightly left", "slightly right", '
        '"looking left", "looking right", "looking up", "looking down", etc.'
    )

    payload = {
        "contents": [{
            "parts": [
                {"text": compare_prompt},
                {"inline_data": {"mime_type": "image/jpeg", "data": source_b64}},
                {"inline_data": {"mime_type": "image/jpeg", "data": target_b64}},
            ]
        }],
        "generationConfig": {
            "responseModalities": ["TEXT"],
            "temperature": 0.1,
        },
    }

    url = f"https://generativelanguage.googleapis.com/v1beta/models/{compare_model_id}:generateContent?key={api_key}"

    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    print(f"[NanoBananaEdit] Comparing gaze via {compare_model_id}...")
    resp = urllib.request.urlopen(req, timeout=60)
    result = json.loads(resp.read())

    text = ""
    for candidate in result.get("candidates", []):
        for part in candidate.get("content", {}).get("parts", []):
            if "text" in part:
                text += part["text"]

    tag = f"{prefix}detect_mode" if prefix else "detect_mode"
    debug_lines.append(f"{tag}: gemini ({compare_model_id})")

    try:
        clean = text.strip()
        if clean.startswith("```"):
            clean = clean.split("\n", 1)[1] if "\n" in clean else clean[3:]
            clean = clean.rsplit("```", 1)[0]
        parsed = json.loads(clean.strip())

        source_gaze = parsed.get("source_gaze", "unknown")
        target_gaze = parsed.get("target_gaze", "unknown")
        same_dir = parsed.get("same_direction", True)
        explanation = parsed.get("explanation", "")

        debug_lines.append(f"{prefix}source_gaze: {source_gaze}")
        debug_lines.append(f"{prefix}target_gaze: {target_gaze}")
        debug_lines.append(f"{prefix}same_direction: {same_dir}")
        debug_lines.append(f"{prefix}explanation: {explanation}")
        debug_lines.append(f"{prefix}needs_edit: {not same_dir}")

        return not same_dir, target_gaze
    except (json.JSONDecodeError, KeyError) as e:
        print(f"[NanoBananaEdit] Failed to parse compare response: {text}")
        debug_lines.append(f"{prefix}raw_response: {text[:200]}")
        debug_lines.append(f"{prefix}parse_error: {e}")
        debug_lines.append(f"{prefix}needs_edit: True (fallback)")
        return True, "unknown"


def _composite_eyes(original_pil, edited_pil, blur_radius=30):
    """Detect eye region on original, paste only that region from edited."""
    import cv2

    app = _get_face_app()
    img_cv2 = cv2.cvtColor(np.array(original_pil), cv2.COLOR_RGB2BGR)
    faces = app.get(img_cv2)

    if not faces:
        print("[NanoBananaEdit] No face detected, returning full edit")
        return edited_pil

    face = faces[0]
    lm = face.landmark_2d_106
    w, h = original_pil.size

    left_eye_pts = lm[33:43]
    right_eye_pts = lm[87:97]
    all_eye_pts = np.vstack([left_eye_pts, right_eye_pts])

    x_min, y_min = all_eye_pts.min(axis=0)
    x_max, y_max = all_eye_pts.max(axis=0)

    pad_x = (x_max - x_min) * 0.4
    pad_y = (y_max - y_min) * 0.8
    x_min = max(0, int(x_min - pad_x))
    y_min = max(0, int(y_min - pad_y))
    x_max = min(w, int(x_max + pad_x))
    y_max = min(h, int(y_max + pad_y))

    mask_np = np.zeros((h, w), dtype=np.uint8)
    cy = (y_min + y_max) / 2
    cx = (x_min + x_max) / 2
    ry = (y_max - y_min) / 2
    rx = (x_max - x_min) / 2

    Y, X = np.ogrid[:h, :w]
    ellipse = ((X - cx) / rx) ** 2 + ((Y - cy) / ry) ** 2
    mask_np[ellipse <= 1.0] = 255

    mask = Image.fromarray(mask_np)
    mask = mask.filter(ImageFilter.GaussianBlur(radius=blur_radius))

    result = Image.composite(edited_pil, original_pil, mask)
    print(f"[NanoBananaEdit] Eyes composited: region ({x_min},{y_min})->({x_max},{y_max})")
    return result


class NanoBananaEdit:
    """
    Image editing via Google Gemini (Nano Banana) API.
    Optionally accepts a target_image to auto-match gaze direction.
    Supports gemini or insightface for gaze comparison.
    Supports eyes-only compositing to preserve the rest of the image.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "api_key": ("STRING", {
                    "default": "",
                    "tooltip": "Google AI API key (from aistudio.google.com)",
                }),
                "model": (list(MODELS.keys()), {
                    "default": "Nano Banana 2 (Flash)",
                }),
                "output_quality": (list(QUALITY_OPTIONS.keys()), {
                    "default": "original",
                    "tooltip": "Output resolution. 'original' keeps the input size.",
                }),
                "composite_mode": (list(COMPOSITE_MODES.keys()), {
                    "default": "eyes_only",
                    "tooltip": "full: return entire edited image. eyes_only: paste only the eye region onto the original.",
                }),
                "blend_radius": ("INT", {
                    "default": 30,
                    "min": 1,
                    "max": 100,
                    "step": 1,
                    "tooltip": "Gaussian blur radius for eye region blending (eyes_only mode).",
                }),
                "detect_mode": (list(DETECT_MODES.keys()), {
                    "default": "gemini",
                    "tooltip": "gemini: use Gemini vision for gaze comparison. insightface: use local landmarks (free, less accurate).",
                }),
                "compare_model": (list(COMPARE_MODELS.keys()), {
                    "default": "gemini-2.5-pro (accurate, ~10 won)",
                    "tooltip": "Model used for gaze comparison. Only used in gemini detect_mode.",
                }),
                "gaze_threshold": ("FLOAT", {
                    "default": 0.03,
                    "min": 0.0,
                    "max": 0.5,
                    "step": 0.01,
                    "tooltip": "Minimum gaze difference to trigger edit. Only used in insightface detect_mode.",
                }),
                "max_retries": ("INT", {
                    "default": 2,
                    "min": 1,
                    "max": 5,
                    "step": 1,
                    "tooltip": "Max edit attempts. After each edit, gaze is verified. If worse, retries from scratch. Only used with target_image.",
                }),
                "save_debug": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Save intermediate images and debug info to ruucm-comfy/debug_output/",
                }),
            },
            "optional": {
                "target_image": ("IMAGE", {
                    "tooltip": "Target image whose gaze direction to match. If provided, gaze is compared and prompt is auto-generated.",
                }),
                "prompt": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "tooltip": "Manual edit prompt. Ignored when target_image is connected.",
                }),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING", "STRING",)
    RETURN_NAMES = ("edited_image", "used_prompt", "debug_info",)
    FUNCTION = "edit"
    CATEGORY = "image/ai"
    DESCRIPTION = (
        "Edit an image using Google Nano Banana (Gemini) API. "
        "Connect a target_image to auto-match its gaze direction. "
        "Choose gemini (accurate) or insightface (free) for gaze detection. "
        "In 'eyes_only' mode, only the eye region is composited onto the original."
    )

    def _call_nano_banana(self, pil_img, final_prompt, model):
        """Call Nano Banana API and return result PIL image or None."""
        img_b64 = _pil_to_base64_full(pil_img)
        model_id = MODELS[model]

        payload = {
            "contents": [{
                "parts": [
                    {"text": final_prompt},
                    {"inline_data": {"mime_type": "image/png", "data": img_b64}},
                ]
            }],
            "generationConfig": {
                "responseModalities": ["TEXT", "IMAGE"],
            },
        }

        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_id}:generateContent?key={self._api_key}"

        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        print(f"[NanoBananaEdit] Sending to {model_id}...")
        resp = urllib.request.urlopen(req, timeout=180)
        result = json.loads(resp.read())

        result_pil = None
        for candidate in result.get("candidates", []):
            for part in candidate.get("content", {}).get("parts", []):
                if "inlineData" in part:
                    img_data = base64.b64decode(part["inlineData"]["data"])
                    result_pil = Image.open(io.BytesIO(img_data)).convert("RGB")
                elif "text" in part:
                    print(f"[NanoBananaEdit] Response: {part['text']}")

        return result_pil

    def _save_debug(self, name, pil_img=None, text=None):
        """Save debug image or text to debug_output/ folder."""
        if not self._save_debug_enabled:
            return
        os.makedirs(self._debug_session_dir, exist_ok=True)
        if pil_img is not None:
            path = os.path.join(self._debug_session_dir, f"{name}.png")
            pil_img.save(path)
            print(f"[NanoBananaEdit] Debug saved: {path}")
        if text is not None:
            path = os.path.join(self._debug_session_dir, f"{name}.txt")
            with open(path, "w", encoding="utf-8") as f:
                f.write(text)

    def edit(self, image, api_key, model, output_quality, composite_mode, blend_radius,
             detect_mode, compare_model, gaze_threshold, max_retries, save_debug,
             target_image=None, prompt=""):
        if not api_key.strip():
            raise ValueError("API key is required. Get one from aistudio.google.com")

        self._api_key = api_key
        self._compare_model_id = COMPARE_MODELS[compare_model]
        self._save_debug_enabled = save_debug
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._debug_session_dir = os.path.join(_DEBUG_DIR, timestamp)

        pil_img = _tensor_to_pil(image)
        original_size = pil_img.size
        debug_lines = []

        self._save_debug("00_input", pil_img)

        # Auto-generate prompt from target_image gaze
        if target_image is not None:
            target_pil = _tensor_to_pil(target_image)
            self._save_debug("00_target", target_pil)

            # Pre-edit comparison
            debug_lines.append("--- PRE-EDIT COMPARISON ---")
            if detect_mode == "gemini":
                needs_edit, target_gaze_desc = _gemini_compare_gaze(
                    pil_img, target_pil, api_key, debug_lines,
                    prefix="pre_", save_debug_fn=self._save_debug,
                    compare_model_id=self._compare_model_id
                )
            else:
                needs_edit, target_gaze_desc = _insightface_compare_gaze(
                    pil_img, target_pil, gaze_threshold, debug_lines
                )

            if not needs_edit:
                final_prompt = "(skipped - gaze already matches)"
                debug_lines.append("action: SKIPPED")
                debug_info = "\n".join(debug_lines)
                self._save_debug("debug_info", text=debug_info)
                print(f"[NanoBananaEdit] {debug_info}")
                return (_pil_to_tensor(pil_img), final_prompt, debug_info,)

            final_prompt = (
                f"Edit only the eyes in this photo so that both pupils look {target_gaze_desc}. "
                f"Do NOT change anything else - keep the face, hair, skin, clothing, "
                f"background, lighting, and all other details exactly the same. "
                f"Only adjust the iris/pupil position."
            )

            # Edit + verify loop
            best_result = None
            for attempt in range(1, max_retries + 1):
                debug_lines.append(f"--- ATTEMPT {attempt}/{max_retries} ---")
                debug_lines.append(f"action: API CALL")
                print(f"[NanoBananaEdit] Attempt {attempt}/{max_retries}...")

                result_pil = self._call_nano_banana(pil_img, final_prompt, model)
                if result_pil is None:
                    debug_lines.append(f"attempt_{attempt}: NO IMAGE RETURNED")
                    continue

                if result_pil.size != original_size:
                    result_pil = result_pil.resize(original_size, Image.LANCZOS)

                self._save_debug(f"{attempt:02d}_raw_edit", result_pil)

                # Composite eyes if needed
                if composite_mode == "eyes_only":
                    result_pil = _composite_eyes(pil_img, result_pil, blur_radius=blend_radius)
                    self._save_debug(f"{attempt:02d}_composited", result_pil)

                # Post-edit verification: is the result closer to target?
                debug_lines.append(f"--- POST-EDIT VERIFY (attempt {attempt}) ---")
                if detect_mode == "gemini":
                    still_needs_edit, _ = _gemini_compare_gaze(
                        result_pil, target_pil, api_key, debug_lines,
                        prefix=f"verify_{attempt}_", save_debug_fn=self._save_debug,
                        compare_model_id=self._compare_model_id
                    )
                else:
                    still_needs_edit, _ = _insightface_compare_gaze(
                        result_pil, target_pil, gaze_threshold, debug_lines
                    )

                if not still_needs_edit:
                    # Verified: result matches target
                    debug_lines.append(f"attempt_{attempt}: VERIFIED OK")
                    best_result = result_pil
                    break
                else:
                    debug_lines.append(f"attempt_{attempt}: STILL DIFFERENT, retrying...")
                    # Keep first attempt as fallback
                    if best_result is None:
                        best_result = result_pil

            if best_result is None:
                # All attempts failed to produce an image, return original
                debug_lines.append("all attempts failed, returning original")
                best_result = pil_img

            result_pil = best_result

        else:
            # Manual prompt mode - no verification loop
            if prompt.strip():
                final_prompt = prompt
                debug_lines.append("mode: manual prompt")
            else:
                final_prompt = (
                    "Edit only the eyes so that both pupils look directly at the camera. "
                    "Keep everything else exactly the same."
                )
                debug_lines.append("mode: default prompt (no target_image)")

            result_pil = self._call_nano_banana(pil_img, final_prompt, model)
            if result_pil is None:
                raise RuntimeError("No image in API response.")

            if result_pil.size != original_size:
                result_pil = result_pil.resize(original_size, Image.LANCZOS)

            if composite_mode == "eyes_only":
                result_pil = _composite_eyes(pil_img, result_pil, blur_radius=blend_radius)

        # Handle output quality / resolution
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

        debug_lines.append(f"edit_model: {MODELS[model]}")
        debug_lines.append(f"output_size: {result_pil.size}")
        debug_info = "\n".join(debug_lines)
        self._save_debug("99_final", result_pil)
        self._save_debug("debug_info", text=debug_info)
        print(f"[NanoBananaEdit] {debug_info}")
        return (_pil_to_tensor(result_pil), final_prompt, debug_info,)


NODE_CLASS_MAPPINGS = {
    "NanoBananaEdit": NanoBananaEdit,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "NanoBananaEdit": "Nano Banana Edit",
}
