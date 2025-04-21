from PIL import Image
import torch
import numpy as np
import base64
import io


# Image Base64 to PIL
def base64_to_bytes(b64: str):
    if 'base64,' in b64:
        b64 = b64.split('base64,')[-1]
    if len(b64) % 4 == 0:
        try:
            bs = base64.b64decode(b64)
            return bs
        except Exception as e:
            # printColorError(f'base64 decode fail: {e}')
            pass
    return None

def base642pil(b64: str):
    bs = base64_to_bytes(b64)
    if bs is not None:
        try:
            bs = Image.open(io.BytesIO(bs))
            return bs
        except Exception as e:
            # printColorError(f'base64 to pil fail: {e}')
            pass
    return None


# Tensor to PIL
def tensor2pil(image):
    return Image.fromarray(np.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))

# PIL to Tensor
def pil2tensor(image):
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)


class RuucmShareScreen:

    @classmethod
    def INPUT_TYPES(cls):
        return {
                "required": {
                    "image_base64": ("BASE64",),
                },
                "optional": {
                    "default_image": ("IMAGE",),
                    # "RGBA": ([False, True], {"default": False}),
                    # "prompt": ("STRING", {"multiline": True, "dynamicPrompts": True}),
                    # "weight": ("FLOAT", {"default": 1, "min": 0, "max": 1, "step": 0.01}),
                    # "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                },
                "hidden": {"extra_pnginfo": "EXTRA_PNGINFO", "unique_id": "UNIQUE_ID"},
            }

    CATEGORY = "ruucm 🎾"
    OUTPUT_NODE = True

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "doit"

    def doit(self, image_base64, default_image=None, extra_pnginfo=None, unique_id=None):
        if isinstance(image_base64, str):
            image = base642pil(image_base64)
        else:
            image = None
        
        if image is None:
            # 自定义兜底
            if default_image is not None:
                image = tensor2pil(default_image)
            else:
                image = Image.new(mode='RGB', size=(512, 512), color=(0, 0, 0))
        
        image = pil2tensor(image.convert('RGB'))

        return (image,)


NODE_CLASS_MAPPINGS = {
    "RuucmShareScreen": RuucmShareScreen,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RuucmShareScreen": "Ruucm Share Screen",
}