import os
import torch
import requests
import uuid
import folder_paths
import comfy.utils
import comfy.sd

class LoraLoader:
    def __init__(self):
        self.loaded_lora = None

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": { 
                "model": ("MODEL", {"tooltip": "The diffusion model the LoRA will be applied to."}),
                "clip": ("CLIP", {"tooltip": "The CLIP model the LoRA will be applied to."}),
                "lora_name": ("STRING", {"default": "", "tooltip": "The name or URL of the LoRA."}),
                "strength_model": ("FLOAT", {"default": 1.0, "min": -100.0, "max": 100.0, "step": 0.01, "tooltip": "Strength for the diffusion model modification. Negative values invert the effect."}),
                "strength_clip": ("FLOAT", {"default": 1.0, "min": -100.0, "max": 100.0, "step": 0.01, "tooltip": "Strength for the CLIP model modification. Negative values invert the effect."}),
            }
        }
    
    RETURN_TYPES = ("MODEL", "CLIP")
    OUTPUT_TOOLTIPS = ("The modified diffusion model.", "The modified CLIP model.")
    FUNCTION = "load_lora"

    CATEGORY = "loaders"
    DESCRIPTION = "Loads a LoRA model to modify diffusion and CLIP models. Supports external URLs for downloading LoRAs."

    def load_lora(self, model, clip, lora_name, strength_model, strength_clip):
        if strength_model == 0 and strength_clip == 0:
            return (model, clip)
        
        # Check if lora_name is a URL
        if lora_name.startswith("http://") or lora_name.startswith("https://"):
            # Download the external LoRA
            unique_filename = str(uuid.uuid4()) + ".safetensors"
            destination_path = os.path.join(
                folder_paths.get_folder_paths("loras")[0], unique_filename
            )
            print(f"Downloading external LoRA from {lora_name} to {destination_path}")
            response = requests.get(
                lora_name,
                headers={"User-Agent": "Mozilla/5.0"},
                allow_redirects=True,
            )
            with open(destination_path, "wb") as out_file:
                out_file.write(response.content)
            lora_path = destination_path
        else:
            # Use the local LoRA file
            lora_path = folder_paths.get_full_path("loras", lora_name)
        
        lora = None
        if self.loaded_lora is not None:
            if self.loaded_lora[0] == lora_path:
                lora = self.loaded_lora[1]
            else:
                temp = self.loaded_lora
                self.loaded_lora = None
                del temp

        if lora is None:
            lora = comfy.utils.load_torch_file(lora_path, safe_load=True)
            self.loaded_lora = (lora_path, lora)

        model_lora, clip_lora = comfy.sd.load_lora_for_models(model, clip, lora, strength_model, strength_clip)
        return (model_lora, clip_lora)

class LoadExternalLoraModelOnly(LoraLoader):
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": { 
                "model": ("MODEL", {"tooltip": "The diffusion model to modify with the LoRA."}),
                "lora_name": ("STRING", {"default": "", "tooltip": "The name or URL of the LoRA."}),
                "strength_model": ("FLOAT", {"default": 1.0, "min": -100.0, "max": 100.0, "step": 0.01, "tooltip": "Strength for the diffusion model modification. Negative values invert the effect."}),
            }
        }
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "load_lora_model_only"

    CATEGORY = "loaders"
    DESCRIPTION = "Loads a LoRA model to modify only the diffusion model. Supports external URLs for downloading LoRAs."

    def load_lora_model_only(self, model, lora_name, strength_model):
        return (self.load_lora(model, None, lora_name, strength_model, 0)[0],)

NODE_CLASS_MAPPINGS = {
    "LoadExternalLoraModelOnly": LoadExternalLoraModelOnly,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LoadExternalLoraModelOnly": "Load External LoRA Model Only"
}
