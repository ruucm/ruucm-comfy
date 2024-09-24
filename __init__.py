import importlib.util
import glob
import os
import sys
from .ruucm import init, get_ext_dir
from aiohttp import web
import server


WEB_DIRECTORY = "entry"

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}



print(f"🎾🎾Loading: Ruucm Comfy")
workspace_path = os.path.join(os.path.dirname(__file__))

dist_path = os.path.join(workspace_path, 'dist/ruucm_comfy_web')
if os.path.exists(dist_path):
    server.PromptServer.instance.app.add_routes([
        web.static('/ruucm_comfy_web/', dist_path),
    ])
else:
    print(f"🎾🎾🔴🔴Error: Web directory not found: {dist_path}")

if init():
    py = get_ext_dir("py")
    files = glob.glob(os.path.join(py, "*.py"), recursive=False)
    for file in files:
        name = os.path.splitext(file)[0]
        spec = importlib.util.spec_from_file_location(name, file)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        if hasattr(module, "NODE_CLASS_MAPPINGS") and getattr(module, "NODE_CLASS_MAPPINGS") is not None:
            NODE_CLASS_MAPPINGS.update(module.NODE_CLASS_MAPPINGS)
            if hasattr(module, "NODE_DISPLAY_NAME_MAPPINGS") and getattr(module, "NODE_DISPLAY_NAME_MAPPINGS") is not None:
                NODE_DISPLAY_NAME_MAPPINGS.update(module.NODE_DISPLAY_NAME_MAPPINGS)

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
