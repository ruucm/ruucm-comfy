//@ts-ignore
import { api } from "../../scripts/api.js";

setTimeout(() => {
  console.log("importing ruucm_comfy_web/input.js");
  import(api.api_base + "/ruucm_comfy_web/input.js");
}, 500);
