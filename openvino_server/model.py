"""
OpenVINO wrapper for Qwen2.5-VL-3B-Instruct.
Handles model loading and image+text inference.
"""

import base64
import re
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image


class QwenVLOpenVINO:
    def __init__(self, model_dir: str, device: str = "CPU"):
        """
        Args:
            model_dir: Path to the exported OpenVINO model folder.
                       This is the folder produced by optimum-cli export openvino
                       OR downloaded from HuggingFace (e.g. the ov-int4 repo).
            device: "CPU", "GPU", or "NPU"
        """
        self.device = device
        self.model_dir = model_dir
        self.pipe = None
        self._load()

    def _load(self):
        try:
            import openvino_genai as ov_genai

            print(f"[model] Loading OpenVINO VLM pipeline from: {self.model_dir}")
            pipeline_config = {"CACHE_DIR": str(Path(self.model_dir) / "ov_cache")}
            self.pipe = ov_genai.VLMPipeline(self.model_dir, self.device, **pipeline_config)
            print(f"[model] Model loaded on {self.device} ✓")
        except Exception as e:
            raise RuntimeError(f"Failed to load OpenVINO model: {e}")

    def _decode_image(self, image_url: str) -> Image.Image:
        """
        Accepts:
          - base64 data URI:  "data:image/png;base64,<data>"
          - plain base64 string (no prefix)
        """
        if image_url.startswith("data:"):
            # strip the data:image/xxx;base64, prefix
            header, data = image_url.split(",", 1)
        else:
            data = image_url

        img_bytes = base64.b64decode(data)
        return Image.open(BytesIO(img_bytes)).convert("RGB")

    def _pil_to_ov_tensor(self, image: Image.Image):
        import openvino as ov

        # Resize to a reasonable size for speed - UI-TARS typically uses 1080p downscaled
        max_side = 1120
        w, h = image.size
        if max(w, h) > max_side:
            scale = max_side / max(w, h)
            image = image.resize((int(w * scale), int(h * scale)), Image.LANCZOS)

        arr = np.array(image.getdata()).reshape(1, image.size[1], image.size[0], 3).astype(np.uint8)
        return ov.Tensor(arr)

    def generate(
        self,
        messages: list[dict],
        max_new_tokens: int = 512,
        temperature: float = 0.0,
    ) -> str:
        """
        messages: OpenAI-style list of {"role": ..., "content": ...}
        content can be a string or a list of {"type": "text"|"image_url", ...} parts.

        Returns the assistant's response string.
        """
        import openvino_genai as ov_genai

        prompt_text = ""
        image_tensor = None

        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")

            if isinstance(content, str):
                prompt_text += content + "\n"

            elif isinstance(content, list):
                for part in content:
                    if part.get("type") == "text":
                        prompt_text += part.get("text", "") + "\n"
                    elif part.get("type") == "image_url":
                        url = part["image_url"]["url"]
                        pil_img = self._decode_image(url)
                        image_tensor = self._pil_to_ov_tensor(pil_img)

        # Build generation config
        gen_config = ov_genai.GenerationConfig()
        gen_config.max_new_tokens = max_new_tokens
        # temperature=0 means greedy decoding
        if temperature > 0:
            gen_config.do_sample = True
            gen_config.temperature = temperature
        else:
            gen_config.do_sample = False

        if image_tensor is not None:
            result = self.pipe.generate(
                prompt_text,
                image=image_tensor,
                generation_config=gen_config,
            )
        else:
            result = self.pipe.generate(
                prompt_text,
                generation_config=gen_config,
            )

        return result.texts[0]