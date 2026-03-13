from models.base_model import BaseVLMModel
from mlx_vlm import load, generate
from mlx_vlm.prompt_utils import apply_chat_template  # ← inaczej niż Qwen
from mlx_vlm.utils import load_config
from contextlib import redirect_stdout
import io
import re


class SmolVLMModel(BaseVLMModel):

    def load(self) -> None:
        self.model, self.processor = load(self.cfg.model_path)
        # SmolVLM wymaga load_config do apply_chat_template
        self.config = load_config(self.cfg.model_path)
        print(f"✅ {self.cfg.model_tag} loaded")

    def analyze(self, frames, prompt) -> tuple[str, float | None]:
        pil_images      = self.frames_to_pil(frames)
        enhanced_prompt = prompt + "\n\nRespond with ONLY: [LABEL] - reason (max 20 words)"

        # apply_chat_template z prompt_utils automatycznie obsługuje
        # tokeny obrazów i format konwersacji — nie trzeba tego robić ręcznie
        prompt_text = apply_chat_template(
            self.processor,
            self.config,
            enhanced_prompt,
            num_images=len(pil_images)
        )

        buf = io.StringIO()
        with redirect_stdout(buf):
            response = generate(
                self.model,
                self.processor,
                prompt=prompt_text,
                image=pil_images,
                temperature=self.cfg.temperature,
                max_tokens=self.cfg.max_tokens,
                verbose=True
            )

        tokens_per_sec = None
        for line in buf.getvalue().splitlines():
            match = re.search(r"([\d.]+)\s*tokens/sec", line)
            if match:
                tokens_per_sec = float(match.group(1))
                break

        pred_raw = response if isinstance(response, str) else response.text
        return pred_raw.strip(), tokens_per_sec