from models.base_model import BaseVLMModel
from mlx_vlm import load, generate
from contextlib import redirect_stdout
import io, re

class Gemma3nE2BModel(BaseVLMModel):

    def load(self) -> None:
        self.model, self.processor = load(self.cfg.model_path)

    def analyze(self, frames, prompt) -> tuple[str, float | None]:
        pil_images = self.frames_to_pil(frames)
        enhanced_prompt = prompt + "\n\nRespond with ONLY: [LABEL] - reason (max 20 words)"
        
        content = [{"type": "image"} for _ in pil_images]
        content.append({"type": "text", "text": enhanced_prompt})
        
        messages = [{"role": "user", "content": content}]
        prompt_text = self.processor.apply_chat_template(messages, add_generation_prompt=True)

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
        return pred_raw, tokens_per_sec