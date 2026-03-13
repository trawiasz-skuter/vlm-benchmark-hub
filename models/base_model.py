from abc import ABC, abstractmethod
from PIL import Image
from src.config import ModelConfig

class BaseVLMModel(ABC):
    """
    Abstract base class for every VLM models.
    Every model has to aply load() and analyze().
    """

    def __init__(self, cfg: ModelConfig):
        self.cfg = cfg
        # This parameters will be set after load()
        self.model = None
        self.processor = None

    #====================#
    #--ABSTRACT METHODS--#
    #====================#

    @abstractmethod
    def load(self) -> None:
        """Load the model and processor to self.model / self.processor"""
        ...

    @abstractmethod
    def analyze(self, frames: list, prompt: str) -> tuple[str, float | None]:
        """
        Analyze video frames and return prediction.

        Args:
            frames: numpy arrays lists (RGB) with extract_frames()
            prompt: system text with config.py

        Returns:
            (prediction_raw, tokens_per_sec)
            prediction_raw - raw model response, e.g. "[ASSULT] - man hitting another"
            tokens_per_sec - inference's speed or None if not avaiable
        """
        ...

    #==================#
    #--COMMON METHODS--#
    #==================#
    
    def is_loaded(self) -> bool:
        """Check if the model has been loaded before inference."""
        return self.model is not None and self.processor is not None
    
    def frames_to_pil(self, frames: list) -> list[Image.Image]:
        """Converts numpy arrays → PIL Images. Useful in any model."""
        return [Image.fromarray(f) for f in frames]
    
    def __repr__(self) -> str:
        status = "loaded" if self.is_loaded() else "not loaded"
        return f"{self.__class__.__name__}(tag='{self.model_tag}', status={status})"