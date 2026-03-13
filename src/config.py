from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

@dataclass
class ModelConfig:
    model_tag:      str
    model_path:     str
    temperature:    float
    max_tokens:     int
    num_frames:     int
    prompt:         str

#=====================#
#--PER-MODEL CONFIGS--#
#=====================#

QWEN_CONFIG = ModelConfig(
    model_tag = "Qwen3VL4B_3bit",
    model_path = "mlx-community/Qwen3-VL-4B-Instruct-3bit",
    temperature = 0.0,
    max_tokens = 50,
    num_frames = 16,
    prompt = """You are an advanced security analyst. Analyze these CCTV frames.
    ARSON: Fire, smoke, ignition.
    ASSAULT: Physical fight, hitting, grappling.
    ABUSE: Physical domination, restraining, intimidation.
    ARREST: Police restraining/handcuffing someone.
    NORMAL: Peaceful activity, walking, no violence."""
)

LLAVA_CONFIG = ModelConfig(
    model_tag = "LlavaNeXT7B_4bit",
    model_path = "mlx-community/llava-v1.6-mistral-7b-4bit",
    temperature = 0.0,
    max_tokens = 50,
    num_frames = 16,
    prompt = """You are an advanced security analyst. Analyze these CCTV frames.
    ARSON: Fire, smoke, ignition.
    ASSAULT: Physical fight, hitting, grappling.
    ABUSE: Physical domination, restraining, intimidation.
    ARREST: Police restraining/handcuffing someone.
    NORMAL: Peaceful activity, walking, no violence."""
)

TINYLLAVA_VIDEO = ModelConfig(
    model_tag = "TinyLLaVA-Video-R1",
    model_path  = "Zhang199/TinyLLaVA-Video-R1",
    temperature = 0.0,
    max_tokens  = 50,
    num_frames  = 16,
    prompt      = """You are a security camera analyst. Study these CCTV frames.
ARSON: Fire, smoke, flames, burning.
ASSAULT: Fighting, hitting, punching, kicking.
ABUSE: Restraining, domination, intimidation.
ARREST: Police handcuffing or detaining someone.
NORMAL: Peaceful scene, walking, shopping, no violence."""
)

SMOLVLM_CONFIG = ModelConfig(
    model_tag   = "SmolVLM2_2.2B_mlx",
    model_path  = "mlx-community/SmolVLM2-2.2B-Instruct-mlx",
    temperature = 0.0,
    max_tokens  = 50,
    num_frames  = 4,
    prompt      = """You are a security camera analyst. Study these CCTV frames carefully.
ARSON: Fire, smoke, flames, burning objects.
ASSAULT: Fighting, hitting, punching, kicking, physical attack.
ABUSE: Restraining, domination, intimidation, one person controlling another.
ARREST: Police handcuffing or forcibly detaining someone.
NORMAL: Peaceful scene, walking, shopping, talking, no violence."""
)


IDEFICS_CONFIG = ModelConfig(
    model_tag   = "Idefics3-8B-Llama3-4bit",
    model_path  = "mlx-community/Idefics3-8B-Llama3-4bit",
    temperature = 0.0,
    max_tokens  = 50,
    num_frames  = 4,
    prompt      = """You are a security camera analyst. Study these CCTV frames carefully.
ARSON: Fire, smoke, flames, burning objects.
ASSAULT: Fighting, hitting, punching, kicking, physical attack.
ABUSE: Restraining, domination, intimidation, one person controlling another.
ARREST: Police handcuffing or forcibly detaining someone.
NORMAL: Peaceful scene, walking, shopping, talking, no violence."""
)

GEMMA4B_CONFIG = ModelConfig(
    model_tag   = "gemma-3-4b-it-4bit",
    model_path  = "mlx-community/gemma-3-4b-it-4bit",
    temperature = 0.0,
    max_tokens  = 50,
    num_frames  = 4,
    prompt      = """You are a security camera analyst. Study these CCTV frames carefully.
ARSON: Fire, smoke, flames, burning objects.
ASSAULT: Fighting, hitting, punching, kicking, physical attack.
ABUSE: Restraining, domination, intimidation, one person controlling another.
ARREST: Police handcuffing or forcibly detaining someone.
NORMAL: Peaceful scene, walking, shopping, talking, no violence."""
)

GEMMA3N2B_CONFIG = ModelConfig(
    model_tag   = "gemma-3n-E2B-4bit",
    model_path  = "mlx-community/gemma-3n-E2B-it-4bit",
    temperature = 0.0,
    max_tokens  = 30,
    num_frames  = 4,
    prompt      = """You are a security camera analyst. Study these CCTV frames carefully.
ARSON: Fire, smoke, flames, burning objects.
ASSAULT: Fighting, hitting, punching, kicking, physical attack.
ABUSE: Restraining, domination, intimidation, one person controlling another.
ARREST: Police handcuffing or forcibly detaining someone.
NORMAL: Peaceful scene, walking, shopping, talking, no violence."""
)

#==================#
#--DATASET CONFIG--#
#==================#
DATA_PATH        = "/Users/jakubmatkowski/Dokumenty/PW_repos/Praca_Badawcza/VLM-Test/UCF-Crime-Subset"
NUM_TEST_SAMPLES = 20

def make_output_prefix(cfg: ModelConfig) -> str:
    output_dir = Path("results")
    output_dir.mkdir(parents=True, exist_ok=True)

    ts = datetime.now().strftime('%Y%m%d_%H%M')
    filename = f"results_{cfg.model_tag}_{cfg.num_frames}frames_{ts}"
    return str(output_dir / filename)