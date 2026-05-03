from pathlib import Path


PROMPT_DIR = Path(__file__).resolve().parent.parent / "prompts"


def load_prompt(file_name: str) -> str:
    """Load a fixed prompt template from the repository prompt directory."""
    prompt_path = PROMPT_DIR / file_name
    if not prompt_path.exists():
        raise FileNotFoundError(f"Prompt template not found: {prompt_path}")
    return prompt_path.read_text(encoding="utf-8").strip()


def render_prompt(file_name: str, **kwargs) -> str:
    """Load and format a prompt template with runtime values."""
    return load_prompt(file_name).format(**kwargs)
