from pathlib import Path

from utils.log import add_file_handler, get_logger


class FMMLoggerManager:
    """Small compatibility wrapper around the project logger setup."""

    def __init__(self):
        self.handler_id = None
        self.log_file_path = None

    def setup_logging(self, output_dir: str | Path) -> str:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        self.log_file_path = output_dir / "fmm_framework.log"
        self.handler_id = add_file_handler(self.log_file_path, filter="FMM", level="DEBUG")
        return str(self.log_file_path)

    def get_fmm_logger(self, module_name: str, emoji: str = ""):
        if not module_name.startswith("FMM"):
            module_name = f"FMM_{module_name}"
        return get_logger(module_name, emoji=emoji)


fmm_logger_manager = FMMLoggerManager()


def setup_fmm_logging(output_dir: str | Path) -> str:
    return fmm_logger_manager.setup_logging(output_dir)


def get_fmm_logger(module_name: str, emoji: str = ""):
    return fmm_logger_manager.get_fmm_logger(module_name, emoji)
