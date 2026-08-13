import logging
import sys
from pathlib import Path
from rich.logging import RichHandler
from rich.console import Console
from rich.text import Text

class PlainTextFormatter(logging.Formatter):
    """Strips rich markup tags (like [bold blue]) before writing to a file."""
    def format(self, record):
        import copy
        record_copy = copy.copy(record)
            
        if isinstance(record_copy.msg, str):
            record_copy.msg = Text.from_markup(record_copy.msg).plain
            
        return super().format(record_copy)

# Shared console for rich
console = Console()

class Logger:
    def __init__(self, name: str = "semantic-msa"):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.INFO)
        
        # Prevent propagation to the root logger to avoid duplicate logs
        self.logger.propagate = False

        # Clear any existing handlers (useful if module is reloaded)
        if self.logger.hasHandlers():
            self.logger.handlers.clear()

        # Add Rich console handler
        rich_handler = RichHandler(
            console=console,
            show_time=True,
            show_path=False,  # Keep it clean
            markup=True,
            rich_tracebacks=True,
        )
        rich_handler.setLevel(logging.INFO)
        
        # We don't use a standard Formatter for RichHandler, as it handles its own formatting
        self.logger.addHandler(rich_handler)

    def add_file_handler(self, logs_dir: Path, run_id: str):
        """
        Adds a file handler to log plain text to pipeline_{run_id}.log
        """
        logs_dir.mkdir(parents=True, exist_ok=True)
        log_file = logs_dir / f"pipeline_{run_id}.log"
        
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(logging.INFO)
        
        # Use the custom formatter to strip markup tags
        formatter = PlainTextFormatter(
            "%(asctime)s [%(levelname)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        file_handler.setFormatter(formatter)
        
        self.logger.addHandler(file_handler)
        self.logger.info(f"File logging initialized at [bold cyan]{log_file}[/bold cyan]")

    # Expose standard logging methods
    def info(self, msg, *args, **kwargs):
        self.logger.info(msg, *args, **kwargs)

    def debug(self, msg, *args, **kwargs):
        self.logger.debug(msg, *args, **kwargs)

    def warning(self, msg, *args, **kwargs):
        self.logger.warning(msg, *args, **kwargs)

    def error(self, msg, *args, **kwargs):
        self.logger.error(msg, *args, **kwargs)

    def success(self, msg, *args, **kwargs):
        """Semantic wrapper for successful steps (logs as INFO with a green checkmark)"""
        self.logger.info(f"[bold green][SUCCESS] ✔[/bold green] {msg}", *args, **kwargs, extra={"markup": True, "category": "SUCCESS"})
        
    def step(self, msg, *args, **kwargs):
        """Semantic wrapper for starting a step (logs as INFO with a blue arrow)"""
        self.logger.info(f"[bold blue][STEP] ➔[/bold blue] {msg}", *args, **kwargs, extra={"markup": True, "category": "STEP"})

    def cache(self, msg, *args, **kwargs):
        """Semantic wrapper for cache hits (logs as INFO with a yellow refresh icon)"""
        self.logger.info(f"[bold yellow][CACHE] ⟳[/bold yellow] {msg}", *args, **kwargs, extra={"markup": True, "category": "CACHE"})

    def param(self, msg, *args, **kwargs):
        """Semantic wrapper for parameter logging (logs as INFO with a magenta dot)"""
        self.logger.info(f"[bold magenta][PARAM] •[/bold magenta] {msg}", *args, **kwargs, extra={"markup": True, "category": "PARAM"})

# Global singleton logger instance
logger = Logger()
