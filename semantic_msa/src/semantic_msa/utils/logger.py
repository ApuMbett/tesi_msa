import logging
import sys
from pathlib import Path
from rich.logging import RichHandler
from rich.console import Console

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
        
        # Use a standard text formatter for the file
        formatter = logging.Formatter(
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
        self.logger.info(f"[bold green]✔[/bold green] {msg}", *args, **kwargs, extra={"markup": True})
        
    def step(self, msg, *args, **kwargs):
        """Semantic wrapper for starting a step (logs as INFO with a blue arrow)"""
        self.logger.info(f"[bold blue]➔[/bold blue] {msg}", *args, **kwargs, extra={"markup": True})

# Global singleton logger instance
logger = Logger()
