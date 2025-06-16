"""
Logging utilities for RivalAI.
"""

import os
import logging
import sys
from datetime import datetime
from typing import Optional, Union, TextIO
from pathlib import Path

def setup_logging(experiment_name: str, level: str = 'INFO') -> logging.Logger:
    """Set up logging configuration.
    
    Args:
        experiment_name: Name of the experiment for log file naming
        level: Logging level (default: 'INFO')
        
    Returns:
        Logger instance
    """
    # Create logs directory if it doesn't exist
    log_dir = Path('logs')
    log_dir.mkdir(exist_ok=True)
    
    # Create log filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"{experiment_name}_{timestamp}.log"
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, level.upper()))
    
    # Create formatters
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Add file handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(file_formatter)
    root_logger.addHandler(file_handler)
    
    # Add console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)
    
    # Log initial information
    root_logger.info(f"Logging to file: {log_file}")
    root_logger.info(f"Python version: {sys.version}")
    root_logger.info(f"PyTorch version: {get_pytorch_version()}")
    root_logger.info(f"CUDA available: {is_cuda_available()}")
    if is_cuda_available():
        root_logger.info(f"CUDA device: {get_cuda_device_name()}")
    
    return root_logger

def get_pytorch_version() -> str:
    """Get PyTorch version."""
    try:
        import torch
        return torch.__version__
    except ImportError:
        return "Not installed"

def is_cuda_available() -> bool:
    """Check if CUDA is available."""
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False

def get_cuda_device_name() -> str:
    """Get CUDA device name."""
    try:
        import torch
        return torch.cuda.get_device_name() if torch.cuda.is_available() else "None"
    except ImportError:
        return "None"

def get_logger(name: str) -> logging.Logger:
    """Get a logger instance with the given name.
    
    Args:
        name: Name for the logger
        
    Returns:
        Logger instance
    """
    return logging.getLogger(name) 