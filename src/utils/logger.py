"""
Logging utilities for experiments.
Provides consistent logging across the project.
"""

import os
import sys
import logging
from pathlib import Path
from typing import Optional
from datetime import datetime


class ColoredFormatter(logging.Formatter):
    """
    Colored log formatter for console output.
    """
    
    # ANSI color codes
    COLORS = {
        'DEBUG': '\033[36m',      # Cyan
        'INFO': '\033[32m',       # Green
        'WARNING': '\033[33m',    # Yellow
        'ERROR': '\033[31m',      # Red
        'CRITICAL': '\033[35m',   # Magenta
        'RESET': '\033[0m',       # Reset
    }
    
    def format(self, record):
        """Format log record with colors."""
        # Add color to levelname
        levelname = record.levelname
        if levelname in self.COLORS:
            record.levelname = f"{self.COLORS[levelname]}{levelname}{self.COLORS['RESET']}"
        
        result = super().format(record)
        
        # Reset color at end
        return result


def setup_logger(
    name: str = 'molecular_property_prediction',
    log_dir: Optional[str] = None,
    log_file: Optional[str] = None,
    level: str = 'INFO',
    console: bool = True,
    file_logging: bool = True,
    colored: bool = True
) -> logging.Logger:
    """
    Set up logger with console and file handlers.
    
    Args:
        name: Logger name
        log_dir: Directory for log files
        log_file: Log file name (if None, generates timestamp-based name)
        level: Logging level ('DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL')
        console: Whether to log to console
        file_logging: Whether to log to file
        colored: Whether to use colored output (console only)
        
    Returns:
        Configured logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))
    logger.handlers = []  # Clear existing handlers
    
    if colored:
        console_formatter = ColoredFormatter(
            '%(levelname)s - %(message)s'
        )
    else:
        console_formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
    
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, level.upper()))
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)
    
    if file_logging:
        if log_dir is None:
            log_dir = './logs'
        
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        
        if log_file is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            log_file = f'{name}_{timestamp}.log'
        
        log_path = log_dir / log_file
        
        file_handler = logging.FileHandler(log_path)
        file_handler.setLevel(getattr(logging, level.upper()))
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
        
        logger.info(f"Logging to file: {log_path}")
    
    return logger


def get_logger(name: str = 'molecular_property_prediction') -> logging.Logger:
    """
    Get existing logger or create new one.
    
    Args:
        name: Logger name
        
    Returns:
        Logger instance
    """
    logger = logging.getLogger(name)
    
    if not logger.handlers:
        logger = setup_logger(name)
    
    return logger


class ExperimentLogger:
    """
    Logger specifically for experiment tracking.
    Provides convenience methods for logging metrics, parameters, etc.
    """
    
    def __init__(
        self,
        exp_name: str,
        log_dir: str = './logs',
        level: str = 'INFO'
    ):
        """
        Initialize experiment logger.
        
        Args:
            exp_name: Experiment name
            log_dir: Directory for log files
            level: Logging level
        """
        self.exp_name = exp_name
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        

        self.logger = setup_logger(
            name=exp_name,
            log_dir=log_dir,
            log_file=f'{exp_name}.log',
            level=level
        )
        

        self.metrics_file = self.log_dir / f'{exp_name}_metrics.csv'
        self._init_metrics_file()
    
    def _init_metrics_file(self):
        """Initialize metrics CSV file."""
        if not self.metrics_file.exists():
            with open(self.metrics_file, 'w') as f:
                f.write('timestamp,epoch,metric,value\n')
    
    def log_params(self, params: dict):
        """
        Log experiment parameters.
        
        Args:
            params: Dictionary of parameters
        """
        self.logger.info("=" * 50)
        self.logger.info("Experiment Parameters:")
        for key, value in params.items():
            self.logger.info(f"  {key}: {value}")
        self.logger.info("=" * 50)
    
    def log_metric(self, metric_name: str, value: float, epoch: Optional[int] = None):
        """
        Log a metric value.
        
        Args:
            metric_name: Name of metric
            value: Metric value
            epoch: Optional epoch number
        """
        # Log to console
        if epoch is not None:
            self.logger.info(f"Epoch {epoch} - {metric_name}: {value:.4f}")
        else:
            self.logger.info(f"{metric_name}: {value:.4f}")
        
        # Log to metrics file
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        epoch_str = str(epoch) if epoch is not None else ''
        
        with open(self.metrics_file, 'a') as f:
            f.write(f'{timestamp},{epoch_str},{metric_name},{value}\n')
    
    def log_metrics(self, metrics: dict, epoch: Optional[int] = None):
        """
        Log multiple metrics.
        
        Args:
            metrics: Dictionary of metrics
            epoch: Optional epoch number
        """
        for name, value in metrics.items():
            self.log_metric(name, value, epoch)
    
    def log_epoch(self, epoch: int, train_loss: float, val_loss: float, **kwargs):
        """
        Log epoch summary.
        
        Args:
            epoch: Epoch number
            train_loss: Training loss
            val_loss: Validation loss
            **kwargs: Additional metrics
        """
        self.logger.info("-" * 50)
        self.logger.info(f"Epoch {epoch}:")
        self.logger.info(f"  Train Loss: {train_loss:.4f}")
        self.logger.info(f"  Val Loss: {val_loss:.4f}")
        
        for key, value in kwargs.items():
            if isinstance(value, float):
                self.logger.info(f"  {key}: {value:.4f}")
            else:
                self.logger.info(f"  {key}: {value}")
    
    def log_model_info(self, model):
        """
        Log model architecture information.
        
        Args:
            model: PyTorch model
        """
        self.logger.info("=" * 50)
        self.logger.info("Model Architecture:")
        self.logger.info(str(model))
        
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        self.logger.info(f"Total Parameters: {total_params:,}")
        self.logger.info(f"Trainable Parameters: {trainable_params:,}")
        self.logger.info("=" * 50)
    
    def log_results(self, results: dict):
        """
        Log final results.
        
        Args:
            results: Dictionary of results
        """
        self.logger.info("=" * 50)
        self.logger.info("Final Results:")
        for key, value in results.items():
            if isinstance(value, float):
                self.logger.info(f"  {key}: {value:.4f}")
            else:
                self.logger.info(f"  {key}: {value}")
        self.logger.info("=" * 50)
    
    def info(self, message: str):
        """Log info message."""
        self.logger.info(message)
    
    def warning(self, message: str):
        """Log warning message."""
        self.logger.warning(message)
    
    def error(self, message: str):
        """Log error message."""
        self.logger.error(message)
    
    def debug(self, message: str):
        """Log debug message."""
        self.logger.debug(message)


if __name__ == "__main__":
    import tempfile
    
    print("Testing logging utilities...")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        print("\n1. Testing Basic Logger")
        logger = setup_logger(
            name='test_logger',
            log_dir=tmpdir,
            level='INFO',
            console=True,
            colored=True
        )
        
        logger.debug("This is a debug message (should not appear)")
        logger.info("This is an info message")
        logger.warning("This is a warning message")
        logger.error("This is an error message")
        
        print("\n2. Testing Get Logger")
        logger2 = get_logger('test_logger')
        logger2.info("Using get_logger()")
        
        print("\n3. Testing Experiment Logger")
        exp_logger = ExperimentLogger(
            exp_name='test_experiment',
            log_dir=tmpdir,
            level='INFO'
        )
        
        params = {
            'model': 'GCN',
            'hidden_dim': 128,
            'learning_rate': 0.001,
            'epochs': 100
        }
        exp_logger.log_params(params)
        
        exp_logger.log_metric('train_loss', 0.5234, epoch=1)
        exp_logger.log_metric('val_loss', 0.4567, epoch=1)
        

        metrics = {
            'roc_auc': 0.8523,
            'accuracy': 0.7891,
            'f1': 0.8012
        }
        exp_logger.log_metrics(metrics, epoch=1)
        
        exp_logger.log_epoch(
            epoch=2,
            train_loss=0.4123,
            val_loss=0.3897,
            roc_auc=0.8756,
            accuracy=0.8012
        )
        
        results = {
            'test_roc_auc': 0.8634,
            'test_accuracy': 0.7945,
            'test_f1': 0.8123
        }
        exp_logger.log_results(results)
        
        print("\n4. Checking Metrics File")
        metrics_file = Path(tmpdir) / 'test_experiment_metrics.csv'
        if metrics_file.exists():
            print(f"   Metrics file created: {metrics_file}")
            with open(metrics_file, 'r') as f:
                print(f"   First 5 lines:")
                for i, line in enumerate(f):
                    if i < 5:
                        print(f"     {line.strip()}")
        
        print("\n5. Testing Different Log Levels")
        for level in ['DEBUG', 'INFO', 'WARNING', 'ERROR']:
            logger = setup_logger(
                name=f'test_{level.lower()}',
                log_dir=tmpdir,
                level=level,
                console=True,
                file_logging=False
            )
            logger.debug(f"DEBUG message at {level} level")
            logger.info(f"INFO message at {level} level")
            logger.warning(f"WARNING message at {level} level")
            logger.error(f"ERROR message at {level} level")
            print()
    
    print("\n Logger tests complete!")
