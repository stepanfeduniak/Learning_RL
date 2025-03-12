import yaml
import time
from pathlib import Path

class ConfigManager:
    def __init__(self, config_path):
        with open(config_path, 'r') as f:
            self._params = yaml.safe_load(f)  # intern als _params speichern
        
        # Generate unique run ID based on timestamp
        self.run_id = time.strftime("DQN_RUN_Normal")
        if self._params['run_management']['resume_from']:
            self.run_id = self._params['run_management']['resume_from']
        
        # Create base directories
        self.base_dir = Path(self._params['logging']['base_log_dir']) / self.run_id
        self.base_dir.mkdir(parents=True, exist_ok=True)
    
    def get_tensorboard_dir(self):
        """Returns the directory for tensorboard logs"""
        tb_dir = self.base_dir / "tensorboard"
        tb_dir.mkdir(exist_ok=True)
        return str(tb_dir)
    
    def get_checkpoint_dir(self):
        """Returns the directory for model checkpoints"""
        checkpoint_dir = self.base_dir / "checkpoints"
        checkpoint_dir.mkdir(exist_ok=True)
        return checkpoint_dir
    
    @property
    def params(self):
        """Access to the full config dictionary."""
        return self._params

