import yaml
import os
from pathlib import Path
from typing import Dict, Any

def load_config() -> Dict[str, Any]:
    """Load application configuration from YAML file"""
    # 1. Check environment variable for config path
    config_path = os.getenv("APP_CONFIG")
    
    # 2. Fallback to default config path
    if not config_path:
        current_dir = "/home/ubuntu/NoMoreAlzimer/app/utils"
        config_path = "/home/ubuntu/NoMoreAlzimer/config/settings.yaml"
    
    # 3. Load and return configuration
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

