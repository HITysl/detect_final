import json
import numpy as np
import os

# Determine the absolute path to the config.json file
# Assuming config.py and config.json are in the same directory
CONFIG_FILE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.json")

def load_config(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        config_data = json.load(f)
    return config_data

_config_data = load_config(CONFIG_FILE_PATH)

CAMERA_PARAMS = _config_data.get("CAMERA_PARAMS", {})
TRANSFORMATIONS_RAW = _config_data.get("TRANSFORMATIONS", {})
GRID_PARAMS = _config_data.get("GRID_PARAMS", {})
YOLO_PARAMS = _config_data.get("YOLO_PARAMS", {})
BOX_WEIGHT = _config_data.get("BOX_WEIGHT")
IP_CAMERA = _config_data.get("IP_CAMERA")
IP_PLC = _config_data.get("IP_PLC")
IP_PORT_Csharp = _config_data.get("IP_PORT_Csharp")
IP_HOST_Csharp = _config_data.get("IP_HOST_Csharp")
MODEL = _config_data.get("MODEL")

# Convert transformation matrices from lists to numpy arrays
TRANSFORMATIONS = {}
for key, value in TRANSFORMATIONS_RAW.items():
    if isinstance(value, list): # Ensure it's a list before converting
        TRANSFORMATIONS[key] = np.array(value)
    else:
        TRANSFORMATIONS[key] = value # Or handle error/default

# Note: JSON null values are automatically converted to Python None by json.load().
# So, GRID_PARAMS['row_count'] and GRID_PARAMS['col_count'] will be None if they are null in config.json.
# Type conversions for other simple types (int, float, str) are also handled by json.load().