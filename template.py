import os
from pathlib import Path
import logging

# Set up logging configuration to display time-stamped messages
logging.basicConfig(level=logging.INFO, format='[%(asctime)s]: %(message)s')

# Define your project name (used in file paths below)
project_name = "cccClassifier"

# List of files and directories you want to create for your project structure
list_of_file = [
    ".github/workflows/.gitkeep",  # GitHub Actions placeholder
    f"src/{project_name}/__init__.py",  # Package initializer
    f"src/{project_name}/components/__init__.py",  # Components submodule
    f"src/{project_name}/utils/__init__.py",  # Utilities submodule
    f"src/{project_name}/config/__init__.py",  # Config submodule
    f"src/{project_name}/config/configuration.py",  # Config logic
    f"src/{project_name}/pipeline/__init__.py",  # Pipeline submodule
    f"src/{project_name}/entity/__init__.py",  # Entity/data models
    f"src/{project_name}/constants/__init__.py",  # Constants submodule
    "config/config.yaml",  # External configuration
    "dvc.yaml",  # DVC pipeline file
    "params.yaml",  # Parameters for model/data
    "requirements.txt",  # Python dependencies
    "setup.py",  # Installation script
    "research/trials.ipynb",  # Experiment notebook
    "templates/index.html"  # HTML template
]

# Loop through each file path and create directories/files if needed
for filepath in list_of_file:
    filepath = Path(filepath)  # Convert to a Path object
    filedir = filepath.parent  # Extract the directory path
    filename = filepath.name   # Extract the file name

    # Create the directory if it doesn't exist
    if filedir != Path(""):
        os.makedirs(filedir, exist_ok=True)
        logging.info(f"Creating directory: {filedir} for the file: {filename}")
    
    # Create the file if it doesn't exist or is empty
    if not filepath.exists() or filepath.stat().st_size == 0:
        with open(filepath, "w") as f:
            pass  # Just create an empty file
        logging.info(f"Creating empty file: {filepath}")
    else:
        # If file already exists and is non-empty, log that info
        logging.info(f"{filename} already exists")
