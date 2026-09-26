import sys
from pathlib import Path

# Make the project modules (config.py, gesture_controller.py, ...) and the test
# helpers importable no matter where pytest is started from.
TESTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(TESTS_DIR.parent))
sys.path.insert(0, str(TESTS_DIR))
