import os
import shutil
import importlib.util

fix_file = os.path.join(os.path.dirname(__file__), 'passive_env_checker.py')

spec = importlib.util.find_spec('gym.utils.passive_env_checker')

try:
    shutil.copy(fix_file, spec.origin)
    print("File patched successfully.")
except Exception as e:
    print(f"Error copying file: {e}")
