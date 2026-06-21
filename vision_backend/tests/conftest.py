"""Pytest configuration: make ``import vision_backend`` work from any cwd."""

import sys
from pathlib import Path

# add the AI4ExoMars dir (parent of vision_backend) to sys.path
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
