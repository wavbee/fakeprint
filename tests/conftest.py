import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "app"))
os.environ.setdefault("FAKEPRINT_MODEL_DIR", os.path.join(os.path.dirname(__file__), "..", "models"))
