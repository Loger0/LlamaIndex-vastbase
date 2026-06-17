import sys
import os

# Add tests/ directory to sys.path so `from conftest import ...` works
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "tests"))
