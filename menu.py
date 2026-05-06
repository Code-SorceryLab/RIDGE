"""Root entry point — run from the project root: python menu.py"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from ridge.menu import run_menu

if __name__ == "__main__":
    run_menu()
