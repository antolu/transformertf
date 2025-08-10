#!/usr/bin/env python3
"""
Unified migration tool for TransformerTF projects.

This script consolidates all migration operations into a single entry point with
consistent behavior and interface.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add the lib directory to the Python path
sys.path.insert(0, str(Path(__file__).parent / "lib"))

from lib.cli import main

if __name__ == "__main__":
    sys.exit(main())
