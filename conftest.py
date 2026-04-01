"""
conftest.py — project root
Adds the project root to sys.path so pytest can import
state, config, engines.*, db.* regardless of where it is invoked from.
"""
import sys
import os

# Insert project root at position 0 — takes priority over any installed packages
sys.path.insert(0, os.path.dirname(__file__))
