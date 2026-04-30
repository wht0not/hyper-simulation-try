#!/usr/bin/env python3

import sys
import typing

print(f"Python version: {sys.version}")

if not hasattr(typing, "NotRequired"):
    print("typing.NotRequired not found, adding patch...")
    from typing import Optional
    typing.NotRequired = Optional
    print("Patch applied successfully")
else:
    print("typing.NotRequired already exists")

print(f"typing.NotRequired = {typing.NotRequired}")

try:
    print("\nTrying to import langmem...")
    from langmem import create_manage_memory_tool, create_search_memory_tool
    print("✓ langmem imported successfully!")
except Exception as e:
    print(f"✗ Error importing langmem: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
