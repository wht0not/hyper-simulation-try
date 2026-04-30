import sys
import typing

if not hasattr(typing, "NotRequired"):
    try:
        from typing_extensions import NotRequired
        typing.NotRequired = NotRequired
    except ImportError:
        from typing import Optional
        typing.NotRequired = Optional
