"""Process-wide startup hooks for the ZAYA serving environment."""

import importlib.abc
import importlib.util
import os
import sys


_BLOCKED_FLASH_ATTN_MODULES = (
    "flash_attn",
    "flash_attn_2_cuda",
    "flash_attn_3_cuda",
    "flash_attn_interface",
    "flash_attn_cuda",
)
_ORIGINAL_FIND_SPEC = importlib.util.find_spec


def _is_blocked_flash_attn_name(fullname):
    return (
        fullname == "flash_attn"
        or fullname.startswith("flash_attn.")
        or fullname in _BLOCKED_FLASH_ATTN_MODULES
    )


class _FlashAttnBlocker(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if _is_blocked_flash_attn_name(fullname):
            raise ModuleNotFoundError(f"No module named {fullname!r}")
        return None


def _find_spec_without_flash_attn(name, package=None):
    if _is_blocked_flash_attn_name(name):
        return None
    return _ORIGINAL_FIND_SPEC(name, package)


if os.environ.get("ZAYA_BLOCK_FLASH_ATTN", "0") == "1":
    importlib.util.find_spec = _find_spec_without_flash_attn
    sys.meta_path.insert(0, _FlashAttnBlocker())
