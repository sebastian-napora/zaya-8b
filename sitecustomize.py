"""Process-wide startup hooks for the ZAYA serving environment."""

import importlib.abc
import os
import sys


class _FlashAttnBlocker(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        blocked = (
            "flash_attn",
            "flash_attn_2_cuda",
            "flash_attn_3_cuda",
            "flash_attn_interface",
            "flash_attn_cuda",
        )
        if fullname == "flash_attn" or fullname.startswith("flash_attn.") or fullname in blocked:
            raise ImportError(
                f"{fullname} is blocked because this environment has a flash-attn build "
                "for the wrong GPU architecture. Set ZAYA_BLOCK_FLASH_ATTN=0 to disable."
            )
        return None


if os.environ.get("ZAYA_BLOCK_FLASH_ATTN", "0") == "1":
    sys.meta_path.insert(0, _FlashAttnBlocker())
