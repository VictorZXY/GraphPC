# GraphPC vendored copy of OSWorld-V2 mm_agents/qwen/__init__.py
# at tag osworld-v2.1 (commit 3d778a3c9a34a079316f70df023b166700445792), blob 2f2aaa58ee2d1a5ebfe7504c9608359210faf6b9.
# The OSWorld Qwen cell's CONTROL is upstream's own code: this file is byte-identical to
# that blob except this header and the deltas listed below. Each delta is marked in place
# (`# >>> GraphPC Dn` ... `# <<< GraphPC Dn`) and carries the upstream lines it
# replaces as `#|` lines, so the file reverses mechanically to upstream's; the Qwen mode
# of scripts/osworld/verify_runner_deltas.py holds it to that. The delta list with its
# reasons is in the header of scripts/osworld/run_multienv_qwen_baseline.py.
# Deltas carried: none.
# end of GraphPC header
"""Qwen computer-use agent for OSWorld-V2."""

from .main import QwenAgent

__all__ = ["QwenAgent"]
