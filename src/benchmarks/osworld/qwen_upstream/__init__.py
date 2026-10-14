"""Copies of the Qwen control's code the GraphPC arm runs, so both arms encode a
frame, parse a tool call and translate an action with the same code.

Upstream: xlang-ai/OSWorld-V2, tag ``osworld-v2.1`` (commit 3d778a3c), added by
b85eae7 "feat: add modular Qwen computer-use agent (#439)"; ``qwen_vl_utils.py`` is
``mm_agents/utils/qwen_vl_utils.py``, identical at ``v2026.08.08`` and ``osworld-v2.1``.

    file              upstream path                        upstream blob  edit
    parser.py         mm_agents/qwen/parser.py             9f18254a       none
    actions.py        mm_agents/qwen/actions.py            91d1c157       none
    images.py         mm_agents/qwen/images.py             81d10416       G1: smart_resize is imported
                                                                          from the copy beside it, not
                                                                          from mm_agents.utils (the host
                                                                          checkout has that module; this
                                                                          tree does not); D9: the
                                                                          control's own delta (frames as
                                                                          JPEG), block for block
    prompts.py        mm_agents/qwen/prompts.py            761d86eb       none
    history.py        mm_agents/qwen/history.py            046515d9       none
    recovery.py       mm_agents/qwen/recovery.py           8b4d9b7a       none
    qwen_vl_utils.py  mm_agents/utils/qwen_vl_utils.py     f39088e0       none

Each file carries the header the control's vendored copies carry
(scripts/osworld/qwen_control/): upstream path, tag, commit and blob, and its deltas,
marked in place with the upstream lines they replace as ``#|`` lines, so it reverses
mechanically to upstream's blob.

Nothing else of the control is copied here (not its client, context rescue or
main agent): the GraphPC arm's backbone (``backbone_qwen.py``), grounding
(``action_qwen.py``), prompt text (``prompts.py``) and timeline retention are its own;
these files are the pieces they call. Edit none of them — a difference from upstream
belongs in the caller. The one exception is a delta the control itself carries: it is copied
here under the control's own label, block for block, so both arms keep running the same code
(D9; ``scripts/osworld/verify_runner_deltas.py --cell qwen`` holds the two copies' blocks
equal).
"""
