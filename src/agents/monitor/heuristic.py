from __future__ import annotations

import io

from dataclasses import dataclass
from typing import Callable, Optional, Sequence, Tuple

from PIL import Image, ImageChops

from .base import MonitorAgent, MonitorInput
from ..common.schema import AgentMessage, Observation


# Full-observation text keys, tried in order for the change-detection diff.
# The FULL uncapped text (a11y_tree) must drive `changed`, NOT the 200-node-capped
# observation_text projection — a state change beyond the cap would otherwise read
# as STALL and M1 would wrongly override a real positive effect to no_change. The
# capped observation_text is the last-resort fallback (e.g. image-mode / OSWorld
# observations that carry no full a11y text).
_FULL_TEXT_KEYS = ("a11y_tree", "accessibility_tree", "observation_text")

# Screenshot diff (OSWorld 2.0, where the change signal is the screenshot itself): a pixel
# counts as changed when its grayscale delta exceeds SCREENSHOT_PIXEL_DELTA; the
# frame counts as changed when at least SCREENSHOT_CHANGED_MIN_PIXELS pixels
# changed. The floor sits deliberately LOW — the safe side of design.md §6's
# asymmetry: a false "changed" only leaves the model's label authoritative for
# that step, a false "unchanged" overrides a real effect to no_change. Frames
# are raw PNG, so identical frames diff to exactly 0; a caret blink (~32 px)
# stays below the floor, a clock-digit tick (~100-200 px) trips it. Both
# constants are fixed in design.md §6; the per-step diff telemetry this monitor
# records is what they are checked against.
#
# Scope: the branch fires only when BOTH observations carry
# content["screenshot"]. WebArena's a11y mode never puts a screenshot in
# content, so it never reaches the branch; WebArena image mode IS a real runner
# option and would. That makes "inert on WebArena" a fact about the observation
# mode, not a property of the code — which is why these two numbers and the pointer mask below are read
# from a per-benchmark ScreenshotDiffConfig rather than from module state.
SCREENSHOT_PIXEL_DELTA = 32
SCREENSHOT_CHANGED_MIN_PIXELS = 64

# Pointer-mask geometry, in pixels from the cursor's hotspot. DERIVED, not
# tuned: the OSWorld 2.0 guest's cursor sprite composites into the captured
# frame and extends dx in [0, 19] and dy in [3, 28] from its hotspot, so the box
# that covers it is ASYMMETRIC — a little slack up and left (anti-aliasing and
# the shadow), the sprite's own extent down and right. A 41x41 box gives the
# identical verdict to a symmetric 32-px box on the measured no-op case while
# removing 60% less area, and so halves the worst-case loss on a real change.
# Not configurable: these follow from the sprite, not from a threshold anyone
# should be turning.
POINTER_MASK_BEFORE_PX = 8
POINTER_MASK_AFTER_PX = 32

# The locator's contract is the TWO positions bracketing the diffed frames (the
# previous step's and the current step's). Enforced here, not merely documented:
# masked area is the only direction in which this mechanism can hide a real
# change, so a locator that returns more must not be able to widen it silently.
POINTER_MASK_MAX_POINTS = 2


@dataclass(frozen=True)
class ScreenshotDiffConfig:
    """Per-benchmark configuration of the screenshot change-detection branch.

    The whole screenshot path exists for OSWorld 2.0, so its parameters are
    supplied by the benchmark that needs them rather than living as shared
    module state. The defaults are the two constants above and NO pointer
    mask, so a benchmark that constructs ``HeuristicMonitorAgent()`` with no
    config (WebArena) computes its verdict on the plain diff, with no mask code
    reachable.

    ``pointer_locator`` is a CALLABLE the benchmark supplies, not a flag and not
    a well-known key this module reads off the runtime state. That is
    deliberate: WebArena's adapter also writes ``_executed_actions``, so a
    coordinate parser living here would be inert on WebArena only by accident —
    it would come alive the day that record's format changed. With a locator,
    the mask is unreachable unless a benchmark hands over the function.

    Locator contract:

      * it is called ONLY on the screenshot branch (both observations carry
        image bytes), with the ``MonitorInput`` for the step just executed;
      * it returns pointer positions in SCREEN PIXELS in the captured frame's
        coordinate system — the two frames being diffed bracket the action, so
        the cursor composites at BOTH the previous and the current position and
        both must be returned for the mask to remove the artifact. Measured on a
        true no-op with only the cursor relocated: masking the current position
        alone leaves 206 changed pixels, over three times the floor, for any
        pointer move beyond about 40 px. At most POINTER_MASK_MAX_POINTS are
        honoured;
      * it returns an EMPTY sequence for a step whose current action moves no
        pointer (a key press, a typed string, a wait), and the monitor then
        masks nothing at all. That is not an optimisation: the pointer did not
        move, so it contributes no changed pixels anyway, while masking the
        previous step's click coordinate would eat the characters of a step
        that types into the field it clicked on the step before — common, and
        the one way this mask could hide a real change.
    """

    pixel_delta: int = SCREENSHOT_PIXEL_DELTA
    changed_min_pixels: int = SCREENSHOT_CHANGED_MIN_PIXELS
    pointer_locator: Optional[Callable[[MonitorInput], Sequence[Tuple[int, int]]]] = None


_DEFAULT_SCREENSHOT_DIFF_CONFIG = ScreenshotDiffConfig()


def _screenshot_bytes(observation: Observation) -> Optional[bytes]:
    content = getattr(observation, "content", None)
    if not isinstance(content, dict):
        return None
    value = content.get("screenshot")
    if isinstance(value, (bytes, bytearray)) and value:
        return bytes(value)
    return None


def screenshot_diff(
    previous: bytes,
    current: bytes,
    *,
    config: ScreenshotDiffConfig = _DEFAULT_SCREENSHOT_DIFF_CONFIG,
    pointer_points: Sequence[Tuple[int, int]] = (),
) -> tuple[bool, int, float, int]:
    """``(changed, diff_pixel_count, diff_fraction, diff_pixel_count_unmasked)``
    between two screenshots: grayscale at full resolution, count pixels whose
    |delta| exceeds ``config.pixel_delta``, changed iff the count reaches
    ``config.changed_min_pixels``. ``diff_fraction`` is telemetry only. Frames of
    different sizes are a change by definition (every pixel counted).

    ``pointer_points`` are screen-pixel cursor positions to exclude before
    counting — the ``ScreenshotDiffConfig.pointer_locator`` output, empty by
    default. Each is zeroed out of the difference mask over the asymmetric box
    that covers the cursor sprite. AT MOST ``POINTER_MASK_MAX_POINTS`` are
    honoured: the contract is the two positions bracketing the two frames, and a
    locator returning more must not be able to widen the mask silently — mask
    area is the one direction in which this mechanism can hide a real change, so
    it is bounded here rather than trusted to the caller.

    The UNMASKED count is returned alongside the masked one on every call so the
    verdict stream stays re-auditable offline: a step where the two straddle the
    floor is a step where the mask changed the verdict, and it can be re-judged
    from the record without re-running.
    """
    with Image.open(io.BytesIO(previous)) as prev_image, Image.open(io.BytesIO(current)) as curr_image:
        prev_gray = prev_image.convert("L")
        curr_gray = curr_image.convert("L")
    total_pixels = curr_gray.width * curr_gray.height
    if prev_gray.size != curr_gray.size:
        return True, total_pixels, 1.0, total_pixels
    # |delta| per pixel, thresholded through a 256-entry LUT and counted from
    # the histogram — all C-side, no per-pixel Python loop over 2M pixels.
    delta = ImageChops.difference(prev_gray, curr_gray)
    mask = delta.point(lambda value: 255 if value > config.pixel_delta else 0)
    diff_pixel_count_unmasked = mask.histogram()[255]
    diff_pixel_count = diff_pixel_count_unmasked
    if pointer_points:
        for point in tuple(pointer_points)[:POINTER_MASK_MAX_POINTS]:
            box = _pointer_mask_box(point, width=mask.width, height=mask.height)
            if box is not None:
                mask.paste(0, box)
        diff_pixel_count = mask.histogram()[255]
    diff_fraction = diff_pixel_count / total_pixels if total_pixels else 0.0
    changed = diff_pixel_count >= config.changed_min_pixels
    return changed, diff_pixel_count, diff_fraction, diff_pixel_count_unmasked


def _pointer_mask_box(
    point: Tuple[int, int], *, width: int, height: int,
) -> Optional[Tuple[int, int, int, int]]:
    """The asymmetric cursor box around ``point``, clamped to the frame, in PIL's
    half-open ``(left, upper, right, lower)`` form. ``None`` when the point lies
    outside the frame entirely, so a bad coordinate masks nothing rather than
    silently wrapping to the opposite edge."""
    x, y = int(point[0]), int(point[1])
    if not (0 <= x < width and 0 <= y < height):
        return None
    left = max(0, x - POINTER_MASK_BEFORE_PX)
    upper = max(0, y - POINTER_MASK_BEFORE_PX)
    right = min(width, x + POINTER_MASK_AFTER_PX + 1)
    lower = min(height, y + POINTER_MASK_AFTER_PX + 1)
    return left, upper, right, lower


class HeuristicMonitorAgent(MonitorAgent):
    def __init__(self, *, screenshot_diff_config: Optional[ScreenshotDiffConfig] = None) -> None:
        # Benchmark-scoped (design.md §10): OSWorld 2.0's build_default hands
        # over a config; WebArena constructs HeuristicMonitorAgent() and gets
        # the default constants with no pointer mask.
        self.screenshot_diff_config = screenshot_diff_config or _DEFAULT_SCREENSHOT_DIFF_CONFIG

    @staticmethod
    def _full_text(observation: Observation) -> str:
        content = getattr(observation, "content", None)
        if not isinstance(content, dict):
            return ""
        for key in _FULL_TEXT_KEYS:
            value = content.get(key)
            if value:
                return str(value)
        return ""

    def evaluate(self, monitor_input: MonitorInput) -> AgentMessage:
        feedback = monitor_input.current_observation.metadata.get("env_feedback", {})
        reward = feedback.get("reward", 0)
        done = bool(feedback.get("done", False))
        info = feedback.get("info", {}) or {}

        # Diff the FULL observation text (uncapped a11y_tree) for `changed`,
        # so a change beyond the compact 200-node cap is not misread as STALL.
        previous_text = self._full_text(monitor_input.prev_observation)
        current_text = self._full_text(monitor_input.current_observation)

        action_types = {action.action_type for action in monitor_input.action_batch.actions}
        task_completed = bool(done and info.get("done", False))
        task_infeasible = bool(done and info.get("fail", False))

        if "done" in action_types:
            task_completed = True
        if "fail" in action_types:
            task_infeasible = True

        # When BOTH observations carry a screenshot the image diff is the
        # change signal (precedence over text — on a screenshot-primary benchmark
        # the text is empty and would read never-changed); else the text diff.
        previous_screenshot = _screenshot_bytes(monitor_input.prev_observation)
        current_screenshot = _screenshot_bytes(monitor_input.current_observation)
        config = self.screenshot_diff_config
        diff_pixel_count: Optional[int] = None
        diff_fraction: Optional[float] = None
        diff_pixel_count_unmasked: Optional[int] = None
        pointer_points: Sequence[Tuple[int, int]] = ()
        if previous_screenshot is not None and current_screenshot is not None:
            signal = "screenshot"
            # The locator is consulted only here, inside the screenshot branch —
            # a text-signal step never calls it, so it cannot fire in WebArena's
            # a11y mode even if one were somehow configured.
            if config.pointer_locator is not None:
                pointer_points = tuple(config.pointer_locator(monitor_input) or ())
            changed, diff_pixel_count, diff_fraction, diff_pixel_count_unmasked = screenshot_diff(
                previous_screenshot, current_screenshot,
                config=config, pointer_points=pointer_points,
            )
        else:
            signal = "text"
            changed = previous_text != current_text
        if task_completed or reward:
            progress_status = "PROGRESS"
        elif task_infeasible:
            progress_status = "REGRESSION"
        elif changed:
            progress_status = "PROGRESS"
        else:
            progress_status = "STALL"

        content = {
            "progress_status": progress_status,
            # Raw external diff signal (Mechanism 1): whether the
            # observation changed at all this step. Distinct from
            # progress_status (which also folds in completion/reward). The
            # verification reconciliation keys on this to override a positive
            # actual-effect claim on a page that did not change.
            "changed": changed,
            # Resolution-level telemetry for the diff: which signal decided
            # `changed` and, on the screenshot branch, the measured pixel
            # counts (None on the text branch) — the data the screenshot threshold
            # is validated against.
            "signal": signal,
            "diff_pixel_count": diff_pixel_count,
            "diff_fraction": diff_fraction,
            "stage_completed": task_completed,
            "task_completed": task_completed,
            "task_infeasible": task_infeasible,
            "need_retrieval": progress_status == "STALL",
            "observed_change": self._describe_change(
                previous_text, current_text, done, info,
                changed=changed, signal=signal, diff_pixel_count=diff_pixel_count,
            ),
            "next_stage": None,
        }
        # Mask audit trail, recorded ONLY where a pointer mask exists (so a
        # benchmark with no locator emits the default key set, unchanged).
        # Both counts every step, per design: the operative masked count and the
        # count the frozen floor would have seen. A step where the two straddle
        # the floor is a step where the mask decided the verdict, and the pair
        # makes it re-judgeable from the record without re-running the episode.
        if config.pointer_locator is not None:
            content["diff_pixel_count_unmasked"] = diff_pixel_count_unmasked
            content["pointer_mask_applied"] = bool(pointer_points)

        return AgentMessage(
            sender="monitor",
            message_type="progress",
            content=content,
        )

    def _describe_change(
        self,
        previous_text: str,
        current_text: str,
        done: bool,
        info: dict,
        *,
        changed: Optional[bool] = None,
        signal: str = "text",
        diff_pixel_count: Optional[int] = None,
    ) -> str:
        if done and info.get("done", False):
            return "environment signaled completion"
        if done and info.get("fail", False):
            return "environment signaled failure"
        if info.get("action_parse_error"):
            return f"previous action failed to ground: {info['action_parse_error']}"
        if info.get("grounding_parse_error"):
            # Action-side parse exhaustion: the grounding response itself was
            # unparseable JSON (distinct from a parsed action string the env
            # rejected) — surfaced as such so the model re-emits, not re-plans.
            return f"previous step's grounding response was unparseable: {info['grounding_parse_error']}"
        if info.get("reasoning_parse_error"):
            # A reasoning-stage failure reads as such, not as a grounding failure.
            return f"previous step's reasoning was unparseable: {info['reasoning_parse_error']}"
        if info.get("completion_gate_blocked"):
            # Mechanism 2: a blocked done is corrective — name the failed rule so
            # the next reasoning step gathers the missing grounded evidence.
            return f"completion claim lacked grounded evidence: {info['completion_gate_blocked']}"
        if changed is None:
            changed = previous_text != current_text
        if changed:
            if signal == "screenshot":
                return f"screen changed ({diff_pixel_count} px differ)"
            return "observation text changed"
        return "no observable change"
