"""Standalone loud-fail gate: does the DeepSeek backbone SEE the screenshot at the
correct scale on both call routes, as the DeepSeek profile builds them?

Run BEFORE any OSWorld 2.0 DeepSeek launch (the operator runs it at launch — it makes
paid calls). The shape of verify_minimax_vision.py: a synthetic
1920x1080 screenshot with two 120 px coloured squares at known centres, sent through
OUR classes exactly as a run does, read off the DeepSeek profile
(``profile_deepseek.deepseek_model_profile``) rather than restated here:

  1. reasoning route — the profile's backbone, ``complete_timeline_json`` with
     thinking on (effort high) + ``response_format=json_object``, the schema in the
     system prompt and the screenshot as the LAST user content part, no output cap
     (what the deciding call sends). Criterion = "sees the frame": the reply names
     both colours with the correct screen quadrant for each.
  2. grounding route — the profile's action agent, ``DeepSeekActionAgent.act`` (the
     forced native ``computer`` function, thinking off, the control's sampling), then
     the profile's translator scales the relative coordinate to screen pixels — the
     FULL chain a real step runs. Three cases, each sampled 3 times:
       * CLICK: one ``left_click`` intended action per square; the error is the
         distance from the executed point to the square's centre (a click of another
         kind inside the decided class's family executes, as in a run, and is measured
         the same way; the executed class is printed).
       * SCROLL: one ``scroll`` intended action on the red square, direction down;
         the executed code must be a scroll AT A POINT (``pyautogui.scroll(n, x, y)``)
         in the decided direction (a negative amount); the error is the point's
         distance to the square's centre — a scroll with no point, or up, is INFINITE.
       * DRAG: one ``left_click_drag`` intended action from the red square to the
         blue one; the executed code must move to a start point and drag to an end
         point (``moveTo`` then ``dragTo``); the error is the larger of the start's
         distance to the red centre and the end's to the blue one — a drag with no
         start point is INFINITE.
     Criterion, every case = the MEDIAN error over its 3 samples <= 60 px, i.e. the
     median point lies inside its square.

All samples are printed, with the frame's encoded size (JPEG, as the backbone encodes it)
and the backbone's usage counters; a grounding reply the agent re-samples or repairs
costs another call.

    python scripts/osworld/verify_deepseek_vision.py

HARD-FAILS (exit 1, unambiguous FAIL lines) if any of:
  * DEEPSEEK_API_KEY missing (repo .env or environment);
  * any reasoning sample fails to name both colours in their quadrants (a rejected
    call or an unusable reply counts as a failed sample);
  * a grounding median exceeds 60 px for any case — a sample whose call is rejected,
    degrades to an empty batch, emits an action of another class, or fails
    translation counts as an INFINITE error.
Cost: about 15 short vision calls (cents). LAUNCH-DAY RULE, as MiniMax's and Qwen's:
exit 1 = one re-run allowed; two consecutive failures block the launch.
"""

from __future__ import annotations

import base64
import io
import json
import os
import re
import statistics
import sys
from pathlib import Path
from typing import Any, Callable, Optional

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from dotenv import load_dotenv  # noqa: E402
import openai  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

from src.agents.action.base import ActionInput  # noqa: E402
from src.agents.common.json_utils import JsonParseExhausted  # noqa: E402
from src.agents.common.schema import AgentMessage, Observation, RuntimeState  # noqa: E402
from src.benchmarks.osworld.action_deepseek import is_class_mismatch  # noqa: E402
from src.benchmarks.osworld.backbone_deepseek import (  # noqa: E402
    DeepSeekReplyRejected, UndecodableResponseError, UnrecognisedProviderError,
)
from src.benchmarks.osworld.profile_deepseek import deepseek_model_profile  # noqa: E402

MODEL = "deepseek-flash"  # the launcher's BACKBONE_MODELS entry, as verify_qwen_vision.py carries its own
API_KEY_ENV = "DEEPSEEK_API_KEY"
# What a provider failure leaves the profile's transport as, once its attempts are spent: a failed sample.
PROVIDER_FAILURES = (openai.OpenAIError, DeepSeekReplyRejected, UndecodableResponseError, UnrecognisedProviderError)
SCREEN_SIZE = (1920, 1080)
SQUARE_HALF = 60  # 120 px squares
SAMPLES_PER_ROUTE = 3
MEDIAN_TOLERANCE_PX = 60  # inside the square
# (name, fill color, true center in screen pixels, true quadrant)
SQUARES = (
    ("red", (220, 30, 30), (480, 300), "top-left"),
    ("blue", (30, 60, 220), (1440, 780), "bottom-right"),
)
QUADRANTS = ("top-left", "top-right", "bottom-left", "bottom-right")
SCROLL_SQUARE, SCROLL_DIRECTION = "red", "down"
DRAG_FROM, DRAG_TO = "red", "blue"

_REASONING_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["squares"],
    "properties": {
        "squares": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["colour", "quadrant"],
                "properties": {
                    "colour": {"type": "string"},
                    "quadrant": {"type": "string", "enum": list(QUADRANTS)},
                },
            },
        },
    },
}

# The translator's click AT A POINT, any kind inside the click family (``click`` / ``rightClick`` / ...).
_CLICK_RE = re.compile(r"^pyautogui\.(?:click|rightClick|doubleClick|middleClick|tripleClick)\((\d+), (\d+)\)$")
# The translator's scroll AT A POINT (``scroll(amount, x, y)``); a scroll without a point does not match.
_SCROLL_RE = re.compile(r"^pyautogui\.scroll\((-?\d+), (\d+), (\d+)\)$")
# The translator's drag: ``moveTo(sx, sy, duration=...)`` then ``dragTo(x, y, duration=...)``; a drag without
# a start point emits ``dragTo`` alone and does not match.
_DRAG_RE = re.compile(r"^pyautogui\.moveTo\((\d+), (\d+), duration=[0-9.]+\)\npyautogui\.dragTo\((\d+), (\d+), duration=[0-9.]+\)$")
_INF = float("inf")


def build_screenshot() -> bytes:
    image = Image.new("RGB", SCREEN_SIZE, (235, 235, 235))
    draw = ImageDraw.Draw(image)
    for _name, color, (cx, cy), _quadrant in SQUARES:
        draw.rectangle([cx - SQUARE_HALF, cy - SQUARE_HALF, cx + SQUARE_HALF, cy + SQUARE_HALF], fill=color)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def error_px(point, truth) -> float:
    return ((point[0] - truth[0]) ** 2 + (point[1] - truth[1]) ** 2) ** 0.5


def _fmt(err: float) -> str:
    return "inf" if err == _INF else f"{err:.1f}px"


def _centre(name: str) -> tuple[int, int]:
    return next(centre for square, _color, centre, _quadrant in SQUARES if square == name)


def sample_reasoning_route(backbone, observation: Observation) -> tuple[bool, str]:
    """One reasoning call -> (sees_the_frame, detail line). Sent in the deciding route's
    shape: schema in the system prompt, the screenshot last."""
    try:
        result = backbone.complete_timeline_json(
            system_prompt=(
                "You see a screenshot of a 1920x1080 screen. Divide it into four equal "
                "quadrants: top-left, top-right, bottom-left, bottom-right.\n\n"
                "Return a single JSON object conforming to this schema (JSON only, no prose):\n"
                + json.dumps(_REASONING_SCHEMA)
            ),
            user_parts=[
                {"type": "text", "text": (
                    "The screenshot contains two colored squares. For each square, name its "
                    "colour and the quadrant it lies in.\nCURRENT SCREEN:"
                )},
                {"type": "image", "image": observation.content["screenshot"], "label": "current",
                 "predict_call": 1},
            ],
            use_thinking=True,
        )
    except (*PROVIDER_FAILURES, JsonParseExhausted, ValueError) as exc:
        return False, f"call failed ({type(exc).__name__}): {exc}"
    payload = result["json"]
    squares = payload.get("squares") if isinstance(payload, dict) else None
    if not isinstance(squares, list):
        return False, f"no squares list in {payload!r}"
    named = {}
    for entry in squares:
        if isinstance(entry, dict):
            named[str(entry.get("colour", "")).strip().lower()] = str(entry.get("quadrant", "")).strip().lower()
    ok = all(next((named[key] for key in named if name in key), None) == quadrant
             for name, _color, _center, quadrant in SQUARES)
    return ok, f"{'OK ' if ok else 'BAD'} named={named} truth={{{', '.join(f'{n}: {q}' for n, _c, _p, q in SQUARES)}}}"


def _ground(agent, translate, observation: Observation, intended: dict[str, Any], expected_class: str):
    """One grounding call for one intended action -> (translated code | None, detail prefix, action | None,
    repairs). The failure cases every grounding sample shares, named in the detail."""
    instruction = f"{intended['action_class']} {intended['target']}"
    action_input = ActionInput(
        instruction=instruction,
        observation=observation,
        runtime_state=RuntimeState(instruction=instruction),
        reasoning_message=AgentMessage(
            sender="reasoning", message_type="local_reasoning",
            content={"intended_actions": [{"text": None, "keys": None, "value": None,
                                           "predicted_effect_class": "unknown", **intended}]},
        ),
        max_actions=1,
    )
    try:
        batch = agent.act(action_input)
    except PROVIDER_FAILURES as exc:
        return None, f"call failed ({type(exc).__name__}): {exc}", None, ""
    # Re-samples (every rejected reply, the repaired ones included) and, of those, the class-family repairs.
    repairs = (f"resamples={int(batch.metadata.get('resample_attempts', 0) or 0)} "
               f"repairs={int(batch.metadata.get('class_mismatch_repairs', 0) or 0)}")
    if not batch.actions:
        return None, f"degraded to an empty batch: {batch.content!r}", None, repairs
    action = batch.actions[0]
    # The class-family rule is the run's: an action inside the decided class's family executes (a right_click for
    # a left_click), one outside it would have been repaired away, so reaching here with one is a defect.
    if is_class_mismatch(expected_class, action.action_type):
        return None, f"expected {expected_class}, got {action.action_type!r} {action.parameters!r}", action, repairs
    try:
        code = translate(action, screen_size=SCREEN_SIZE)
    except ValueError as exc:
        return None, f"translation failed: {exc}", action, repairs
    return code, "", action, repairs


def sample_click(agent, translate, observation: Observation, name: str) -> tuple[float, str]:
    """One CLICK sample for one square -> (error_px, detail line)."""
    truth = _centre(name)
    code, failure, action, repairs = _ground(agent, translate, observation, {
        "action_class": "left_click", "target": f"the {name} square",
        "intent": f"click the center of the {name} square"}, "left_click")
    if code is None:
        return _INF, failure
    match = _CLICK_RE.match(code)
    if not match:
        return _INF, f"not a click at a point: {code!r} (executed {action.action_type!r} {action.parameters!r})"
    screen_point = (int(match.group(1)), int(match.group(2)))
    err = error_px(screen_point, truth)
    return err, (f"{action.action_type} relative={action.parameters.get('coordinate')} screen={screen_point} "
                 f"truth={truth} err={_fmt(err)} {repairs}")


def sample_scroll(agent, translate, observation: Observation) -> tuple[float, str]:
    """One SCROLL sample on the red square, direction down -> (error_px, detail line): the executed code
    must scroll at a point inside the square (the amount negative = down); the error is that point's
    distance to the centre."""
    truth = _centre(SCROLL_SQUARE)
    code, failure, action, repairs = _ground(agent, translate, observation, {
        "action_class": "scroll", "target": f"the {SCROLL_SQUARE} square",
        "value": f"{SCROLL_DIRECTION} 3",
        "intent": f"scroll {SCROLL_DIRECTION} by 3 with the pointer on the center of the {SCROLL_SQUARE} square"},
        "scroll")
    if code is None:
        return _INF, failure
    match = _SCROLL_RE.match(code)
    if not match:
        return _INF, f"not a scroll at a point: {code!r} (parameters {action.parameters!r})"
    amount, screen_point = int(match.group(1)), (int(match.group(2)), int(match.group(3)))
    direction = "down" if amount < 0 else "up" if amount > 0 else "none"
    if direction != SCROLL_DIRECTION:
        return _INF, (f"scrolled {direction} (amount {amount}), not {SCROLL_DIRECTION}: {code!r} "
                      f"(parameters {action.parameters!r})")
    err = error_px(screen_point, truth)
    return err, (f"relative={action.parameters.get('coordinate')} direction={direction} amount={amount} "
                 f"screen={screen_point} truth={truth} err={_fmt(err)} {repairs}")


def sample_drag(agent, translate, observation: Observation) -> tuple[float, str]:
    """One DRAG sample from the red square to the blue one -> (error_px, detail line): the executed code
    must move to a start point and drag to an end point; the error is the larger of the start's distance
    to the red centre and the end's to the blue one."""
    start_truth, end_truth = _centre(DRAG_FROM), _centre(DRAG_TO)
    code, failure, action, repairs = _ground(agent, translate, observation, {
        "action_class": "left_click_drag", "target": f"from the {DRAG_FROM} square to the {DRAG_TO} square",
        "intent": (f"drag from the center of the {DRAG_FROM} square to the center of the {DRAG_TO} square "
                   f"(start_coordinate on the {DRAG_FROM} square, coordinate on the {DRAG_TO} square)")},
        "left_click_drag")
    if code is None:
        return _INF, failure
    match = _DRAG_RE.match(code)
    if not match:
        return _INF, f"not a drag with a start point: {code!r} (parameters {action.parameters!r})"
    start = (int(match.group(1)), int(match.group(2)))
    end = (int(match.group(3)), int(match.group(4)))
    err = max(error_px(start, start_truth), error_px(end, end_truth))
    return err, (f"relative start={action.parameters.get('start_coordinate')} end={action.parameters.get('coordinate')} "
                 f"screen start={start} (truth {start_truth}) end={end} (truth {end_truth}) err={_fmt(err)} "
                 f"{repairs}")


def build(client_factory: Optional[Callable[[Optional[str]], Any]] = None):
    """The profile's backbone and action agent, built as ``build_default`` builds them (the profile's
    sampling, cap, timeout and attempts); ``client_factory``, when given, stands in for the SDK client."""
    profile = deepseek_model_profile()
    kwargs: dict[str, Any] = {"max_tokens": profile.max_tokens, "request_timeout": profile.request_timeout}
    if client_factory is not None:
        kwargs["client_factory"] = client_factory
    backbone = profile.backbone_class(
        model=MODEL, grounding_temperature=profile.grounding_temperature, **kwargs,
    )
    agent = profile.action_agent_class(backbone=backbone, model_call_attempts=profile.call_attempts)
    return profile, backbone, agent


def main(client_factory: Optional[Callable[[Optional[str]], Any]] = None) -> int:
    if client_factory is None:
        load_dotenv(_REPO_ROOT / ".env")
    if not os.environ.get(API_KEY_ENV):
        print(f"FAIL: {API_KEY_ENV} not set (repo .env or environment).")
        return 1

    profile, backbone, agent = build(client_factory)
    screenshot = build_screenshot()
    observation = Observation(
        backend="osworld", content={"screenshot": screenshot, "observation_text": ""},
        metadata={"screen_size": SCREEN_SIZE},
    )
    with Image.open(io.BytesIO(base64.b64decode(backbone._encode_screenshot(screenshot)))) as encoded:
        frame = f"{encoded.format} {encoded.size[0]}x{encoded.size[1]}"

    failures: list[str] = []
    reasoning_ok: list[bool] = []
    for sample in range(1, SAMPLES_PER_ROUTE + 1):
        ok, detail = sample_reasoning_route(backbone, observation)
        reasoning_ok.append(ok)
        print(f"  reasoning  sample {sample}: {detail}")
    if not all(reasoning_ok):
        failures.append(f"reasoning route: {reasoning_ok.count(False)} of {SAMPLES_PER_ROUTE} samples did not name "
                        "both colours in their quadrants")

    cases: list[tuple[str, Callable[[], tuple[float, str]]]] = [
        *((f"click/{name}", (lambda n=name: sample_click(agent, profile.translate_action, observation, n)))
          for name, _color, _truth, _quadrant in SQUARES),
        (f"scroll/{SCROLL_SQUARE}-{SCROLL_DIRECTION}", lambda: sample_scroll(agent, profile.translate_action, observation)),
        (f"drag/{DRAG_FROM}-to-{DRAG_TO}", lambda: sample_drag(agent, profile.translate_action, observation)),
    ]
    medians: list[str] = []
    for label, sample_once in cases:
        samples: list[float] = []
        for sample in range(1, SAMPLES_PER_ROUTE + 1):
            err, detail = sample_once()
            samples.append(err)
            print(f"  grounding  {label:<22} sample {sample}: {detail}")
        median = statistics.median(samples)
        medians.append(f"grounding/{label} median={_fmt(median)} samples=[{', '.join(_fmt(e) for e in samples)}]")
        if median > MEDIAN_TOLERANCE_PX:
            failures.append(f"grounding route: {label} median error {_fmt(median)} > {MEDIAN_TOLERANCE_PX}px")

    for line in medians:
        print(f"  {line}")
    print(f"  config: model={MODEL} screen={SCREEN_SIZE[0]}x{SCREEN_SIZE[1]} frame sent as {frame}; "
          f"output cap={profile.max_tokens} grounding temperature={backbone.grounding_temperature} "
          f"request timeout={profile.request_timeout}s; reasoning=complete_timeline_json, thinking on "
          f"({backbone._extra_body(True)}) + json_object, image last; grounding=the forced native "
          f"`computer` function, thinking off ({backbone._extra_body(False)}), re-sampled or repaired up to "
          f"{profile.call_attempts} attempts")
    print(f"  usage: calls={backbone.call_count} input_tokens={backbone.total_input_tokens} "
          f"output_tokens={backbone.total_output_tokens}")
    if failures:
        print(f"FAIL: {len(failures)} problem(s) on model {MODEL}:")
        for failure in failures:
            print(f"  - {failure}")
        return 1
    print(f"PASS: model {MODEL} — reasoning names both colours in their quadrants "
          f"({SAMPLES_PER_ROUTE}/{SAMPLES_PER_ROUTE}); grounding median error <= {MEDIAN_TOLERANCE_PX}px "
          f"for every case: click on both squares, scroll {SCROLL_DIRECTION} on the {SCROLL_SQUARE} square, "
          f"drag {DRAG_FROM} to {DRAG_TO} ({SAMPLES_PER_ROUTE} samples each).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
