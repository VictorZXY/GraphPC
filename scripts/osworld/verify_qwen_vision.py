"""Standalone loud-fail gate: does the Qwen backbone SEE the screenshot at the
correct scale on both call routes, as the Qwen profile builds them?

Run BEFORE any OSWorld 2.0 Qwen launch (the operator runs it at launch — it makes
paid calls). The shape of verify_minimax_vision.py: a synthetic
1920x1080 screenshot with two 120 px coloured squares at known centres, sent through
OUR classes exactly as a run does, read off the Qwen profile
(``profile_qwen.qwen_model_profile``) rather than restated here:

  1. reasoning route — the profile's backbone, ``complete_timeline_json`` with
     thinking on + ``response_format=json_object``, the schema in the system prompt
     and the screenshot as the LAST user content part, with the profile's output cap
     and sampling (what the deciding call sends). Criterion = "sees the frame": the
     reply names both colours with the correct screen quadrant for each.
  2. grounding route — the profile's action agent, one ``left_click`` intended action
     per square (the control's XML tool call, reasoning off, the control's sampling),
     then the profile's translator scales the relative coordinate to screen pixels —
     the FULL chain a real step runs. Criterion = the MEDIAN error over 3 samples per
     square <= 60 px, i.e. the median point lies inside the square.

Each route is sampled 3 times (9 or more calls: a grounding reply the agent repairs
costs another). All samples are printed, with the frame's encoded size (the control's
smart_resize: 1920x1088, JPEG since its delta D9) and the backbone's usage counters.

    python scripts/osworld/verify_qwen_vision.py

HARD-FAILS (exit 1, unambiguous FAIL lines) if any of:
  * OPENROUTER_API_KEY missing (repo .env or environment);
  * any reasoning sample fails to name both colours in their quadrants (a rejected
    call or an unusable reply counts as a failed sample);
  * a grounding median exceeds 60 px for either square — a sample whose call is
    rejected, degrades to an empty batch, emits a non-click action, or fails
    translation counts as an INFINITE error.
Cost: about 9 short vision calls (cents). LAUNCH-DAY RULE, as MiniMax's: exit 1 = one
re-run allowed; two consecutive failures block the launch.
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

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from dotenv import load_dotenv  # noqa: E402
import openai  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

from src.agents.action.base import ActionInput  # noqa: E402
from src.agents.backbone.qwen.main import OpenRouterCallError  # noqa: E402
from src.agents.common.json_utils import JsonParseExhausted  # noqa: E402
from src.agents.common.schema import AgentMessage, Observation, RuntimeState  # noqa: E402
from src.benchmarks.osworld.backbone_qwen import UndecodableResponseError, UnrecognisedProviderError  # noqa: E402
from src.benchmarks.osworld.profile_qwen import qwen_model_profile  # noqa: E402

MODEL = "qwen/qwen3.7-plus"
API_KEY_ENV = "OPENROUTER_API_KEY"
# What a provider failure leaves the profile's transport as, once its retries are spent: a failed sample.
PROVIDER_FAILURES = (openai.OpenAIError, OpenRouterCallError, UndecodableResponseError, UnrecognisedProviderError)
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

_CLICK_RE = re.compile(r"^pyautogui\.click\((\d+), (\d+)\)$")
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


def sample_grounding_route(agent, translate, observation: Observation, name: str, truth) -> tuple[float, str]:
    """One grounding call for one square -> (error_px, detail line)."""
    action_input = ActionInput(
        instruction=f"Click the {name} square.",
        observation=observation,
        runtime_state=RuntimeState(instruction=f"Click the {name} square."),
        reasoning_message=AgentMessage(
            sender="reasoning", message_type="local_reasoning",
            content={"intended_actions": [{
                "action_class": "left_click", "target": f"the {name} square",
                "text": None, "keys": None, "value": None,
                "intent": f"click the center of the {name} square",
                "predicted_effect_class": "unknown",
            }]},
        ),
        max_actions=1,
    )
    try:
        batch = agent.act(action_input)
    except PROVIDER_FAILURES as exc:
        return _INF, f"call failed ({type(exc).__name__}): {exc}"
    if not batch.actions:
        return _INF, f"degraded to an empty batch: {batch.content!r}"
    action = batch.actions[0]
    if action.action_type != "left_click":
        return _INF, f"expected left_click, got {action.action_type!r} {action.parameters!r}"
    try:
        code = translate(action, screen_size=SCREEN_SIZE)
    except ValueError as exc:
        return _INF, f"translation failed: {exc}"
    match = _CLICK_RE.match(code)
    if not match:
        return _INF, f"unexpected pyautogui code {code!r}"
    screen_point = (int(match.group(1)), int(match.group(2)))
    err = error_px(screen_point, truth)
    repairs = batch.metadata.get("resample_attempts", 0)
    return err, (f"relative={action.parameters.get('coordinate')} screen={screen_point} truth={truth} "
                 f"err={_fmt(err)} repairs={repairs}")


def main() -> int:
    load_dotenv(_REPO_ROOT / ".env")
    if not os.environ.get(API_KEY_ENV):
        print(f"FAIL: {API_KEY_ENV} not set (repo .env or environment).")
        return 1

    profile = qwen_model_profile()
    backbone = profile.backbone_class(
        model=MODEL, grounding_temperature=profile.grounding_temperature,
        max_tokens=profile.max_tokens, request_timeout=profile.request_timeout,
    )
    agent = profile.action_agent_class(backbone=backbone, model_call_attempts=profile.call_attempts)
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

    medians: list[str] = []
    for name, _color, truth, _quadrant in SQUARES:
        samples: list[float] = []
        for sample in range(1, SAMPLES_PER_ROUTE + 1):
            err, detail = sample_grounding_route(agent, profile.translate_action, observation, name, truth)
            samples.append(err)
            print(f"  grounding  {name:<4} sample {sample}: {detail}")
        median = statistics.median(samples)
        medians.append(f"grounding/{name} median={_fmt(median)} samples=[{', '.join(_fmt(e) for e in samples)}]")
        if median > MEDIAN_TOLERANCE_PX:
            failures.append(f"grounding route: {name} square median error {_fmt(median)} > {MEDIAN_TOLERANCE_PX}px")

    for line in medians:
        print(f"  {line}")
    print(f"  config: model={MODEL} screen={SCREEN_SIZE[0]}x{SCREEN_SIZE[1]} frame sent as {frame}; "
          f"output cap={profile.max_tokens} temperature={type(backbone).TEMPERATURE} top_p={type(backbone).TOP_P} "
          f"request timeout={profile.request_timeout}s; reasoning=complete_timeline_json, thinking on + "
          f"json_object, image last; grounding=the control's XML tool call, reasoning off, repaired up to "
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
          f"for both squares ({SAMPLES_PER_ROUTE} samples each).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
