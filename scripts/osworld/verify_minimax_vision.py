"""Standalone loud-fail gate: does the MiniMax backbone SEE the screenshot at
the correct scale on both call routes?

Run BEFORE any OSWorld 2.0 MiniMax launch. Builds
a synthetic 1920x1080 screenshot with two 120 px colored squares at known
centers and sends it through OUR backbone exactly as a run does:

  1. reasoning route — ``OSWorldMiniMaxAgent.complete_timeline_json`` with
     thinking on + ``response_format=json_object``, the schema in the system
     prompt and the screenshot as the LAST user content part (what the
     deciding call sends; no sampling params). Criterion = "sees the frame": the reply must name
     both colours with the correct screen quadrant for each (reasoning never
     emits coordinates in the pipeline, so no pixel tolerance applies).
  2. grounding route — ``MiniMaxActionAgent.act`` (OSWorld-V2's native
     format: ``complete_text(grounding=True)``, non-thinking, pinned to
     ``grounding_temperature=0.6`` — the OSWorld build_default instantiation
     (the V2 M3 runner's non-thinking default; the baseline arm itself grounds
     thinking-adaptive at the forced temperature 1)), one ``click`` intended action per
     square, then ``translate_m3_action`` scales the emitted relative
     coordinate to screen pixels — the FULL chain a real step runs.
     Criterion = the MEDIAN error over 3 samples per square <= 60 px, i.e.
     the median point lies inside the square.

Each route is sampled 3 times (9 calls). All samples are printed.

    python scripts/osworld/verify_minimax_vision.py
    python scripts/osworld/verify_minimax_vision.py --model MiniMax-M3

HARD-FAILS (exit 1, unambiguous FAIL lines) if any of:
  * MINIMAX_API_KEY missing (repo .env or environment);
  * any reasoning sample fails to name both colours in their quadrants (a
    rejected call or unparseable reply counts as a failed sample);
  * a grounding median exceeds 60 px for either square — a sample whose call
    is rejected, degrades to an empty batch, emits a non-click action, or
    fails translation counts as an INFINITE error, so two such samples out of
    three fail that square.

On success prints PASS with the samples, the active configuration and the
backbone's usage counters. It ships under scripts/ (synced to the host)
because it gates a launch and must run where the launch runs.
Cost: 9 short vision calls (cents).

LAUNCH-DAY RULE: exit 1 = one re-run allowed; two consecutive failures block
the launch. Rationale: a single failure on one square (outlier grounding
samples just outside a corner of the square, every reasoning sample passing)
is model noise on a flat synthetic frame; a repeated failure is not.
"""

from __future__ import annotations

import argparse
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
from src.agents.common.schema import AgentMessage, Observation, RuntimeState  # noqa: E402
from src.benchmarks.osworld.action_minimax import MiniMaxActionAgent  # noqa: E402
from src.benchmarks.osworld.m3_action_translator import translate_m3_action  # noqa: E402
from src.benchmarks.osworld.reasoning import OSWorldMiniMaxAgent  # noqa: E402

SCREEN_SIZE = (1920, 1080)
SQUARE_HALF = 60  # 120 px squares
SAMPLES_PER_ROUTE = 3
MEDIAN_TOLERANCE_PX = 60  # inside the square
# The OSWorld build_default instantiation: the V2 M3 runner's default
# --temperature (scripts/python/run_multienv_m3.py; its NON-thinking value).
# Not a parity pin — the baseline arm runs --thinking_mode adaptive, where
# M3Agent forces temperature 1.
GROUNDING_TEMPERATURE = 0.6
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
        draw.rectangle(
            [cx - SQUARE_HALF, cy - SQUARE_HALF, cx + SQUARE_HALF, cy + SQUARE_HALF], fill=color,
        )
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def error_px(point, truth) -> float:
    return ((point[0] - truth[0]) ** 2 + (point[1] - truth[1]) ** 2) ** 0.5


def _fmt(err: float) -> str:
    return "inf" if err == _INF else f"{err:.1f}px"


def sample_reasoning_route(backbone: OSWorldMiniMaxAgent, observation: Observation) -> tuple[bool, str]:
    """One reasoning call -> (sees_the_frame, detail line). Sent in the deciding
    route's shape: schema in the system prompt, the screenshot last."""
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
    except (openai.OpenAIError, ValueError) as exc:
        return False, f"call failed ({type(exc).__name__}): {exc}"
    payload = result["json"]
    squares = payload.get("squares") if isinstance(payload, dict) else None
    if not isinstance(squares, list):
        return False, f"no squares list in {payload!r}"
    named = {}
    for entry in squares:
        if isinstance(entry, dict):
            named[str(entry.get("colour", "")).strip().lower()] = str(entry.get("quadrant", "")).strip().lower()
    ok = True
    for name, _color, _center, quadrant in SQUARES:
        seen = next((named[key] for key in named if name in key), None)
        if seen != quadrant:
            ok = False
    return ok, f"{'OK ' if ok else 'BAD'} named={named} truth={{{', '.join(f'{n}: {q}' for n, _c, _p, q in SQUARES)}}}"


def sample_grounding_route(agent: MiniMaxActionAgent, observation: Observation, name: str, truth) -> tuple[float, str]:
    """One grounding call for one square -> (error_px, detail line)."""
    action_input = ActionInput(
        instruction=f"Click the {name} square.",
        observation=observation,
        runtime_state=RuntimeState(instruction=f"Click the {name} square."),
        reasoning_message=AgentMessage(
            sender="reasoning", message_type="local_reasoning",
            content={"intended_actions": [{
                "action_class": "click", "target": f"the {name} square",
                "text": None, "keys": None, "value": None,
                "intent": f"click the center of the {name} square",
                "predicted_effect_class": "unknown",
            }]},
        ),
        max_actions=1,
    )
    try:
        batch = agent.act(action_input)
    except openai.OpenAIError as exc:
        return _INF, f"call failed ({type(exc).__name__}): {exc}"
    if not batch.actions:
        return _INF, f"degraded to an empty batch: {batch.content!r}"
    action = batch.actions[0]
    if action.action_type != "left_click":
        return _INF, f"expected left_click, got {action.action_type!r} {action.parameters!r}"
    try:
        code = translate_m3_action(action, screen_size=SCREEN_SIZE)
    except ValueError as exc:
        return _INF, f"translation failed: {exc}"
    match = _CLICK_RE.match(code)
    if not match:
        return _INF, f"unexpected pyautogui code {code!r}"
    screen_point = (int(match.group(1)), int(match.group(2)))
    err = error_px(screen_point, truth)
    return err, (f"relative={action.parameters.get('coordinate')} screen={screen_point} "
                 f"truth={truth} err={_fmt(err)}  [{batch.content.get('reasoning_text', '')!r}]")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", type=str, default="MiniMax-M3", help="MiniMax model id to probe.")
    args = parser.parse_args()

    load_dotenv(_REPO_ROOT / ".env")
    if not os.environ.get("MINIMAX_API_KEY"):
        print("FAIL: MINIMAX_API_KEY not set (repo .env or environment).")
        return 1

    backbone = OSWorldMiniMaxAgent(
        model=args.model, grounding_temperature=GROUNDING_TEMPERATURE,
    )
    agent = MiniMaxActionAgent(backbone=backbone)
    observation = Observation(
        backend="osworld",
        content={"screenshot": build_screenshot(), "observation_text": ""},
        metadata={"screen_size": SCREEN_SIZE},
    )

    failures: list[str] = []
    reasoning_ok: list[bool] = []
    for sample in range(1, SAMPLES_PER_ROUTE + 1):
        ok, detail = sample_reasoning_route(backbone, observation)
        reasoning_ok.append(ok)
        print(f"  reasoning  sample {sample}: {detail}")
    if not all(reasoning_ok):
        failures.append(
            f"reasoning route: {reasoning_ok.count(False)} of {SAMPLES_PER_ROUTE} samples did not name both "
            "colours in their quadrants"
        )

    medians: list[str] = []
    for name, _color, truth, _quadrant in SQUARES:
        samples: list[float] = []
        for sample in range(1, SAMPLES_PER_ROUTE + 1):
            err, detail = sample_grounding_route(agent, observation, name, truth)
            samples.append(err)
            print(f"  grounding  {name:<4} sample {sample}: {detail}")
        median = statistics.median(samples)
        medians.append(f"grounding/{name} median={_fmt(median)} samples=[{', '.join(_fmt(e) for e in samples)}]")
        if median > MEDIAN_TOLERANCE_PX:
            failures.append(f"grounding route: {name} square median error {_fmt(median)} > {MEDIAN_TOLERANCE_PX}px")

    for line in medians:
        print(f"  {line}")
    config = (
        f"config: model={args.model} screen={SCREEN_SIZE[0]}x{SCREEN_SIZE[1]} jpeg_q90 "
        f"reasoning=complete_timeline_json, thinking(adaptive)+json_object, image last, no sampling params; "
        f"grounding=V2 native <tool_call> text via complete_text(grounding=True), non-thinking, "
        f"temperature={GROUNDING_TEMPERATURE}, stop=</tool_call>"
    )
    usage = (
        f"usage: calls={backbone.call_count} input_tokens={backbone.total_input_tokens} "
        f"output_tokens={backbone.total_output_tokens}"
    )
    print(f"  {config}")
    print(f"  {usage}")
    if failures:
        print(f"FAIL: {len(failures)} problem(s) on model {args.model}:")
        for failure in failures:
            print(f"  - {failure}")
        return 1
    print(f"PASS: model {args.model} — reasoning names both colours in their quadrants "
          f"({SAMPLES_PER_ROUTE}/{SAMPLES_PER_ROUTE}); grounding median error <= {MEDIAN_TOLERANCE_PX}px "
          f"for both squares ({SAMPLES_PER_ROUTE} samples each).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
