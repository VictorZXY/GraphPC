"""OSWorld-V2 observation -> our ``Observation`` (the inward boundary).

Screenshot-primary (resolution parity with the control): the only
percept on OSWorld 2.0 is the raw PNG the guest renders at the configured
screen size, carried on ``content["screenshot"]`` untouched — the backbone
re-encodes it at native resolution, no resize. There is no accessibility
channel on this benchmark (the V2 runner's ``--observation_type screenshot``
leaves ``obs["accessibility_tree"]`` ``None``), so there is no
a11y projection on this path.

``content["observation_text"]`` stays a (constantly empty) text field so
every text consumer keeps its contract (the reasoning prompt, retrieval query
composition, the completion gate's answer grounding); retrieval query and
candidate text therefore reduce to the instruction on this benchmark. The
monitor's change signal comes from the screenshot itself (see
``src/agents/monitor/heuristic.py``).

``content["environment_brief"]`` carries the environment FACTS the bare
baseline's system prompt tells its model, rendered by the profile's
``environment_brief`` (MiniMax and DeepSeek: OSWorld-V2/mm_agents/m3/prompts.py's
platform, browser, home directory, sudo password, current date; Qwen: its
control's one fact, the date) — not its
tool list or feasibility instructions — so the reasoner never sees less than
the control (design.md §2 parity floor). The OSWorld reasoning agent renders it
into the deciding call's SYSTEM prompt, where the control keeps it
(``reasoning.py``). It is a separate key on purpose: a constant prefix inside
``observation_text`` or the instruction would shift every retrieval cosine
score and the relevance gate. The brief ends with one sentence the control does
NOT carry, ``environment_note`` (the profile's typing note, from
``build_default``; empty on Qwen): how a long `type` behaves on this benchmark.

The driver's ``user_response`` key (the answer to an ask, on a profile with an
ask-user channel: Qwen) is left on ``raw``; the adapter reads it from the
driver's obs, not from this ``Observation``.
"""

from __future__ import annotations

import io

from datetime import datetime
from typing import Any, Callable, Optional

from PIL import Image

from src.agents.common.schema import Observation, RuntimeState

# The V2 images' sudo password; ``DesktopEnv`` normalises an empty
# ``--client_password`` to this value (desktop_env.py:83).
DEFAULT_CLIENT_PASSWORD = "osworld-public-evaluation"

# The obs-dict key the agent uses to hand the driver-set ``task_current_date``
# to this adapter (popped here; never reaches ``raw``).
TASK_CURRENT_DATE_KEY = "_task_current_date"


def screenshot_size(screenshot: Any) -> Optional[tuple[int, int]]:
    """``(width, height)`` of a PNG/JPEG screenshot from its header (PIL reads
    the header lazily; the pixel data is not decoded). ``None`` when the value
    is not decodable image bytes."""
    if not isinstance(screenshot, (bytes, bytearray)) or not screenshot:
        return None
    try:
        with Image.open(io.BytesIO(bytes(screenshot))) as image:
            width, height = image.size
    except Exception:  # noqa: BLE001 — undecodable bytes: report "unknown", the caller falls back.
        return None
    return int(width), int(height)


def format_task_date(value: Any) -> Optional[str]:
    """The M3 agent's date rendering (``_format_system_date``): an ISO date
    (first 10 chars) -> "Monday, June 01, 2025"; an unparseable non-empty
    string is passed through; ``None``/empty -> ``None``."""
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.strftime("%A, %B %d, %Y")
    raw = str(value).strip()
    if not raw:
        return None
    try:
        return datetime.fromisoformat(raw[:10]).strftime("%A, %B %d, %Y")
    except ValueError:
        return raw


def environment_brief(*, client_password: str, task_current_date: Any = None) -> str:
    """The environment facts of the MiniMax-M3 control's system prompt, as one
    string — that model profile's brief (``graphpc_agent.OSWorldModelProfile``)."""
    date_str = format_task_date(task_current_date) or datetime.now().strftime("%A, %B %d, %Y")
    return (
        "Ubuntu virtual machine (x86_64) with internet access. The installed browser is "
        "Chrome (open it via the Chrome icon). Home directory: /home/user. "
        f"If you need a password for sudo, the password of the computer is '{client_password}'. "
        f"The current date is {date_str}."
    )


class OSWorldObservationAdapter:
    """Convert a V2 driver obs dict to our ``Observation`` envelope.

    Input contract: the dict ``DesktopEnv._get_obs()`` returns
    (``screenshot`` bytes, ``accessibility_tree``/``terminal`` ``None``,
    ``instruction``), plus the agent-injected ``_env_feedback`` (the prior
    step's degrade signal, popped into ``metadata["env_feedback"]`` for the
    monitor) and ``_task_current_date`` (popped into the environment brief).
    Unknown keys survive on ``Observation.raw``.

    ``screen_size`` is the configured guest resolution; the actual size read
    off the screenshot header takes precedence in ``metadata["screen_size"]``
    (the value the action translator scales relative coordinates against).
    ``environment_note`` ends the environment brief, after one space.
    ``environment_brief`` renders the facts before it: the model profile's
    (``graphpc_agent.OSWorldModelProfile``), since each model's control states
    its own; the default is M3's, the one ``build_default`` passes for MiniMax.
    """

    def __init__(
        self,
        *,
        environment_note: str,
        screen_size: tuple[int, int] = (1920, 1080),
        client_password: str = DEFAULT_CLIENT_PASSWORD,
        environment_brief: Callable[..., str] = environment_brief,
    ) -> None:
        self.screen_size = (int(screen_size[0]), int(screen_size[1]))
        self.client_password = client_password or DEFAULT_CLIENT_PASSWORD
        self.environment_note = environment_note
        self.environment_brief = environment_brief

    def __call__(
        self,
        observation_input: dict[str, Any],
        runtime_state: RuntimeState,
    ) -> Observation:
        # Shallow copy: the screenshot is an immutable bytes blob (a deepcopy
        # would return the same object anyway) and the caller's dict must not
        # see the pops.
        payload = dict(observation_input)
        env_feedback = payload.pop("_env_feedback", {})
        task_current_date = payload.pop(TASK_CURRENT_DATE_KEY, None)

        content: dict[str, Any] = {}
        screenshot = payload.get("screenshot")
        if isinstance(screenshot, (bytes, bytearray)) and screenshot:
            # Present only as real bytes: the backbone attaches an image iff
            # this key exists, and the monitor takes the screenshot-diff branch
            # iff both observations carry it.
            content["screenshot"] = bytes(screenshot)
        content["observation_text"] = ""
        content["environment_brief"] = " ".join(text for text in (
            self.environment_brief(client_password=self.client_password, task_current_date=task_current_date),
            self.environment_note,
        ) if text)

        return Observation(
            backend="osworld",
            content=content,
            raw=payload,
            metadata={
                "env_feedback": env_feedback,
                "step_idx": runtime_state.step_idx,
                "screen_size": screenshot_size(screenshot) or self.screen_size,
            },
        )
