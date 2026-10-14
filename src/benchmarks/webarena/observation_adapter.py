"""WebArena observation -> our ``Observation`` (the inward boundary).

Mirrors ``OSWorldObservationAdapter`` in shape; the only WebArena-specific
piece is which keys carry the a11y string and the screenshot. Used by the
orchestrator's ``perception_processor`` hook so the rest of the agent stack
treats WebArena trajectories the same as OSWorld ones.
"""

from __future__ import annotations

import hashlib
from copy import deepcopy
from typing import Any

from src.agents.common.schema import Observation, RuntimeState

from .a11y_parser import parse_axtree
from .url_mappings import map_url_to_real


class MissingPageUrlError(RuntimeError):
    """A WebArena state reached the adapter with no page, or with a page that has
    no URL, so the deciding view cannot carry the control's ``URL:`` line.

    There is no fallback: a step rendered without the line would not show the
    model what the control's view shows. The runner classifies this ``infra_error`` by type
    (``scripts/webarena/run_graphpc.py``), so the episode is excluded and
    retried. It is never scored as an ``agent_error`` 0.0."""


def _page_url(info: dict[str, Any]) -> str:
    """``info["page"].url``: the ``DetachedPage`` attribute on a live run; a
    dict-shaped page (a recorded or synthetic state) carries it as a key."""
    page = info.get("page")
    if page is None:
        raise MissingPageUrlError(f"the WebArena state's info carries no page (info keys: {sorted(info)})")
    url = page.get("url") if isinstance(page, dict) else getattr(page, "url", None)
    if not isinstance(url, str) or not url:
        raise MissingPageUrlError(f"the WebArena state's page carries no URL (page: {page!r})")
    return url


class WebArenaObservationAdapter:
    """Convert a WebArena observation dict to our ``Observation`` envelope.

    Input contract: a dict shaped like the ``StateInfo`` ``observation`` payload
    plus any per-step metadata the agent shim wants to surface — typically
    ``{text, image, observation_metadata, info, _env_feedback}``. Unknown keys
    survive on ``Observation.raw`` so they reach downstream stages without
    being lost.

    ``observation_type`` mirrors WebArena's ``ScriptBrowserEnv`` setting.
    WebArena's ``get_observation`` ALWAYS returns a visual frame (a numpy
    ndarray) alongside the text, even in ``accessibility_tree``/``html`` mode.
    The screenshot only enters ``content`` — and only then as PNG bytes, the
    downstream contract — when running in ``image`` mode; otherwise the frame
    stays on ``Observation.raw`` and never reaches the image-encoding path in
    the backbone (which would otherwise base64 a raw ndarray and get a 400).

    ``url_mappings`` is the site-URL table (``url_mappings.py``), required and
    with no default. ``build_default`` resolves it once and hands the same object
    to the action agent, which maps the other direction.
    """

    def __init__(self, observation_type: str = "accessibility_tree", *, url_mappings: dict[str, str]) -> None:
        self.observation_type = observation_type
        self.url_mappings = url_mappings

    def __call__(
        self,
        observation_input: dict[str, Any],
        runtime_state: RuntimeState,
    ) -> Observation:
        payload = deepcopy(observation_input)
        env_feedback = payload.pop("_env_feedback", {})
        # WebArena's StateInfo "info" key carries observation_metadata + extras.
        info = payload.pop("info", {}) if isinstance(payload.get("info"), dict) else {}
        observation_metadata = info.get("observation_metadata") or payload.pop(
            "observation_metadata", {}
        )
        page_url = _page_url(info)
        displayed_url = map_url_to_real(page_url, self.url_mappings)

        content: dict[str, Any] = {}
        # The a11y string lives at observation["text"] (the WebArena default
        # observation type); mirror under both keys so the rest of the stack
        # (the merge-a11y helper) finds it under "a11y_tree".
        #
        # The control's URL line follows it, in the control's template
        # (p_cot_id_actree_2s.py:69-71, "{observation}\nURL: {url}"): the alias form of
        # the page URL (CoTPromptConstructor.construct's map_url_to_real), at every
        # step, with that joiner and no strip. It is added even when the tree text is
        # empty, as on a Chrome error page, where the URL is the only thing that says
        # where the model is. Only the deciding view reads a11y_tree (promoted by the
        # orchestrator). Grounding, retrieval, completion-gate rule 1 and stored
        # entries read the projection below, which does not carry it. The monitor's
        # change diff also reads a11y_tree, so a URL-only change reads as "changed".
        text = payload.get("text")
        content["a11y_tree"] = (text or "") + "\nURL: " + displayed_url
        # Telemetry: the raw tree's digest BEFORE the URL line. Two steps with the same digest and different
        # URLs are a URL-only transition, which the monitor's diff reads as "changed".
        tree_sha256 = hashlib.sha256((text or "").encode("utf-8")).hexdigest()
        # Screenshot only for image-mode runs. WebArena always hands back a
        # visual frame, but in a11y/html mode it must NOT enter content — it
        # would reach the backbone's image encoder as a raw ndarray. When it
        # does enter (image mode), normalize to PNG bytes (the downstream
        # contract): convert a numpy ndarray via PIL; pass bytes through.
        if self.observation_type == "image" and "image" in payload:
            content["screenshot"] = self._to_png_bytes(payload["image"])
        # observation_metadata: WebArena's id ↔ a11y-node mapping, carried
        # through as cheap benign context. It is NOT used to validate emitted
        # ids (no id-validator is wired — see a11y_parser).
        if observation_metadata:
            content["observation_metadata"] = observation_metadata
        content["observation_text"] = self._build_observation_text(payload)

        return Observation(
            backend="webarena",
            content=content,
            raw=payload,
            metadata={
                "env_feedback": env_feedback,
                "observation_metadata": observation_metadata,
                "step_idx": runtime_state.step_idx,
                # Telemetry only (nothing renders Observation.metadata into a prompt):
                # the WebArena agent copies these into runtime_state.content.
                "page_url": page_url,
                "displayed_url": displayed_url,
                "tree_sha256": tree_sha256,
            },
        )

    @staticmethod
    def _to_png_bytes(image: Any) -> Any:
        """Normalize an image-mode frame to PNG bytes. WebArena hands back a
        numpy uint8 ndarray (H,W,3); encode it via PIL (lazily imported so the
        package keeps importing without Pillow). Already-``bytes`` frames pass
        through unchanged; anything else is returned as-is (defensive)."""
        if isinstance(image, bytes):
            return image
        if hasattr(image, "shape"):  # numpy ndarray — encode to PNG.
            import io

            from PIL import Image

            buffer = io.BytesIO()
            Image.fromarray(image).save(buffer, format="PNG")
            return buffer.getvalue()
        return image

    def _build_observation_text(self, payload: dict[str, Any]) -> str:
        """Project the WebArena AXTree through ``parse_axtree`` (cap 200 nodes)
        so the compact reasoning channel doesn't carry the raw multi-K-token
        tree.
        """
        text = payload.get("text")
        if not text:
            return ""
        parsed = parse_axtree(str(text), max_nodes=200)
        if not parsed["nodes"]:
            return ""
        lines = []
        for node in parsed["nodes"]:
            wa_id = node.get("webarena_id")
            role = node.get("role", "")
            name = node.get("name") or ""
            prefix = f"[{wa_id}] " if wa_id is not None else ""
            if name:
                lines.append(f"{prefix}{role} '{name}'")
            else:
                lines.append(f"{prefix}{role}")
        return f"AXTree (parsed, {len(lines)} nodes):\n" + "\n".join(lines)
