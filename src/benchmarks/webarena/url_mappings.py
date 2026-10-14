"""WebArena's site-URL table and the control's two URL mappers, copied from upstream.

The bare control (WebArena's own ``run.py --agent_type prompt``) shows the model
each page's URL in its alias form (``http://gitlab.com/...``, never the real
host) and maps whatever alias the model writes back to the real host before
the action reaches the env. The GraphPC arm does the same through these three
functions, so both arms use one table and one pair of substring rules.

PROVENANCE: the WebArena checkout at dce04686a56253aefba7b18a4fa0937cf1dc987b,
copied VERBATIM (copy first, adapt minimally):

  * ``url_mappings_from_env``: ``browser_env/env_config.py:4-29`` and ``:43-51``.
    That is the seven ``os.environ.get`` calls, the non-empty assert and the
    ``URL_MAPPINGS`` dict, in upstream's order, with upstream's names. The one
    adaptation: upstream builds the table when the module is imported, and here
    it is a function. ``build_default`` calls it once, when it builds the agent,
    so a missing site variable aborts before the first episode. It is never
    read lazily per step.
  * ``map_url_to_real`` / ``map_url_to_local``: ``PromptConstructor``'s methods
    of those names, ``agent/prompts/prompt_constructor.py:122-137``. The one
    adaptation: ``self`` and the module-level ``URL_MAPPINGS`` become the
    ``mappings`` argument.

Upstream's quirks are REPLICATED, not improved, because the control has them:
  * The rules are substring replaces, so ``www.``-prefixed hosts and
    ``http://luma.com/`` without ``/admin`` are never mapped.
  * ``chrome-error://`` and ``about:blank`` pass through unchanged.
  * Only Kiwix's landing page maps, in either direction.
  * ``map_url_to_local`` runs over the WHOLE action string, as upstream's
    ``extract_action`` does, so a ``type`` text or a ``stop`` answer is mapped too.

The live launch preflight (``scripts/webarena/launch_preflight.py``) asserts
that these equal upstream's objects in the WebArena interpreter: the table
equals ``browser_env.env_config.URL_MAPPINGS`` in values and order, and the
mappers equal ``PromptConstructor``'s on a fixture set.
"""

from __future__ import annotations

import os


def url_mappings_from_env() -> dict[str, str]:
    """``{local site URL: alias URL}`` from the seven site variables, as
    ``browser_env.env_config`` builds ``URL_MAPPINGS``. Upstream's assert is kept
    verbatim; ``python -O`` strips it, so ``build_default`` checks the table again."""
    REDDIT = os.environ.get("REDDIT", "")
    SHOPPING = os.environ.get("SHOPPING", "")
    SHOPPING_ADMIN = os.environ.get("SHOPPING_ADMIN", "")
    GITLAB = os.environ.get("GITLAB", "")
    WIKIPEDIA = os.environ.get("WIKIPEDIA", "")
    MAP = os.environ.get("MAP", "")
    HOMEPAGE = os.environ.get("HOMEPAGE", "")

    assert (
        REDDIT
        and SHOPPING
        and SHOPPING_ADMIN
        and GITLAB
        and WIKIPEDIA
        and MAP
        and HOMEPAGE
    ), (
        f"Please setup the URLs to each site. Current: \n"
        + f"Reddit: {REDDIT}\n"
        + f"Shopping: {SHOPPING}\n"
        + f"Shopping Admin: {SHOPPING_ADMIN}\n"
        + f"Gitlab: {GITLAB}\n"
        + f"Wikipedia: {WIKIPEDIA}\n"
        + f"Map: {MAP}\n"
        + f"Homepage: {HOMEPAGE}\n"
    )

    URL_MAPPINGS = {
        REDDIT: "http://reddit.com",
        SHOPPING: "http://onestopmarket.com",
        SHOPPING_ADMIN: "http://luma.com/admin",
        GITLAB: "http://gitlab.com",
        WIKIPEDIA: "http://wikipedia.org",
        MAP: "http://openstreetmap.org",
        HOMEPAGE: "http://homepage.com",
    }
    return URL_MAPPINGS


def map_url_to_real(url: str, mappings: dict[str, str]) -> str:
    """Map the urls to their real world counterparts"""
    for i, j in mappings.items():
        if i in url:
            url = url.replace(i, j)
    return url


def map_url_to_local(url: str, mappings: dict[str, str]) -> str:
    """Map the urls to their local counterparts"""
    for i, j in mappings.items():
        if j in url:
            url = url.replace(j, i)
        # https
        if j.replace("http", "https") in url:
            url = url.replace(j.replace("http", "https"), i)
    return url
