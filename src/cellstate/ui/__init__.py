"""A local, read-only web surface over the committed GSE274113 slice.

Importing this package requires the ``ui`` optional extra (``uv sync --extra ui``), which brings
FastAPI and uvicorn.  The core install stays numpy / pydantic / scipy, because a research package
should not carry a web server for everyone who wants an estimator.
"""

from __future__ import annotations

__all__ = ["app", "main"]


def __getattr__(name: str) -> object:
    """Defer the FastAPI import until something is actually used.

    ``import cellstate.ui`` must not explode for an install without the extra, and it must not
    quietly pretend to work either.  The failure arrives when the app is asked for, and it says
    which command fixes it.
    """

    if name in __all__:
        try:
            from . import server
        except ModuleNotFoundError as error:  # pragma: no cover - exercised by the extra's absence
            raise ModuleNotFoundError(
                "the web UI needs the 'ui' extra; install it with `uv sync --extra ui` "
                f"(missing: {error.name})"
            ) from error
        return getattr(server, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
