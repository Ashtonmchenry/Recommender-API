"""Utilities for A/B model routing."""


def pick_model(user_id: int, model: str | None = None) -> str:
    """Return the bucket name for ``user_id`` or honor an explicit override."""

    if model:
        return model
    return "A" if (user_id % 2 == 0) else "B"
