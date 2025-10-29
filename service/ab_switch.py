# Simple A/B switch based on user_id parity
def pick_model(user_id: int, model: str | None = None) -> str:
    if model:
        return model
    return "A" if (user_id % 2 == 0) else "B"