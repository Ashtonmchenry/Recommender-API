"""generate predictions"""

# recommender/serve.py
from __future__ import annotations
from typing import List
import numpy as np
from scipy import sparse
from .serialize import load_model

# Build a minimal UI row on the fly if needed (cold-start => empty row)
def recommend_for_user(model_path: str, user_id: int, k: int = 20) -> List[int]:
    model = load_model(model_path)  # ALSModel
    uidx = model.user_map.get(user_id, None)
    # Build a 1 x n_items user row (empty -> cold start)
    n_items = model.item_factors.shape[0]
    if uidx is None:
        # cold-start: return top items by global popularity proxy (norm of item factors)
        scores = np.linalg.norm(model.item_factors, axis=1)
        top = np.argpartition(scores, -k)[-k:]
        iidx = top[np.argsort(-scores[top])]
        # map back to external ids
        inv_item = {v:k for k,v in model.item_map.items()}
        return [inv_item[i] for i in iidx]
    else:
        # Build a one-row UI with zeros (already-seen filtering is done inside recommend via ALS if you pass UI)
        # For serving we skip seen filtering unless you cache UI; this is a simple variant:
        # Return best items by dot(user_factors[uidx], item_factors.T)
        s = model.user_factors[uidx] @ model.item_factors.T
        top = np.argpartition(s, -k)[-k:]
        iidx = top[np.argsort(-s[top])]
        inv_item = {v:k for k,v in model.item_map.items()}
        return [inv_item[i] for i in iidx]

