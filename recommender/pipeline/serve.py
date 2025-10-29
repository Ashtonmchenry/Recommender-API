"""generate predictions"""

from typing import List, Dict, Any

def score(model, feats: List[Dict[str, Any]]) -> List[float]:
    import pandas as pd
    
    X = pd.get_dummies(pd.DataFrame(feats), drop_first=True)

    # align columns
    import numpy as np
    for col in set(model.feature_names_in_) - set(X.columns):
        X[col] = 0
    X = X[model.feature_names_in_]
    return model.predict_proba(X)[:,1].tolist()
