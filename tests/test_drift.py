import pandas as pd
from quality.drift import detect_drift, any_drift

def test_drift_report_runs():
    ref = pd.DataFrame({'user_id':[1,2,3,4,5], 'movie_id':[10,10,11,12,12]})
    cur = pd.DataFrame({'user_id':[100,200,300,400,500], 'movie_id':[10,10,11,12,12]})
    report = detect_drift(ref, cur)
    assert 'user_id' in report
    assert isinstance(any_drift(report), bool)
