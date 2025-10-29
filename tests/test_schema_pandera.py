import pandas as pd
from recommender.schemas.events import EventSchema
def test_event_schema_validates():
    df = pd.DataFrame([{
        'request_id': 'r1', 'user_id': 1, 'movie_id': 2, 'timestamp': 1700000000, 'rating': 4.0
    }])
    validated = EventSchema.validate(df)
    assert 'user_id' in validated.columns
