import json
import sys
from collections import Counter, defaultdict
from datetime import datetime

# Usage: python analysis/kafka_loop_kpis.py kafka_watch_sample.jsonl

if len(sys.argv) != 2:
    print("Usage: python kafka_loop_kpis.py <watch_jsonl_file>")
    sys.exit(1)

path = sys.argv[1]

# Adjust these to match your message schema:
MOVIE_FIELD = "movie_id"   # change if your field is called "item_id" etc.
TS_FIELD = "ts"            # change if it's "timestamp", "event_time", etc.

def parse_date(raw):
    if raw is None:
        return None
    if isinstance(raw, (int, float)):
        # epoch ms
        return datetime.utcfromtimestamp(raw / 1000.0).date()
    if isinstance(raw, str):
        try:
            return datetime.fromisoformat(raw.replace("Z", "")).date()
        except Exception:
            return None
    return None

overall_counts = Counter()
per_day_counts = defaultdict(Counter)

with open(path, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue

        payload = obj.get("payload", obj)

        movie = payload.get(MOVIE_FIELD)
        ts_raw = payload.get(TS_FIELD)

        if movie is None:
            continue

        overall_counts[movie] += 1
        day = parse_date(ts_raw)
        if day is not None:
            per_day_counts[day][movie] += 1

total_watches = sum(overall_counts.values())
distinct_movies = len(overall_counts)

if total_watches == 0:
    print("No usable watch events found. Check MOVIE_FIELD/TS_FIELD.")
    sys.exit(0)

top5_n = max(1, int(distinct_movies * 0.05))
top5_watches = sum(c for _, c in overall_counts.most_common(top5_n))
top5_exposure = top5_watches / total_watches

print("===== Overall KPIs from Kafka watch logs =====")
print(f"Total watch events: {total_watches}")
print(f"Distinct movies watched: {distinct_movies}")
print(f"Top 5% movies count: {top5_n}")
print(f"Watches on top 5% movies: {top5_watches}")
print(f"Top-5% exposure: {top5_exposure:.3f} (fraction of all watches)")
print()

print("Top 10 movies by watches:")
for movie, c in overall_counts.most_common(10):
    print(f"  movie={movie}  watches={c}")

if per_day_counts:
    print("\n===== Per-day Top-5% exposure (loop check) =====")
    for day in sorted(per_day_counts):
        counts = per_day_counts[day]
        tot = sum(counts.values())
        distinct = len(counts)
        top_n = max(1, int(distinct * 0.05))
        top_w = sum(c for _, c in counts.most_common(top_n))
        exposure = top_w / tot
        print(f"{day}: exposure={exposure:.3f}  total_watches={tot}  distinct_movies={distinct}")
