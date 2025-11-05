import json, pandas as pd, pathlib
out_dir = pathlib.Path("out_eval")
j = json.load(open(out_dir/"metrics.json"))

ov = []
for k in j["k_values"]:
    ov.append({
        "k": k,
        "precision": j["overall"]["precision"][str(k)],
        "recall":    j["overall"]["recall"][str(k)],
        "ndcg":      j["overall"]["ndcg"][str(k)],
        "map":       j["overall"]["map"][str(k)],
        "users_evaluated": j["users_evaluated"],
    })
pd.DataFrame(ov).to_csv(out_dir/"overall_metrics.csv", index=False)

rows = []
for name, sect in j["subpopulations"].items():
    for k in j["k_values"]:
        rows.append({
            "subpopulation": name,
            "k": k,
            "precision": sect["precision"][str(k)],
            "recall":    sect["recall"][str(k)],
            "ndcg":      sect["ndcg"][str(k)],
            "map":       sect["map"][str(k)],
        })
pd.DataFrame(rows).to_csv(out_dir/"subpopulation_summary.csv", index=False)
print("Wrote overall_metrics.csv and subpopulation_summary.csv")
