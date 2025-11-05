import argparse, csv, os
import pandas as pd

def sniff_sep(path):
    with open(path, "rb") as f:
        sample = f.read(4096)
    txt = sample.decode("latin1", errors="ignore")
    try:
        det = csv.Sniffer().sniff(txt, delimiters=[",",";","\t","|",":"])
        sep = det.delimiter
        if sep == ":" and "::" in txt:  # MovieLens "ratings.dat"
            return "::"
        return sep
    except Exception:
        return ","  # reasonable default

def load_and_normalize(ratings_path):
    sep = sniff_sep(ratings_path)
    df = pd.read_csv(ratings_path, sep=sep, engine="python", header=None)

    # Try to map columns to user_id,item_id,rating[,ts]
    cols = [c.lower() for c in df.columns.astype(str)]
    if len(df.columns) >= 4 and set(cols) == set(["0","1","2","3"]):
        df.columns = ["user_id","item_id","rating","ts"]
    elif {"userId","movieId","rating","timestamp"}.issubset(set(df.columns)):
        df = df.rename(columns={"userId":"user_id","movieId":"item_id","timestamp":"ts"})
    elif {"user_id","item_id","rating"}.issubset(set(df.columns)):
        pass
    else:
        new_cols = ["user_id","item_id","rating"] + (["ts"] if df.shape[1] >= 4 else [])
        df.columns = new_cols + [f"c{i}" for i in range(df.shape[1]-len(new_cols))]

    for c in ["user_id","item_id"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
    df = df.dropna(subset=["user_id","item_id","rating"]).astype({"user_id":"int64","item_id":"int64"})
    return df

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ratings", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--min-interactions", type=int, default=5)
    args = ap.parse_args()

    df = load_and_normalize(args.ratings)

    # filter by min interactions per user
    keep_users = df.groupby("user_id")["item_id"].count()
    keep_users = keep_users[keep_users >= args.min_interactions].index
    df = df[df["user_id"].isin(keep_users)]

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    cols = ["user_id","item_id","rating"] + (["ts"] if "ts" in df.columns else [])
    df[cols].to_csv(args.out, index=False)
    print(f"Wrote {len(df):,} rows to {args.out}")

if __name__ == "__main__":
    main()
