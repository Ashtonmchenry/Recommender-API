import os, time, requests, random

API = os.environ.get('RECO_API', 'http://localhost:8080')
def main():
    user = random.randint(1, 1000)
    try:
        r = requests.get(f"{API}/recommend/{user}", timeout=5)
        print("status:", r.status_code, "body:", r.text[:100])
    except Exception as e:
        print("probe error:", e)
if __name__ == '__main__':
    main()
