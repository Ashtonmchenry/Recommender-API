# Cloud-Native Recommender (Starter)

## Quickstart
1. **Python setup**
   ```bash
   python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```
2. **Environment**
   - Copy `.env.example` → `.env` and set values.
   - Copy `infra/kafka.env.example` → `infra/kafka.env` and set Kafka credentials.
3. **Run API locally**
   ```bash
   uvicorn service.app:app --reload --port 8080
   curl http://localhost:8080/healthz
   ```
4. **Docker build**
   ```bash
   docker build -f docker/recommender.Dockerfile -t reco-api:dev .
   docker run -p 8080:8080 --env-file .env reco-api:dev
   ```
5. **Kafka sanity (kcat)**
   ```bash
   # example (adjust flags for your cluster)
   kcat -b $BOOTSTRAP -X security.protocol=SASL_SSL -X sasl.mechanisms=PLAIN         -X sasl.username=$API_KEY -X sasl.password=$API_SECRET         -t $TEAM.watch -C -o -5 -q
   ```
6. **CI/CD** (GitHub Actions)
   - Push to main → CI runs tests, builds/pushes image, CD deploys to cloud.

## Endpoints
- `GET /healthz` → `{ "status": "ok", "version": "vX.Y" }`
- `GET /recommend/{user_id}?k=20&model=A` → comma-separated movie IDs
- `GET /metrics` → Prometheus exposition

## Project Structure
See repo tree; update components as you progress through milestones.

## Secrets & Safety
- Use **GitHub Environments/Secrets**. Never commit secrets. Rotate if leaked.
- Follow your team contract’s Definition of Done. Lint and tests must pass before merge.
