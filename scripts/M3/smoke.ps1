# hits /health, /recommend, and /metrics to verify the deployed service
# smoke.ps1

param(
  [string]$Region  = $env:GCP_REGION,
  [string]$Service = $env:CLOUD_RUN_SERVICE
)

if (-not $Region)  { throw "Set -Region or GCP_REGION." }
if (-not $Service) { $Service = "reco-api" }

$URL = gcloud run services describe $Service --region $Region --format "value(status.url)"
if (-not $URL) { throw "Could not resolve Cloud Run URL. Is the service deployed?" }
"URL: $URL"

"--- HEALTH ---"
(Invoke-WebRequest "$URL/health").Content

"--- RECOMMEND ---"
(Invoke-WebRequest "$URL/recommend?user_id=1&k=5").Content

"--- METRICS (first 20 lines) ---"
$metrics = (Invoke-WebRequest "$URL/metrics").Content -split "`n"
$metrics | Select-Object -First 20

