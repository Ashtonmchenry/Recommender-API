# deploys that image to Cloud Run and prints the service URL
# deploy.ps1
param(
  [string]$Project   = $env:GCP_PROJECT_ID,
  [string]$Region    = $env:GCP_REGION,
  [string]$Service   = $env:CLOUD_RUN_SERVICE
)

if (-not $Project) { throw "Set -Project or GCP_PROJECT_ID." }
if (-not $Region)  { throw "Set -Region or GCP_REGION." }
if (-not $Service) { $Service = "reco-api" }

$RepoRoot = (Resolve-Path "$PSScriptRoot\..\..").Path
$ImageFile = Join-Path $RepoRoot '.image_uri'
if (-not (Test-Path $ImageFile)) { throw "Missing $ImageFile. Run build.ps1 first." }
$ImageUri = Get-Content $ImageFile -Raw

Write-Host "Deploying $Service with $ImageUri in $Region..."
gcloud run deploy $Service `
  --image   $ImageUri `
  --region  $Region `
  --platform managed `
  --allow-unauthenticated

$Url = gcloud run services describe $Service --region $Region --format "value(status.url)"
Write-Host "Service URL: $Url"
Set-Content -Path (Join-Path $RepoRoot '.service_url') -Value $Url -Encoding ASCII


