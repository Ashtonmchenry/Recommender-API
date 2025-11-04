# builds service image with Cloud Build using cloudbuild.yaml and tags/pushes to Artifact Registry
# build.ps1
param(
  [string]$Project    = $env:GCP_PROJECT_ID,
  [string]$Region     = $env:GCP_REGION,
  [string]$Repo       = "reco-repo",
  [string]$ImageName  = "reco-api",
  [string]$Tag        = "manual"
)

if (-not $Project) { throw "Set -Project or environment variable GCP_PROJECT_ID." }
if (-not $Region)  { throw "Set -Region or environment variable GCP_REGION." }

# Repo root
$RepoRoot = (Resolve-Path "$PSScriptRoot\..\..").Path
$Cloudbuild = Join-Path $RepoRoot 'cloudbuild.yaml'

$HostName  = "$Region-docker.pkg.dev"
$ImageUri  = "{0}/{1}/{2}/{3}:{4}" -f $HostName,$Project,$Repo,$ImageName,$Tag
Write-Host "IMAGE_URI = $ImageUri"

gcloud builds submit `
  --config "$Cloudbuild" `
  --substitutions "_IMAGE_URI=$ImageUri" `
  "$RepoRoot"

# persist for deploy step
Set-Content -Path (Join-Path $RepoRoot '.image_uri') -Value $ImageUri -Encoding ASCII
Write-Host "Wrote $(Join-Path $RepoRoot '.image_uri')"


