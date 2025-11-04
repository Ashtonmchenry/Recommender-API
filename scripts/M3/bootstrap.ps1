# one-time Google Cloud Project setup per project/region: enable APIs, create Artifact Registry repo if missing.

param(
  [string]$Project = $env:GCP_PROJECT_ID,
  [string]$Region  = $env:GCP_REGION,
  [string]$Repo    = "reco-repo"
)

if (-not $Project) { throw "Set -Project or environment variable GCP_PROJECT_ID." }
if (-not $Region)  { throw "Set -Region or environment variable GCP_REGION." }

gcloud config set project $Project | Out-Null

Write-Host "Enabling required APIs..."
gcloud services enable `
  run.googleapis.com `
  artifactregistry.googleapis.com `
  cloudbuild.googleapis.com `
  iam.googleapis.com `
  iamcredentials.googleapis.com | Out-Null

Write-Host "Ensuring Artifact Registry repo '$Repo' exists in $Region..."
gcloud artifacts repositories create $Repo `
  --repository-format=docker `
  --location=$Region `
  --description="Recommender API images" `
  2>$null
Write-Host "Bootstrap complete."
