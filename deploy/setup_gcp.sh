#!/usr/bin/env bash
# One-time GCP setup for CI/CD. Run in Google Cloud Shell (already signed in):
#   PROJECT_ID=your-project GITHUB_REPO=ByteBoss123/mac-pricing-optimizer bash deploy/setup_gcp.sh
# Creates: Artifact Registry repo, a deploy service account, a staging bucket,
# and Workload Identity Federation so GitHub Actions can deploy without keys.
set -euo pipefail

: "${PROJECT_ID:?set PROJECT_ID}"
: "${GITHUB_REPO:?set GITHUB_REPO as owner/name}"
REGION="${REGION:-us-central1}"
AR_REPO="${AR_REPO:-pricing}"
SA_NAME="${SA_NAME:-gh-deployer}"
POOL="${POOL:-github-pool}"
PROVIDER="${PROVIDER:-github-provider}"
BUCKET="${BUCKET:-${PROJECT_ID}-pricing-models}"

gcloud config set project "$PROJECT_ID"
PROJECT_NUMBER=$(gcloud projects describe "$PROJECT_ID" --format='value(projectNumber)')
SA_EMAIL="${SA_NAME}@${PROJECT_ID}.iam.gserviceaccount.com"

gcloud services enable run.googleapis.com artifactregistry.googleapis.com \
  aiplatform.googleapis.com iamcredentials.googleapis.com sts.googleapis.com \
  cloudbuild.googleapis.com

gcloud artifacts repositories describe "$AR_REPO" --location "$REGION" >/dev/null 2>&1 || \
  gcloud artifacts repositories create "$AR_REPO" --repository-format=docker --location "$REGION"

gcloud storage buckets describe "gs://$BUCKET" >/dev/null 2>&1 || \
  gcloud storage buckets create "gs://$BUCKET" --location "$REGION" --uniform-bucket-level-access

gcloud iam service-accounts describe "$SA_EMAIL" >/dev/null 2>&1 || \
  gcloud iam service-accounts create "$SA_NAME" --display-name "GitHub Actions deployer"

for role in roles/run.admin roles/artifactregistry.writer roles/iam.serviceAccountUser \
            roles/aiplatform.user roles/storage.objectAdmin; do
  gcloud projects add-iam-policy-binding "$PROJECT_ID" --member "serviceAccount:$SA_EMAIL" \
    --role "$role" --condition=None >/dev/null
done

# lets CI mint an ID token as this SA for the authenticated Cloud Run smoke test
gcloud iam service-accounts add-iam-policy-binding "$SA_EMAIL" --member "serviceAccount:$SA_EMAIL" \
  --role roles/iam.serviceAccountTokenCreator >/dev/null

gcloud iam workload-identity-pools describe "$POOL" --location global >/dev/null 2>&1 || \
  gcloud iam workload-identity-pools create "$POOL" --location global --display-name "GitHub"

gcloud iam workload-identity-pools providers describe "$PROVIDER" --location global \
  --workload-identity-pool "$POOL" >/dev/null 2>&1 || \
  gcloud iam workload-identity-pools providers create-oidc "$PROVIDER" --location global \
    --workload-identity-pool "$POOL" --issuer-uri "https://token.actions.githubusercontent.com" \
    --attribute-mapping "google.subject=assertion.sub,attribute.repository=assertion.repository" \
    --attribute-condition "assertion.repository=='${GITHUB_REPO}'"

gcloud iam service-accounts add-iam-policy-binding "$SA_EMAIL" \
  --role roles/iam.workloadIdentityUser \
  --member "principalSet://iam.googleapis.com/projects/${PROJECT_NUMBER}/locations/global/workloadIdentityPools/${POOL}/attribute.repository/${GITHUB_REPO}" >/dev/null

cat <<EOF

Add these as GitHub repository variables (Settings > Secrets and variables > Actions > Variables):
  GCP_PROJECT_ID  = $PROJECT_ID
  GCP_REGION      = $REGION
  GCP_AR_REPO     = $AR_REPO
  GCP_BUCKET      = $BUCKET
  GCP_SA_EMAIL    = $SA_EMAIL
  GCP_WIF_PROVIDER= projects/${PROJECT_NUMBER}/locations/global/workloadIdentityPools/${POOL}/providers/${PROVIDER}
EOF
