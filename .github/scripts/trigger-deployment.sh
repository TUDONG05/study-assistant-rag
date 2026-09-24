#!/usr/bin/env bash

set -euo pipefail

: "${DEPLOYMENT_TARGET:?DEPLOYMENT_TARGET must be set}"
: "${DEPLOYMENT_SHA:?DEPLOYMENT_SHA must be set}"
: "${DEPLOY_HOOK_URL:?Deployment hook URL must be configured}"
: "${DEPLOY_HEALTHCHECK_URL:?Health-check URL must be configured}"

# Deployment hooks must treat this SHA as their idempotency key and deploy that revision.
curl --fail --silent --show-error --request POST \
  --connect-timeout 10 --max-time 60 \
  --header "X-Deployment-SHA: ${DEPLOYMENT_SHA}" \
  "${DEPLOY_HOOK_URL}"

# The health endpoint must respond only after this revision is serving and echo the SHA.
for attempt in 1 2 3 4 5 6; do
  response_headers="$(mktemp)"
  if curl --fail --silent --show-error --location \
    --connect-timeout 10 --max-time 30 --dump-header "${response_headers}" --output /dev/null \
    "${DEPLOY_HEALTHCHECK_URL}" \
    && grep --ignore-case --quiet "^X-Deployment-SHA: ${DEPLOYMENT_SHA}" "${response_headers}"; then
    rm -f "${response_headers}"
    echo "${DEPLOYMENT_TARGET} deployment ${DEPLOYMENT_SHA} passed its health check."
    exit 0
  fi
  rm -f "${response_headers}"
  sleep 10
done

echo "${DEPLOYMENT_TARGET} deployment did not become healthy at ${DEPLOYMENT_SHA}." >&2
exit 1
