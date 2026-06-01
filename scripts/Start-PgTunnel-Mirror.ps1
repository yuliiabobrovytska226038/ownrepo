#!/usr/bin/env pwsh
<#
.SYNOPSIS
    Opens a local PostgreSQL tunnel to rev-prod-test-mirror.postgres.database.azure.com
    via a temporary socat pod in OpenShift. Blocks until Ctrl+C — pod is automatically
    deleted on exit.

.DESCRIPTION
    1. Verifies oc is authenticated and switches to the target namespace.
    2. Creates an ephemeral alpine/socat pod that forwards TCP :5432 to Azure PG.
    3. Runs oc port-forward in the foreground (this terminal is "in use" while open).
    4. On exit — whether via Ctrl+C, terminal close, or error — the pod is deleted
       automatically via a try/finally block. No separate stop script is needed.
    5. The pod has an activeDeadlineSeconds TTL (default: 8 h) enforced by Kubernetes.
       If the terminal is force-killed and finally cannot run, the pod self-destructs
       after the deadline regardless.

.PARAMETER LocalPort
    The local port to bind on your machine. Default: 5434.

.PARAMETER Namespace
    The OpenShift namespace. Default: rev-ha-tms-prd-test-mirror.

.PARAMETER PgHost
    The Azure PostgreSQL FQDN. Default: rev-prod-test-mirror.postgres.database.azure.com.

.PARAMETER PgPort
    The remote PostgreSQL port. Default: 5432.

.PARAMETER TtlHours
    Maximum lifetime of the socat pod in hours, enforced by Kubernetes.
    Acts as a safety net if the terminal is force-killed. Default: 8.

.PARAMETER SecretName
    Name of the OpenShift secret containing database connection metadata.
    Default: tms-db-secret.

.EXAMPLE
    .\Start-PgTunnel-Mirror.ps1
    .\Start-PgTunnel-Mirror.ps1 -LocalPort 5432
    .\Start-PgTunnel-Mirror.ps1 -TtlHours 4
#>
param(
    [int]$LocalPort     = 5434,
    [string]$Namespace  = "rev-ha-tms-prd-test-mirror",
    [string]$PgHost     = "rev-prod-test-mirror.postgres.database.azure.com",
    [int]$PgPort        = 5432,
    [int]$TtlHours      = 8,
    [string]$SecretName = "tms-db-secret"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

# ── 1. Verify oc is available ────────────────────────────────────────────────
if (-not (Get-Command oc -ErrorAction SilentlyContinue)) {
    Write-Error "oc CLI not found. Install the OpenShift CLI and try again."
    exit 1
}

# ── 2. Verify authentication ─────────────────────────────────────────────────
$currentUser = oc whoami 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Error "Not logged in to OpenShift. Run 'oc login' first."
    exit 1
}
Write-Host "Authenticated as: $currentUser" -ForegroundColor Green

# ── 3. Switch to the correct namespace ───────────────────────────────────────
Write-Host "Switching to namespace: $Namespace" -ForegroundColor Cyan
oc project $Namespace 2>&1 | Out-Null
if ($LASTEXITCODE -ne 0) {
    Write-Error "Failed to switch to namespace '$Namespace'."
    exit 1
}

# ── 4. Remove any leftover pg-tunnel pod ─────────────────────────────────────
oc get pod pg-tunnel --no-headers 2>&1 | Out-Null
if ($LASTEXITCODE -eq 0) {
    Write-Host "Removing leftover pg-tunnel pod..." -ForegroundColor Yellow
    oc delete pod pg-tunnel --wait=true 2>&1 | Out-Null
}

# ── 5. Create the socat tunnel pod ───────────────────────────────────────────
#
#  activeDeadlineSeconds is a Kubernetes-native safety net: if the terminal is
#  force-killed (SIGKILL, power loss, VS Code crash) and the try/finally block
#  cannot run, the cluster itself will terminate the pod after this deadline.
#  The pod will NOT forward traffic after termination — the port-forward will
#  simply drop. Next script run detects and removes the stale pod automatically.
#
$ttlSeconds = $TtlHours * 3600
$overrides  = '{"spec":{"activeDeadlineSeconds":' + $ttlSeconds + '}}'
Write-Host "Creating socat tunnel pod: pg-tunnel (TTL: ${TtlHours}h)" -ForegroundColor Cyan
oc run pg-tunnel --restart=Never --image=alpine/socat --overrides=$overrides `
    -- TCP-LISTEN:5432,fork,reuseaddr "TCP:${PgHost}:${PgPort}" 2>&1 | Out-Null
if ($LASTEXITCODE -ne 0) {
    Write-Error "Failed to create pg-tunnel pod."
    exit 1
}

# ── 6. Wait for the pod to be ready ──────────────────────────────────────────
Write-Host "Waiting for pg-tunnel pod to be ready..." -ForegroundColor Cyan
oc wait pod/pg-tunnel --for=condition=Ready --timeout=60s
if ($LASTEXITCODE -ne 0) {
    oc delete pod pg-tunnel --ignore-not-found 2>&1 | Out-Null
    Write-Error "pg-tunnel pod did not become ready within 60s. Check: oc describe pod pg-tunnel"
    exit 1
}
Write-Host "Pod is ready." -ForegroundColor Green

# ── 7. Check if local port is already in use ─────────────────────────────────
$portInUse = Test-NetConnection -ComputerName localhost -Port $LocalPort `
    -WarningAction SilentlyContinue -ErrorAction SilentlyContinue
if ($portInUse.TcpTestSucceeded) {
    oc delete pod pg-tunnel --ignore-not-found 2>&1 | Out-Null
    Write-Error "Port $LocalPort is already in use on localhost. Use -LocalPort to pick a different port."
    exit 1
}

# ── 8. Fetch connection metadata from secret ─────────────────────────────────
#  tms-db-secret uses field names: database-name, database-application-user
$creds  = oc get secret $SecretName -o jsonpath='{.data}' 2>$null | ConvertFrom-Json
$pgDb   = if ($creds -and $creds.'database-name')             { [System.Text.Encoding]::UTF8.GetString([System.Convert]::FromBase64String($creds.'database-name')) }             else { "<database>" }
$pgUser = if ($creds -and $creds.'database-application-user') { [System.Text.Encoding]::UTF8.GetString([System.Convert]::FromBase64String($creds.'database-application-user')) } else { "<user>" }

# ── 9. Print connection details ───────────────────────────────────────────────
Write-Host ""
Write-Host "════════════════════════════════════════════════════" -ForegroundColor Green
Write-Host " pg-tunnel is up — press Ctrl+C to stop and clean up" -ForegroundColor Green
Write-Host "════════════════════════════════════════════════════" -ForegroundColor Green
Write-Host " Host     : localhost"
Write-Host " Port     : $LocalPort"
Write-Host " Database : $pgDb"
Write-Host " User     : $pgUser"
Write-Host " SSL      : sslmode=require"
Write-Host " Pod TTL  : ${TtlHours}h (auto-destroyed by cluster if terminal is force-killed)"
Write-Host ""
Write-Host " URI:  postgresql://${pgUser}:<password>@localhost:${LocalPort}/${pgDb}?sslmode=require"
Write-Host "════════════════════════════════════════════════════" -ForegroundColor Green
Write-Host ""

# ── 10. Run port-forward in foreground; delete pod on any exit ────────────────
#
#  try/finally guarantees the pod is deleted whether the user hits Ctrl+C,
#  port-forward dies on its own, or an error is thrown. The terminal is "in use"
#  for the duration — no background jobs, no dangling resources.
#
try {
    oc port-forward pod/pg-tunnel "${LocalPort}:5432"
} finally {
    Write-Host ""
    Write-Host "Cleaning up pg-tunnel pod..." -ForegroundColor Yellow
    oc delete pod pg-tunnel --ignore-not-found 2>&1 | Out-Null
    Write-Host "Done. Tunnel closed and pod deleted." -ForegroundColor Green
}
