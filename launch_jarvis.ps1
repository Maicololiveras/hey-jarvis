<#
.SYNOPSIS
    Launches Hey Jarvis daemon with singleton enforcement.
.DESCRIPTION
    Checks if Jarvis is already running. If not, launches it.
    Supports crash recovery with restart limits.
#>
param(
    [switch]$Force  # Kill existing instance and restart
)

$ErrorActionPreference = "Stop"
$JARVIS_DIR = Split-Path -Parent $MyInvocation.MyCommand.Path
$LOG_FILE = Join-Path $JARVIS_DIR "jarvis.log"
$RESTART_FILE = Join-Path $env:TEMP "jarvis_restarts.txt"
$MAX_RESTARTS = 3
$RESTART_WINDOW_SECONDS = 300  # 5 minutes

function Get-JarvisProcess {
    Get-WmiObject Win32_Process -Filter "Name='python.exe'" 2>$null |
        Where-Object { $_.CommandLine -match "jarvis" }
}

# Singleton check
$existing = Get-JarvisProcess
if ($existing -and -not $Force) {
    Write-Host "[Jarvis] Already running (PID: $($existing.ProcessId))"
    exit 0
}

if ($existing -and $Force) {
    Write-Host "[Jarvis] Killing existing instance (PID: $($existing.ProcessId))"
    Stop-Process -Id $existing.ProcessId -Force
    Start-Sleep -Seconds 2
}

# Crash recovery tracking (TASK-024)
function Test-RestartLimit {
    if (-not (Test-Path $RESTART_FILE)) { return $false }
    $lines = Get-Content $RESTART_FILE
    $now = [DateTimeOffset]::UtcNow.ToUnixTimeSeconds()
    $recent = $lines | Where-Object { ($now - [long]$_) -lt $RESTART_WINDOW_SECONDS }
    return ($recent.Count -ge $MAX_RESTARTS)
}

function Add-RestartEntry {
    $now = [DateTimeOffset]::UtcNow.ToUnixTimeSeconds()
    Add-Content -Path $RESTART_FILE -Value $now
    # Prune old entries
    $lines = Get-Content $RESTART_FILE
    $cutoff = $now - $RESTART_WINDOW_SECONDS
    $lines | Where-Object { ([long]$_) -ge $cutoff } | Set-Content $RESTART_FILE
}

# Launch loop with crash recovery
while ($true) {
    if (Test-RestartLimit) {
        Write-Host "[Jarvis] Max restart attempts ($MAX_RESTARTS in ${RESTART_WINDOW_SECONDS}s) exceeded. Giving up."
        exit 1
    }

    Write-Host "[Jarvis] Launching daemon..."
    $proc = Start-Process -FilePath "python" -ArgumentList "-u", "-m", "jarvis" `
        -WorkingDirectory "C:\SDK" `
        -WindowStyle Hidden `
        -RedirectStandardOutput $LOG_FILE `
        -RedirectStandardError $LOG_FILE `
        -PassThru

    Write-Host "[Jarvis] Daemon started (PID: $($proc.Id))"
    $proc.WaitForExit()

    $exitCode = $proc.ExitCode
    Write-Host "[Jarvis] Daemon exited with code $exitCode"

    if ($exitCode -eq 0) {
        Write-Host "[Jarvis] Clean exit. Not restarting."
        break
    }

    # Non-zero exit — crash recovery
    Add-RestartEntry
    Write-Host "[Jarvis] Crash detected. Restarting in 10 seconds..."
    Start-Sleep -Seconds 10
}
