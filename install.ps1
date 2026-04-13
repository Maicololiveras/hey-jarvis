<#
.SYNOPSIS
    Register/unregister Hey Jarvis as a Windows Task Scheduler task.
.PARAMETER Uninstall
    Remove the scheduled task instead of creating it.
#>
param(
    [switch]$Uninstall
)

$ErrorActionPreference = "Stop"
$TASK_NAME = "HeyJarvis"
$JARVIS_DIR = Split-Path -Parent $MyInvocation.MyCommand.Path
$LAUNCHER = Join-Path $JARVIS_DIR "launch_jarvis.ps1"

if ($Uninstall) {
    Write-Host "[Jarvis] Removing scheduled task '$TASK_NAME'..."
    Unregister-ScheduledTask -TaskName $TASK_NAME -Confirm:$false -ErrorAction SilentlyContinue
    Write-Host "[Jarvis] Task removed."
    exit 0
}

# Create task
Write-Host "[Jarvis] Registering scheduled task '$TASK_NAME'..."

$action = New-ScheduledTaskAction `
    -Execute "powershell.exe" `
    -Argument "-ExecutionPolicy Bypass -WindowStyle Hidden -File `"$LAUNCHER`"" `
    -WorkingDirectory $JARVIS_DIR

$trigger = New-ScheduledTaskTrigger -AtLogOn

$settings = New-ScheduledTaskSettingsSet `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -RestartInterval (New-TimeSpan -Seconds 30) `
    -RestartCount 999 `
    -ExecutionTimeLimit (New-TimeSpan -Days 365)

$principal = New-ScheduledTaskPrincipal -UserId $env:USERNAME -RunLevel Highest -LogonType Interactive

Register-ScheduledTask `
    -TaskName $TASK_NAME `
    -Action $action `
    -Trigger $trigger `
    -Settings $settings `
    -Principal $principal `
    -Description "Hey Jarvis — Voice-first AI assistant daemon" `
    -Force

Write-Host "[Jarvis] Task '$TASK_NAME' registered. Jarvis will auto-start on next logon."
Write-Host "[Jarvis] To start now: .\launch_jarvis.ps1"
Write-Host "[Jarvis] To uninstall: .\install.ps1 -Uninstall"
