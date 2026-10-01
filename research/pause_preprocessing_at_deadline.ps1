param(
    [string]$ProjectRoot = (Split-Path -Parent $PSScriptRoot),
    [datetime]$Deadline = (Get-Date).Date.AddHours(17)
)

$ErrorActionPreference = "Stop"
$logRoot = Join-Path $ProjectRoot "research_results\preprocessing_embeddings"
$stateFile = Join-Path $logRoot "active_preprocessing_workers.json"
$pauseLog = Join-Path $logRoot "scheduled_pause.log"
$protocolId = "librispeech_dev_clean_chapter_heldout_2fold__01ff6bcaaebb"
$summary = Join-Path $ProjectRoot (
    "research_results\preprocessing_benchmark\protocols\" +
    $protocolId + "\summary\logistic_regression\full\summary_metadata.json"
)

function Write-PauseLog([string]$Message) {
    $line = "{0:u} {1}" -f (Get-Date), $Message
    Add-Content -LiteralPath $pauseLog -Value $line -Encoding UTF8
}

function Get-DescendantProcessIds([int[]]$ParentIds) {
    $all = @(Get-CimInstance Win32_Process)
    $result = [System.Collections.Generic.List[int]]::new()
    $frontier = [System.Collections.Generic.List[int]]::new()
    foreach ($id in $ParentIds) {
        $frontier.Add($id)
    }
    while ($frontier.Count -gt 0) {
        $parent = $frontier[0]
        $frontier.RemoveAt(0)
        foreach ($process in $all | Where-Object ParentProcessId -eq $parent) {
            if (-not $result.Contains([int]$process.ProcessId)) {
                $result.Add([int]$process.ProcessId)
                $frontier.Add([int]$process.ProcessId)
            }
        }
    }
    return @($result)
}

if ($Deadline -le (Get-Date)) {
    throw "Deadline must be in the future: $Deadline"
}

Write-PauseLog "Scheduled preprocessing pause for $($Deadline.ToString('u'))."
while ((Get-Date) -lt $Deadline) {
    Start-Sleep -Seconds 15
}

if (Test-Path -LiteralPath $summary) {
    Write-PauseLog "Final summary already exists; no processes were stopped."
    exit 0
}

$rootIds = [System.Collections.Generic.List[int]]::new()
if (Test-Path -LiteralPath $stateFile) {
    $workers = Get-Content -LiteralPath $stateFile -Raw | ConvertFrom-Json
    foreach ($worker in $workers) {
        if (Get-Process -Id ([int]$worker.Pid) -ErrorAction SilentlyContinue) {
            $rootIds.Add([int]$worker.Pid)
        }
    }
}

$watchers = @(
    Get-CimInstance Win32_Process |
        Where-Object {
            $_.CommandLine -like "*wait_and_run_preprocessing_benchmark.ps1*" -and
            $_.CommandLine -like "* -File *"
        }
)
foreach ($watcher in $watchers) {
    if (-not $rootIds.Contains([int]$watcher.ProcessId)) {
        $rootIds.Add([int]$watcher.ProcessId)
    }
}

$descendants = Get-DescendantProcessIds -ParentIds @($rootIds)
$targets = @($descendants) + @($rootIds)
$targets = @($targets | Sort-Object -Unique -Descending)
foreach ($processId in $targets) {
    Stop-Process -Id $processId -Force -ErrorAction SilentlyContinue
}

Write-PauseLog (
    "Deadline reached before completion. Stopped process IDs: " +
    ($targets -join ",") +
    ". Resume-safe checkpoints remain on disk."
)
