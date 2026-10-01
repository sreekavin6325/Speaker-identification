param(
    [string]$ProjectRoot = (Split-Path -Parent $PSScriptRoot),
    [int]$RefreshSeconds = 5
)

$protocolId = "librispeech_dev_clean_chapter_heldout_2fold__01ff6bcaaebb"
$root = Join-Path $ProjectRoot (
    "research_results\preprocessing_embeddings\" + $protocolId
)
$logRoot = Join-Path $ProjectRoot "research_results\preprocessing_embeddings"
$workersFile = Join-Path $logRoot "active_preprocessing_workers.json"
$benchmarkLog = Join-Path $ProjectRoot (
    "research_results\preprocessing_benchmark\automatic_benchmark_stdout.log"
)

while ($true) {
    Clear-Host
    Write-Host "SPEAKER RECOGNITION - PREPROCESSING ABLATION" -ForegroundColor Cyan
    Write-Host ("Updated: " + (Get-Date -Format "dd-MM-yyyy HH:mm:ss"))
    Write-Host ("=" * 78)

    if (Test-Path -LiteralPath $workersFile) {
        $workers = Get-Content $workersFile -Raw | ConvertFrom-Json
        foreach ($worker in $workers) {
            $process = Get-Process -Id $worker.Pid -ErrorAction SilentlyContinue
            $status = if ($null -ne $process) { "RUNNING" } else { "STOPPED" }
            $colour = if ($null -ne $process) { "Green" } else { "Red" }
            Write-Host (
                "{0,-10} {1,-8} {2}" -f $worker.Name, $status, $worker.Models
            ) -ForegroundColor $colour
        }
    }

    $completed = @(
        Get-ChildItem -LiteralPath $root -Recurse -Filter "metadata.json" `
            -ErrorAction SilentlyContinue |
        Where-Object { $_.FullName -like "*\full\*" }
    ).Count
    $percentage = [math]::Round(100.0 * $completed / 28, 1)
    Write-Host ""
    Write-Host "Completed model/variants: $completed / 28 ($percentage%)" `
        -ForegroundColor Yellow

    Write-Host ""
    Write-Host "ACTIVE CHECKPOINTS" -ForegroundColor Cyan
    Write-Host ("-" * 78)
    $states = @(
        Get-ChildItem -LiteralPath $root -Recurse `
            -Filter "checkpoint_state.json" -ErrorAction SilentlyContinue |
        Where-Object { $_.FullName -like "*\full\*" } |
        ForEach-Object {
            $metadata = Join-Path $_.Directory "metadata.json"
            if (-not (Test-Path -LiteralPath $metadata)) {
                $state = Get-Content $_.FullName -Raw | ConvertFrom-Json
                [pscustomobject]@{
                    Model = $state.model_id
                    Variant = $state.variant
                    Processed = [int]$state.processed_rows
                    Success = [int]$state.successful_rows
                    Failed = [int]$state.processed_rows - [int]$state.successful_rows
                    Remaining = 2185 - [int]$state.processed_rows
                    Percent = [math]::Round(
                        100.0 * [int]$state.processed_rows / 2185,
                        1
                    )
                }
            }
        }
    )
    if ($states.Count -gt 0) {
        $states | Format-Table -AutoSize
    }
    else {
        Write-Host "No active checkpoint."
    }

    Write-Host "LATEST LOG STATUS" -ForegroundColor Cyan
    Write-Host ("-" * 78)
    $workerNames = @("worker_a", "worker_b", "resume_worker")
    if (Test-Path -LiteralPath $workersFile) {
        $workerNames = @(
            Get-Content $workersFile -Raw |
                ConvertFrom-Json |
                ForEach-Object { $_.Name }
        )
    }
    foreach ($name in $workerNames) {
        $log = Join-Path $logRoot ($name + "_stdout.log")
        Write-Host "[$name]" -ForegroundColor DarkCyan
        if (Test-Path -LiteralPath $log) {
            Get-Content $log -Tail 1
        }
    }
    Write-Host "[benchmark watcher]" -ForegroundColor DarkCyan
    if (Test-Path -LiteralPath $benchmarkLog) {
        Get-Content $benchmarkLog -Tail 1
    }
    Write-Host ""
    Write-Host "Closing this monitor does not stop the experiment." `
        -ForegroundColor DarkGray
    Start-Sleep -Seconds $RefreshSeconds
}
