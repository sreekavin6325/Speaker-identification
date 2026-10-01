param(
    [string]$ProjectRoot = (Split-Path -Parent $PSScriptRoot),
    [int]$RefreshSeconds = 5
)

$protocolId = "librispeech_dev_clean_chapter_heldout_2fold__01ff6bcaaebb"
$embeddingRoot = Join-Path $ProjectRoot (
    "research_results\robustness_embeddings\" + $protocolId
)
$logRoot = Join-Path $ProjectRoot "research_results\robustness_embeddings"
$benchmarkLog = Join-Path $ProjectRoot (
    "research_results\robustness_benchmark\" +
    "parallel_automatic_benchmark_stdout.log"
)
$workersFile = Join-Path $logRoot "active_workers.json"

while ($true) {
    Clear-Host
    Write-Host "SPEAKER RECOGNITION - ROBUSTNESS EXPERIMENT" -ForegroundColor Cyan
    Write-Host ("Updated: " + (Get-Date -Format "dd-MM-yyyy HH:mm:ss"))
    Write-Host ("=" * 72)

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

    $completedFiles = @(
        Get-ChildItem `
            -LiteralPath $embeddingRoot `
            -Recurse `
            -Filter "metadata.json" `
            -ErrorAction SilentlyContinue |
        Where-Object { $_.FullName -like "*\full\*" }
    )
    $completed = $completedFiles.Count
    $percentage = [math]::Round(100.0 * $completed / 28, 1)
    Write-Host ""
    Write-Host "Completed conditions: $completed / 28 ($percentage%)" `
        -ForegroundColor Yellow

    Write-Host ""
    Write-Host "CURRENT CONDITIONS" -ForegroundColor Cyan
    Write-Host ("-" * 72)
    $states = @(
        Get-ChildItem `
            -LiteralPath $embeddingRoot `
            -Recurse `
            -Filter "checkpoint_state.json" `
            -ErrorAction SilentlyContinue |
        Where-Object { $_.FullName -like "*\full\*" } |
        ForEach-Object {
            $metadata = Join-Path $_.Directory "metadata.json"
            if (-not (Test-Path -LiteralPath $metadata)) {
                $state = Get-Content $_.FullName -Raw | ConvertFrom-Json
                [pscustomobject]@{
                    Model = $state.model_id
                    Condition = $state.condition
                    Processed = [int]$state.processed_rows
                    Success = [int]$state.successful_rows
                    Failed = (
                        [int]$state.processed_rows - [int]$state.successful_rows
                    )
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
        Write-Host "No active condition checkpoint."
    }

    Write-Host "LATEST WORKER PROGRESS" -ForegroundColor Cyan
    Write-Host ("-" * 72)
    foreach ($workerName in @("worker_a", "worker_b")) {
        $errorLog = Join-Path $logRoot ($workerName + "_stderr.log")
        Write-Host ("[$workerName]") -ForegroundColor DarkCyan
        if (Test-Path -LiteralPath $errorLog) {
            Get-Content $errorLog -Tail 1
        }
    }

    Write-Host ""
    Write-Host "AUTOMATIC BENCHMARK WATCHER" -ForegroundColor Cyan
    Write-Host ("-" * 72)
    if (Test-Path -LiteralPath $benchmarkLog) {
        Get-Content $benchmarkLog -Tail 1
    }

    Write-Host ""
    Write-Host (
        "This window only monitors progress. Closing it will NOT stop extraction."
    ) -ForegroundColor DarkGray
    Write-Host "Press Ctrl+C or close this window to stop monitoring." `
        -ForegroundColor DarkGray

    Start-Sleep -Seconds $RefreshSeconds
}
