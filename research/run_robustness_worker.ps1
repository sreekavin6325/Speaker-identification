param(
    [Parameter(Mandatory = $true)]
    [string]$ModelsCsv,

    [string]$ProjectRoot = (Split-Path -Parent $PSScriptRoot),

    [string]$Tag = "full",

    [int]$Threads = 8
)

$ErrorActionPreference = "Stop"
$env:OMP_NUM_THREADS = [string]$Threads
$env:MKL_NUM_THREADS = [string]$Threads
$env:OPENBLAS_NUM_THREADS = [string]$Threads

$python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$protocolId = "librispeech_dev_clean_chapter_heldout_2fold__01ff6bcaaebb"
$root = Join-Path $ProjectRoot (
    "research_results\robustness_embeddings\" + $protocolId
)
$conditions = @(
    "short_0p5s",
    "short_1s",
    "short_2s",
    "short_3s",
    "noise_white_20db",
    "noise_white_10db",
    "noise_white_0db"
)
$models = @(
    $ModelsCsv.Split(",") |
        ForEach-Object { $_.Trim() } |
        Where-Object { $_ }
)

if ($models.Count -eq 0) {
    throw "ModelsCsv did not contain a model ID."
}

Set-Location -LiteralPath $ProjectRoot

foreach ($model in $models) {
    foreach ($condition in $conditions) {
        $destination = Join-Path $root (
            "$model\$condition\$Tag"
        )
        $metadata = Join-Path $destination "metadata.json"
        $checkpoint = Join-Path $destination "checkpoint_state.json"

        if (Test-Path -LiteralPath $metadata) {
            Write-Output "Skipping completed: $model / $condition"
            continue
        }

        $arguments = @(
            "-m",
            "research.extract_robustness_embeddings",
            "--model",
            $model,
            "--condition",
            $condition,
            "--tag",
            $Tag,
            "--checkpoint-every",
            "100"
        )

        if (Test-Path -LiteralPath $checkpoint) {
            $arguments += "--resume"
            Write-Output "Resuming: $model / $condition"
        }
        else {
            Write-Output "Starting: $model / $condition"
        }

        & $python @arguments

        if ($LASTEXITCODE -ne 0) {
            throw (
                "Extraction failed for $model / $condition " +
                "with exit code $LASTEXITCODE."
            )
        }
    }
}

Write-Output "Worker completed models: $ModelsCsv"
