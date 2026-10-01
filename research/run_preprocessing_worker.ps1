param(
    [Parameter(Mandatory = $true)]
    [string]$ModelsCsv,

    [string]$ProjectRoot = (Split-Path -Parent $PSScriptRoot),

    [string]$Tag = "full",

    [int]$Threads = 4
)

$ErrorActionPreference = "Stop"
$env:OMP_NUM_THREADS = [string]$Threads
$env:MKL_NUM_THREADS = [string]$Threads
$env:OPENBLAS_NUM_THREADS = [string]$Threads

$python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$protocolId = "librispeech_dev_clean_chapter_heldout_2fold__01ff6bcaaebb"
$root = Join-Path $ProjectRoot (
    "research_results\preprocessing_embeddings\" + $protocolId
)
$variants = @(
    "normalize",
    "trim",
    "denoise",
    "trim_normalize",
    "denoise_normalize",
    "trim_denoise",
    "trim_denoise_normalize"
)
$models = @(
    $ModelsCsv.Split(",") |
        ForEach-Object { $_.Trim() } |
        Where-Object { $_ }
)

if ($models.Count -eq 0) {
    throw "ModelsCsv did not contain a model ID."
}
if (-not (Test-Path -LiteralPath $python)) {
    throw "Virtual-environment Python was not found: $python"
}

Set-Location -LiteralPath $ProjectRoot

foreach ($model in $models) {
    foreach ($variant in $variants) {
        $destination = Join-Path $root "$model\$variant\$Tag"
        $metadata = Join-Path $destination "metadata.json"
        $checkpoint = Join-Path $destination "checkpoint_state.json"

        if (Test-Path -LiteralPath $metadata) {
            Write-Output "Skipping completed: $model / $variant"
            continue
        }

        $arguments = @(
            "-m",
            "research.extract_preprocessing_embeddings",
            "--model",
            $model,
            "--variant",
            $variant,
            "--tag",
            $Tag,
            "--checkpoint-every",
            "100"
        )

        if (Test-Path -LiteralPath $checkpoint) {
            $arguments += "--resume"
            Write-Output "Resuming: $model / $variant"
        }
        else {
            Write-Output "Starting: $model / $variant"
        }

        & $python @arguments
        if ($LASTEXITCODE -ne 0) {
            throw (
                "Extraction failed for $model / $variant " +
                "with exit code $LASTEXITCODE."
            )
        }
    }
}

Write-Output "Worker completed models: $ModelsCsv"
