param(
    [Parameter(Mandatory = $true)]
    [string]$ExtractionProcessIdsCsv,

    [string]$ProjectRoot = (Split-Path -Parent $PSScriptRoot),

    [string]$ProtocolId = (
        "librispeech_dev_clean_chapter_heldout_2fold__01ff6bcaaebb"
    ),

    [int]$ExpectedVariants = 28,

    [int]$PollSeconds = 30
)

$ErrorActionPreference = "Stop"
$embeddingRoot = Join-Path $ProjectRoot (
    "research_results\preprocessing_embeddings\" + $ProtocolId
)
$python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$protocol = Join-Path $ProjectRoot (
    "manifests\librispeech_dev_clean_chapter_heldout_2fold_protocol.csv"
)
$processIds = @(
    $ExtractionProcessIdsCsv.Split(",") |
        ForEach-Object { [int]$_.Trim() }
)

Set-Location -LiteralPath $ProjectRoot

while ($true) {
    $completed = @(
        Get-ChildItem -LiteralPath $embeddingRoot -Recurse `
            -Filter "metadata.json" -ErrorAction SilentlyContinue |
        Where-Object { $_.FullName -like "*\full\*" }
    ).Count
    Write-Output (
        "{0:u} Completed preprocessing variants: {1}/{2}" -f `
            (Get-Date), $completed, $ExpectedVariants
    )
    if ($completed -eq $ExpectedVariants) {
        break
    }
    if ($completed -gt $ExpectedVariants) {
        throw "Found more completed variants than expected: $completed/$ExpectedVariants"
    }
    $running = @(
        foreach ($processId in $processIds) {
            Get-Process -Id $processId -ErrorAction SilentlyContinue
        }
    )
    if ($running.Count -eq 0) {
        throw (
            "All extraction workers ended before completion " +
            "($completed/$ExpectedVariants)."
        )
    }
    Start-Sleep -Seconds $PollSeconds
}

Write-Output "All preprocessing embeddings complete. Starting benchmark."
& $python -m research.run_preprocessing_ablation `
    --all-models `
    --protocol-file $protocol `
    --output-tag full `
    --overwrite

if ($LASTEXITCODE -ne 0) {
    throw "Preprocessing benchmark failed with exit code $LASTEXITCODE."
}
Write-Output "Preprocessing benchmark and summary completed successfully."
