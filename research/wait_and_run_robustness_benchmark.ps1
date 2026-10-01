param(
    [Parameter(Mandatory = $true)]
    [string]$ExtractionProcessIdsCsv,

    [string]$ProjectRoot = (
        Split-Path -Parent $PSScriptRoot
    ),

    [string]$ProtocolId = (
        "librispeech_dev_clean_chapter_heldout_2fold__01ff6bcaaebb"
    ),

    [int]$ExpectedConditions = 28,

    [int]$PollSeconds = 30
)

$ErrorActionPreference = "Stop"

$embeddingRoot = Join-Path $ProjectRoot (
    "research_results\robustness_embeddings\" + $ProtocolId
)
$python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$protocol = Join-Path $ProjectRoot (
    "manifests\librispeech_dev_clean_chapter_heldout_2fold_protocol.csv"
)

Set-Location -LiteralPath $ProjectRoot

$extractionProcessIds = @(
    $ExtractionProcessIdsCsv.Split(",") |
        ForEach-Object { [int]$_.Trim() }
)

if ($extractionProcessIds.Count -eq 0) {
    throw "No extraction process IDs were supplied."
}

while ($true) {
    $completed = @(
        Get-ChildItem `
            -LiteralPath $embeddingRoot `
            -Recurse `
            -Filter "metadata.json" `
            -ErrorAction SilentlyContinue |
        Where-Object {
            $_.FullName -like "*\full\*"
        }
    ).Count

    Write-Output (
        "{0:u} Completed robustness conditions: {1}/{2}" -f `
            (Get-Date), $completed, $ExpectedConditions
    )

    if ($completed -eq $ExpectedConditions) {
        break
    }

    if ($completed -gt $ExpectedConditions) {
        throw (
            "Found more completed conditions than expected: " +
            "$completed/$ExpectedConditions"
        )
    }

    $runningExtractionProcesses = @(
        foreach ($processId in $extractionProcessIds) {
            Get-Process -Id $processId -ErrorAction SilentlyContinue
        }
    )

    if ($runningExtractionProcesses.Count -eq 0) {
        throw (
            "All extraction workers ended before all " +
            "conditions completed ($completed/$ExpectedConditions)."
        )
    }

    Start-Sleep -Seconds $PollSeconds
}

Write-Output "All condition embeddings are complete. Starting benchmark."

& $python `
    -m research.run_robustness_benchmark `
    --all-models `
    --protocol-file $protocol `
    --tag full `
    --overwrite

if ($LASTEXITCODE -ne 0) {
    throw "Robustness benchmark failed with exit code $LASTEXITCODE."
}

Write-Output "Robustness benchmark and summary completed successfully."
