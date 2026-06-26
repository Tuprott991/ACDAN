# Reproduce the headline offline results into results/ (Windows PowerShell).
# Usage:  powershell -ExecutionPolicy Bypass -File scripts/reproduce.ps1
$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")
New-Item -ItemType Directory -Force -Path results | Out-Null

Write-Host ">> Full ACDAN evaluation (seed 0)"
python -m acdan.cli eval --config configs/default.yaml --seed 0 --out results/full_seed0.json

Write-Host ">> Baseline CoT evaluation (seed 0)"
python -m acdan.cli eval --config configs/ablations/baseline_cot.yaml --seed 0 --out results/baseline_seed0.json

Write-Host ">> Ablation grid (seed 0)"
python -m acdan.cli ablation --config configs/default.yaml --seed 0 --out results/ablation_seed0.json

Write-Host ">> Demo trace (one task)"
python -m acdan.cli demo --config configs/default.yaml --seed 0 --task-index 0

Write-Host ""
Write-Host "Done. JSON artefacts written to results/."
