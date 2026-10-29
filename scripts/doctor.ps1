Set-Location (Join-Path $PSScriptRoot "..")
uv run clicr doctor @args
exit $LASTEXITCODE
