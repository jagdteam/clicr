# Run the full verify; print only failures and the final summary line.
Set-Location (Join-Path $PSScriptRoot '..')
$out = & cmd /c 'uv run ruff check . && uv run ruff format --check . && uv run pytest -q 2>&1' 2>&1 | ForEach-Object { "$_" }
$rc = $LASTEXITCODE
if ($rc -ne 0) { $out | Select-String -CaseSensitive -Pattern 'FAILED|ERROR|error|Would reformat|^[A-Z]+[0-9]+ ' | ForEach-Object { $_.Line } }
$out | Select-Object -Last 1
exit $rc
