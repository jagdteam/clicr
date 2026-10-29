# Last 50 lines of the newest log under the clicr state dir, if any.
$home_ = if ($env:CLICR_HOME) { $env:CLICR_HOME } else { Join-Path $HOME '.clicr' }
$f = Get-ChildItem -Path $home_ -Recurse -Filter *.log -File -ErrorAction SilentlyContinue | Sort-Object LastWriteTime -Descending | Select-Object -First 1
if ($f) { Get-Content -Tail 50 $f.FullName } else { "No logs found. clicr state lives in $home_ (set by CLICR_HOME, see src/clicr/paths.py)." }
