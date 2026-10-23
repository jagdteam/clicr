# Records docs/media/demo.gif (+ ask.png, explain.png) with VHS in Docker.
# Needs: Docker, and Ollama running on the host with the chat and embedding models pulled.
$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

docker build -f docs/media/Dockerfile.vhs -t clicr-vhs .
if ($LASTEXITCODE -ne 0) { throw 'docker build failed' }

# The mount replaces /clicr/docs/media so the outputs land in the repo.
docker run --rm `
    --add-host=host.docker.internal:host-gateway `
    -e CLICR_OLLAMA_HOST=http://host.docker.internal:11434 `
    -v "${root}/docs/media:/clicr/docs/media" `
    clicr-vhs docs/media/demo.tape
if ($LASTEXITCODE -ne 0) { throw 'docker run failed' }

foreach ($name in 'demo.gif', 'ask.png', 'explain.png') {
    $f = Join-Path $root "docs/media/$name"
    if (Test-Path $f) {
        '{0}: {1:N0} KB' -f $name, ((Get-Item $f).Length / 1KB)
    } else {
        Write-Warning "$name was not written"
    }
}
