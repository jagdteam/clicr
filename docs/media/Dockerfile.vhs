# Records docs/media/demo.tape with VHS. Build from the repo root:
#   docker build -f docs/media/Dockerfile.vhs -t clicr-vhs .
FROM ghcr.io/charmbracelet/vhs:v0.12.1

COPY --from=ghcr.io/astral-sh/uv:0.13.5 /uv /uvx /usr/local/bin/

COPY . /clicr
WORKDIR /clicr

# Runtime only (no dev group); uv fetches Python 3.12 itself. Lockfile is not rewritten.
RUN uv sync --frozen --no-dev
ENV PATH="/clicr/.venv/bin:${PATH}"
# tree-sitter grammars are downloaded on demand; fetch them at build time. doctor exits 1
# here because Ollama is unreachable during the build, so check the grammar row instead.
RUN clicr doctor --fix; clicr doctor | grep -q "grammars.*OK"

# The base image's entrypoint is `vhs`; pass the tape path as the argument.
