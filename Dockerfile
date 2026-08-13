# Reproducible run/test image for the Care agent.
#
#   docker build -t care-agent .
#   docker run --rm -v "$PWD":/out care-agent \
#       python run_all.py --json /out/report.json --md /out/report.md
#
# The default command runs the full evaluation suite offline: no API key, no network, and a
# deterministic result. Set ANTHROPIC_API_KEY and add --live to run against the real Claude API.

FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Dependencies first, so editing source does not invalidate the install layer.
COPY requirements.txt pyproject.toml ./
RUN pip install -r requirements.txt

# Then the project itself.
COPY src/ ./src/
COPY policy/ ./policy/
COPY eval/ ./eval/
COPY tests/ ./tests/
COPY scripts/ ./scripts/
COPY run_all.py ./
RUN pip install --no-deps -e .

# Run unprivileged. Reports are written relative to /app unless an absolute path is given,
# so mount a volume and pass --json/--md to collect them on the host.
RUN useradd --create-home --uid 1000 care && chown -R care:care /app
USER care

CMD ["python", "run_all.py"]
