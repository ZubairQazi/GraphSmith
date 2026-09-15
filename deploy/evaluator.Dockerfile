FROM python:3.14-slim@sha256:cad9a2c871761c413caa6fdd6441c783451e740a48aaeba60ae62a8b53525ef6
WORKDIR /app
COPY graphsmith/__init__.py graphsmith/data.py graphsmith/metrics.py graphsmith/evaluator.py /app/graphsmith/
RUN mkdir /vault && chown 65532:65532 /vault
USER 65532:65532
ENV PYTHONDONTWRITEBYTECODE=1
VOLUME /vault
EXPOSE 8080
CMD ["python", "-m", "graphsmith.evaluator", "/vault/evaluator.sqlite"]
