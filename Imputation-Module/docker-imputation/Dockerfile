FROM python:3.7-slim-bookworm

RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app/pkg

COPY ["Imputation-Module/docker-imputation/requirements.txt", "./"]
RUN pip install --no-cache-dir -r requirements.txt

COPY ["Imputation-Module/src/", "/app/pkg/"]

RUN mkdir -p /app/cache /io /io/audit_logs /data

ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/pkg \
    IMPUTER_DATA_DIR=/data \
    IMPUTER_RECENT_HA_CSV="/data/Cons_Hotel Academic_2026-03-22_2026-04-10.csv" \
    IMPUTER_OUTPUT_DIR=/app/cache \
    IMPUTER_AUDIT_LOG_DIR=/io/audit_logs

ENTRYPOINT ["python", "impute_cli.py"]
CMD ["--help"]
