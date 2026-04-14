FROM python:3.7-slim-bookworm

RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app/pkg

COPY ["phase-2/Imputation Module/docker-imputation/requirements.txt", "./"]
RUN pip install --no-cache-dir -r requirements.txt

COPY ["phase-2/Imputation Module/src/", "/app/pkg/"]

RUN mkdir -p /app/cache /io /data

ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/pkg \
    IMPUTER_DATA_DIR=/data \
    IMPUTER_RECENT_HA_CSV="/data/Cons_Hotel Academic_2026-03-22_2026-04-10.csv" \
    IMPUTER_OUTPUT_DIR=/app/cache

ENTRYPOINT ["python", "impute_cli.py"]
CMD ["--help"]
