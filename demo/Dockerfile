FROM python:3.7-slim-bookworm

# System dependencies (h5py / scipy may need compilation)
RUN apt-get update && \
    apt-get install -y --no-install-recommends libhdf5-dev gcc && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app/demo

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Code is volume-mounted, not copied,rebuild only when deps change
RUN mkdir -p /app/demo/output

ENV MPLBACKEND=Agg
ENV TF_CPP_MIN_LOG_LEVEL=2
ENV PYTHONUNBUFFERED=1

ENTRYPOINT ["python", "run_demo.py"]
CMD ["--target-date", "2026-01-15"]
