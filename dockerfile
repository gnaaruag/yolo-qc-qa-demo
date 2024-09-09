FROM python:3.11-slim

WORKDIR /app

# COPY requirements.txt ./requirements.txt

# RUN apt-get update && apt-get install -y --no-install-recommends \
#         ca-certificates \
#         netbase \
#         && rm -rf /var/lib/apt/lists/*