FROM python:3.11-slim


COPY requirements.txt ./requirements.txt

RUN apt-get update && apt-get install -y git

RUN git config --global http.postBuffer 524288000

RUN git clone --depth 1 https://github.com/gnaaruag/yolo-qc-qa-demo.git /app

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
        netbase \
        && rm -rf /var/lib/apt/lists/*



RUN pip3 install -r requirements.txt

EXPOSE 8501

ENTRYPOINT ["streamlit", "run"]

CMD ["app.py"]
