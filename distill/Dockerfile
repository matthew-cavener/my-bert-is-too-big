FROM tensorflow/tensorflow:1.13.2-py3

RUN mkdir /app
WORKDIR /app

RUN apt-get update && \
  apt-get install -y \
  python3-dev

COPY ./requirements.txt .
RUN pip install --upgrade pip && \
  pip install -r requirements.txt

ENV TFHUB_CACHE_DIR=/app/distill/models/hub_modules
ENV PYTHONUNBUFFERED=0
ENV TF_CPP_MIN_LOG_LEVEL=3

COPY . ./distill
EXPOSE 8081