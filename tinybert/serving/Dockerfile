FROM python:3.10.9-slim-buster

RUN apt-get update -y \
    && apt-get install -y --no-install-recommends gcc linux-libc-dev curl net-tools

WORKDIR /app
ADD . /app

RUN pip install -r requirements.txt --no-cache-dir
RUN pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cpu

RUN mkdir models

EXPOSE 80
