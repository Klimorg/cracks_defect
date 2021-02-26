FROM nvcr.io/nvidia/tensorflow:20.12-tf2-py3

ARG USERNAME=vorph
ARG USER_UID=1000
ARG USER_GID=$USER_UID

COPY requirements.txt .
COPY requirements-dev.txt .

RUN /bin/bash -c "pip install --no-cache-dir -r requirements.txt"

RUN /bin/bash -c "pip install --no-cache-dir -r requirements-dev.txt"

EXPOSE 5000
EXPOSE 8001