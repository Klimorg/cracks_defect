#FROM tensorflow/tensorflow:nightly-gpu
FROM nvcr.io/nvidia/tensorflow:20.12-tf2-py3

COPY requirements.txt .

RUN /bin/bash -c "pip install --no-cache-dir -r requirements.txt"


RUN apt-get update && apt-get install -y git

RUN useradd -m -s /bin/bash vorph
#EXPOSE 8001
