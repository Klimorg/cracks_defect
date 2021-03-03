#FROM nvcr.io/nvidia/tensorflow:21.02-tf2-py3
FROM nvcr.io/nvidia/tensorflow:20.12-tf2-py3


COPY requirements.txt .
COPY requirements-dev.txt .

ARG USERNAME=vorph
ARG USER_UID=1000
ARG USER_GID=1000

RUN groupadd -g $USER_GID -o $USERNAME
RUN useradd -m -u $USER_UID -g $USER_GID -o -s /bin/bash $USERNAME

USER $USERNAME

RUN /bin/bash -c "pip install --no-cache-dir -r requirements.txt"

# RUN /bin/bash -c "pip install --no-cache-dir -r requirements-dev.txt"

# EXPOSE 5000
# EXPOSE 8001