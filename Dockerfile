FROM nvcr.io/nvidia/tensorflow:21.02-tf2-py3
#FROM nvcr.io/nvidia/tensorflow:20.12-tf2-py3


COPY requirements.txt .
COPY requirements-dev.txt .

# set username/id to be non root user and get same rights as in my ubuntu
ARG USERNAME=vorph
ARG USER_UID=1000
ARG USER_GID=1000

RUN groupadd -g $USER_GID -o $USERNAME
RUN useradd -m -u $USER_UID -g $USER_GID -o -s /bin/bash $USERNAME

USER $USERNAME

# set path for python libs
ENV PATH "$PATH:/home/vorph/.local/bin"

RUN /bin/bash -c "pip install -r requirements.txt"

RUN /bin/bash -c "pip install -r requirements-dev.txt"

# expose ports, 5000 for mlflow, 8001 for mkdocs
EXPOSE 5000
EXPOSE 8001