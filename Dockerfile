FROM tensorflow/tensorflow:2.3.2-gpu

# RUN  mkdir -p /work/
# WORKDIR /work/

# # Dockerfile
# ARG DOCKER_BASE_IMAGE=<BASE IMAGE NAME>
# FROM $DOCKER_BASE_IMAGE
# ARG USER=docker
# ARG UID=1000
# ARG GID=1000

# # default password for user
# ARG PW=docker

# # Option1: Using unencrypted password/ specifying password
# RUN useradd -m ${USER} --uid=${UID} && echo "${USER}:${PW}" | \
#       chpasswd
# # Option2: Using the same encrypted password as host

# #COPY /etc/group /etc/group 
# #COPY /etc/passwd /etc/passwd
# #COPY /etc/shadow /etc/shadow# Setup default user, when enter docker container
# USER ${UID}:${GID}
# WORKDIR /home/${USER}

COPY requirements.txt .

RUN /bin/bash -c "pip install --no-cache-dir -r requirements.txt"

# Set up SSH vorph
RUN apt-get update && apt-get install -y git
#openssh-server
RUN useradd -m -s /bin/bash vorph
#RUN mkdir /var/run/sshd
#RUN echo 'vorph:050188' | chpasswd
#EXPOSE 22
#CMD ["/usr/sbin/sshd", "-D"]