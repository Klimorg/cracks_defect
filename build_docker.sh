
sudo docker build --build-arg USER=$USER --build-arg UID=$UID- -rm -f Dockerfile -t docker_cracks .

docker build --build-arg USER=$USER \
             --build-arg UID=$UID \
             --build-arg GID=$GID \
             --build-arg PW=<PASSWORD IN CONTAINER> \
             -t <IMAGE NAME> \
             -f <DOCKERFILE NAME>\
             .

export UID=$(id -u)
export GID=$(id -g)
docker build --build-arg USER=$USER \
             --build-arg UID=$UID \
             --build-arg GID=$GID \
             --build-arg PW=<PASSWORD IN CONTAINER> \
             -t <IMAGE NAME> \
             -f <DOCKERFILE NAME>\
             .