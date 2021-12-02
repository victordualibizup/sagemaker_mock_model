# quick how to

# base image - OS and python dependencies
docker build -t catboost-base .

# sg-custom-catboost image
docker build -f Dockerfile2 -t sgc .


docker run sgc train

docker run -it --entrypoint bash sgc
