SHELL := /bin/bash


IMAGE=hello-world
HOST_IP=95.216.168.158
PORT=4444
TAG=${HOST_IP}:${PORT}/${IMAGE}
# 95.216.168.158:4444/hello-world

# /etc/docker/daemon.json

registry:
	docker run -d -p 4444:5000 --restart=always --name registry registry:2

stop:
	docker ps -f name=registry -aq | xargs docker stop | xargs docker rm || true

prepare:
	docker pull ${IMAGE}
	docker tag ${IMAGE} ${TAG}
	docker push ${TAG}
	docker image remove ${IMAGE}
	docker image remove ${TAG}

test:
	docker pull ${TAG}

rm_containers:
	docker rm -f $(docker ps -a -q)