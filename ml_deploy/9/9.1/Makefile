SHELL := /bin/bash

IMAGE=web_app_9_1
TAG1=${IMAGE}:1
TAG2=${IMAGE}:2
PORT=5001
SERVICE=web_app_9

REG_IP=95.216.168.158
REG_PORT=4444

CONTAINER=uwsgi_app

build: stop_container
	docker build -t ${TAG1} --build-arg MODULE_NAME=app .
	docker build -t ${TAG2} --build-arg MODULE_NAME=app2 .

run:
	# docker run -d -p ${PORT}:${PORT} --name ${CONTAINER} ${TAG1}
	docker run -d -p ${PORT}:${PORT} --name ${CONTAINER} ${TAG2}

stop_container:
	docker ps -f name=${CONTAINER} -aq | xargs docker stop | xargs docker rm || true

image_cmd:
	docker run -ti --entrypoint /bin/bash ${TAG2}

push:
	docker tag ${TAG1} ${REG_IP}:${REG_PORT}/${IMAGE}:1
	docker push ${REG_IP}:${REG_PORT}/${IMAGE}:1
	docker tag ${TAG2} ${REG_IP}:${REG_PORT}/${IMAGE}:2
	docker push ${REG_IP}:${REG_PORT}/${IMAGE}:2

# docker-compose

start: stop_container
	docker-compose up -d

stop:
	docker-compose down

restart: stop start

build_mlflow:
	docker build -t mlflow:latest -f Dockerfile.mlflow .

service:
# Создаем сервис
	docker service create --name ${SERVICE} ${REG_IP}:${REG_PORT}/${IMAGE}:1
# Открываем порт. Эти действия можно выполнить в одной команде, но если выполнять их по отдельности, то чуть проще понять, где что-то пошло не так
	docker service update --publish-add published=${PORT},target=${PORT} ${SERVICE}
# Докидываем реплик
	docker service update --replicas 1 ${SERVICE}

service_stop:
	docker service rm ${SERVICE}

hook:
	curl -X POST https://git.lab.karpov.courses/api/v4/projects/206/ref/master/trigger/pipeline?token=2ddcf5be76a361fe549d50e03b5f05