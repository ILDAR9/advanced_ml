SHELL := /bin/bash

PORT=5000
SERVICE=web_app_6_1

REG_IP=95.216.168.158
REG_PORT=4444
IMAGE=web_app_6_1
TAG=${REG_IP}:${REG_PORT}/${IMAGE}

CONTAINER=uwsgi_app

build: stop
	docker build -t ${IMAGE}:latest --build-arg SECRET_NUMBER_ARG=1111 .

run:
	docker run -d -p ${PORT}:5000 --name ${CONTAINER} ${IMAGE}

stop:
	docker ps -f name=${CONTAINER} -aq | xargs docker stop | xargs docker rm || true

image_cmd:
	docker run -ti --entrypoint /bin/bash ${IMAGE}

prepare:
	docker tag ${IMAGE} ${TAG}
	docker push ${TAG}

service:
# Создаем сервис
	docker service create --name ${SERVICE}_prod ${TAG}:prod
	docker service create --name ${SERVICE}_test ${TAG}:test
# Открываем порт. Эти действия можно выполнить в одной команде, но если выполнять их по отдельности, то чуть проще понять, где что-то пошло не так
	docker service update --publish-add published=5000,target=${PORT} ${SERVICE}_prod
	docker service update --publish-add published=5001,target=${PORT} ${SERVICE}_test
# Докидываем реплик
	docker service update --replicas 1 ${SERVICE}_prod
	docker service update --replicas 1 ${SERVICE}_test
	docker service ls

service_stop:
	docker service rm ${SERVICE} || true
	docker service rm ${SERVICE}_prod || true
	docker service rm ${SERVICE}_test || true


hook_test:
	curl -X POST https://git.lab.karpov.courses/api/v4/projects/203/ref/TEST/trigger/pipeline?token=c5dc37c5bc840c7309f15658e0d0de

hook_prod:
	curl -X POST https://git.lab.karpov.courses/api/v4/projects/203/ref/PROD/trigger/pipeline?token=c5dc37c5bc840c7309f15658e0d0de