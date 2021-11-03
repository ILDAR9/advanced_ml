SHELL := /bin/bash

IMAGE=web_app_5_2
CONTAINER=uwsgi_app
PORT=5000
# SERVICE=web_app_5_2
SERVICE=web_app

REG_IP=95.216.168.158
REG_PORT=4444
TAG=${REG_IP}:${REG_PORT}/${IMAGE}

build: stop
	docker build -t ${IMAGE}:latest .

run:
	docker run -d -p ${PORT}:5000 --name ${CONTAINER} ${IMAGE}

attach:
	docker exec -it ${CONTAINER} bash

stop:
	docker ps -f name=${CONTAINER} -aq | xargs docker stop | xargs docker rm || true

image_cmd:
	docker run -ti --entrypoint /bin/bash ${IMAGE}

swarm:
	docker swarm init

prepare:
	docker tag ${IMAGE} ${TAG}
	docker push ${TAG}

test:
	docker pull ${TAG}

service:
# Создаем сервис
	docker service create --name ${SERVICE} ${TAG}
# Открываем порт. Эти действия можно выполнить в одной команде, но если выполнять их по отдельности, то чуть проще понять, где что-то пошло не так
	docker service update --publish-add published=${PORT},target=5000 ${SERVICE}
# Докидываем реплик
	docker service update --replicas 6 ${SERVICE}

service_stop:
	docker service rm ${SERVICE}

# utils
