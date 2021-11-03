SHELL := /bin/bash

IMAGE=web_app_6_1
CONTAINER=uwsgi_app
PORT=5000
SERVICE=web_app_6_1

start:
	docker-compose up -d

stop:
	docker-compose down

restart: stop start

build_mlflow:
	docker build -t mlflow:latest -f Dockerfile.mlflow .

image_cmd:
	docker run -ti --entrypoint /bin/bash ${IMAGE}
