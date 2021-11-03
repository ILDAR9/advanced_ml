SHELL := /bin/bash

IMAGE=service-discovery
CONTAINER=web_app_2
PORT=5000

run_local:
	export FLASK_APP=main && \
	source /root/env/bin/activate && \
	python main.py

build:
	docker build -t ${IMAGE}:latest .

run:
	docker run -p ${PORT}:${PORT} --env PORT=${PORT}  --name ${CONTAINER} ${IMAGE}

rund:
	docker run -d -p ${PORT}:${PORT} --env PORT=${PORT} --name ${CONTAINER} ${IMAGE}

ti:
	docker run -ti --entrypoint /bin/bash ${IMAGE}

attach:
	docker attach ${CONTAINER}

exec:
	docker exec -it ${CONTAINER} bash

kill:
	docker kill --signal=SIGTERM ${CONTAINER}
	docker ps

stop:
	docker ps -f name=web_app -aq | xargs docker stop | xargs docker rm

clear:
	rm -rf __pycache__
	rm -rf hw/__pycache__

several:
	docker run -d -p 5000:5000 --name web_app_0 service-discovery
	docker run -d -p 5001:5001 --name web_app_1 --env PORT=5001 service-discovery
	docker run -d -p 5002:5002 --name web_app_2 --env PORT=5002 service-discovery
	docker ps -f name=web_app

clean: clear