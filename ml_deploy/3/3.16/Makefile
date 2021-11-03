SHELL := /bin/bash

IMAGE=uwsgi_hw3
CONTAINER=uwsgi_app
PORT=5000

# uwsgi --http 0.0.0.0:9090 --wsgi-file app_flask.py --master --processes 2 --threads 5 --lazy-apps --master-fifo=/var/master.fifo --touch-chain-reload=/var/reloadFile

run_local:
	export FLASK_APP=main && \
	source /root/env/bin/activate && \
	cd app && \
	python main.py

build: stop
	docker build -t ${IMAGE}:latest .

run:
	docker run -d -p ${PORT}:5000 --name ${CONTAINER} ${IMAGE}

exec:
	docker exec -d ${CONTAINER} touch /tmp/execWorks

attach:
	docker exec -it ${CONTAINER} bash

stop:
	docker ps -f name=${CONTAINER} -aq | xargs docker stop | xargs docker rm || true

image_cmd:
	# docker run -ti --entrypoint /bin/bash tiangolo/uwsgi-nginx-flask:python3.8
	docker run -ti --entrypoint /bin/bash ${IMAGE}

prepare_ini:
	uwsgiconf compile > myconf.ini

ncores:
	lscpu | grep -E '^Thread|^Core|^Socket|^CPU\('

make clear:
	rm -rf app/__pycache__

make clean: clear