FROM python:3

COPY ./requirements.txt /app/requirements.txt

WORKDIR /app

RUN pip install -r requirements.txt

COPY . /app

ENV export FLASK_APP main

ENTRYPOINT [ "/bin/bash" ]
CMD [ "run.sh" ]