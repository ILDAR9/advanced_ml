FROM tiangolo/uwsgi-nginx-flask:python3.9
ARG SECRET_NUMBER_ARG

COPY ./app /app
WORKDIR /app

EXPOSE 5000

ENV SECRET_NUMBER ${SECRET_NUMBER_ARG}

CMD "uwsgi"