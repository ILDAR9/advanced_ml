FROM tiangolo/uwsgi-nginx-flask:python3.7
RUN pip install requests uwsgi
COPY ./app /app
WORKDIR /app

EXPOSE 5000
# ENV FLASK_APP main

CMD "uwsgi"