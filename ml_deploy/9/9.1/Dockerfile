FROM tiangolo/uwsgi-nginx-flask:python3.9
ARG MODULE_NAME

RUN pip install \
    mlflow==1.14.1 \
    psycopg2-binary \
    scikit-learn \
    boto3

RUN echo "MODULE_NAME=${MODULE_NAME}"
COPY ./${MODULE_NAME} /app
WORKDIR /app

EXPOSE 5001

CMD "uwsgi"