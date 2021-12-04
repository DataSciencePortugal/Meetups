FROM continuumio/miniconda3:latest

ADD . /app
WORKDIR /app

COPY start-service.sh start-service.sh
RUN chmod +x start-service.sh 

COPY wait-for-it.sh wait-for-it.sh 
RUN chmod +x wait-for-it.sh

RUN pip install mlflow==1.21.0 boto3 psycopg2-binary

RUN wget https://dl.min.io/client/mc/release/linux-amd64/mc

RUN chmod +x /app/mc
