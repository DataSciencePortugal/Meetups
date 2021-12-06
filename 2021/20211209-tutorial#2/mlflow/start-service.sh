#!/bin/bash
./mc alias set minio ${MLFLOW_S3_ENDPOINT_URL} ${AWS_ACCESS_KEY_ID} ${AWS_SECRET_ACCESS_KEY}
if [[ $(./mc ls minio | grep $AWS_BUCKET_NAME | wc -l) -eq 0 ]]; then
 	./mc mb minio/${AWS_BUCKET_NAME}
fi
./wait-for-it.sh db:5432 -t 90 -- mlflow server --backend-store-uri postgresql+psycopg2://${POSTGRES_USER}:${POSTGRES_PASSWORD}@db:5432/${POSTGRES_DATABASE} --default-artifact-root s3://${AWS_BUCKET_NAME} -h 0.0.0.0
