import os 

from minio import Minio
from minio.error import InvalidResponseError

accessID = "IAM_ACCESS_KEY"
accessSecret = "IAM_SECRET_KEY"
minioUrl = "http://95.216.168.89:19001"
bucketName = "mlflow"

minioUrlHostWithPort = minioUrl.split('//')[1]
print('[*] minio url: ', minioUrlHostWithPort)

s3Client = Minio(
    minioUrlHostWithPort,
    access_key=accessID,
    secret_key=accessSecret,
    secure=False
)

s3Client.make_bucket(bucketName)