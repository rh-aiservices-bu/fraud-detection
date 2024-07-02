import ray
import os
import boto3
import botocore



@ray.remote
def get_object_keys():
    aws_access_key_id = os.environ.get('AWS_ACCESS_KEY_ID')
    aws_secret_access_key = os.environ.get('AWS_SECRET_ACCESS_KEY')
    endpoint_url = os.environ.get('AWS_S3_ENDPOINT')
    region_name = os.environ.get('AWS_DEFAULT_REGION')
    bucket_name = os.environ.get('AWS_S3_BUCKET')

    session = boto3.session.Session(aws_access_key_id=aws_access_key_id,
                                    aws_secret_access_key=aws_secret_access_key)


    s3_resource = session.resource(
        's3',
        config=botocore.client.Config(signature_version='s3v4'),
        endpoint_url=endpoint_url,
        region_name=region_name)

    bucket = s3_resource.Bucket(bucket_name)


    def list_objects(prefix):
        filter = bucket.objects.filter(Prefix=prefix)
        keys = [obj.key for obj in filter.all()]

        return keys

    return list_objects("")

# Automatically connect to the running Ray cluster.
ray.init()
keys = ray.get(get_object_keys.remote())
for key in keys:
    print(key)
