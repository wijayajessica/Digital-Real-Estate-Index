import boto3
import os


# https://stackoverflow.com/questions/49772151/download-a-folder-from-s3-using-boto3
def downloadDirectoryFroms3(bucketName, remoteDirectoryName):
    s3_resource = boto3.resource('s3')
    bucket = s3_resource.Bucket(bucketName) 
    for obj in bucket.objects.filter(Prefix = remoteDirectoryName):
        if not os.path.exists(os.path.dirname(obj.key)):
            os.makedirs(os.path.dirname(obj.key))
        if not os.path.exists(obj.key):
            bucket.download_file(obj.key, obj.key)

downloadDirectoryFroms3('rex-harvard-iacs', 'index-team-data/')

