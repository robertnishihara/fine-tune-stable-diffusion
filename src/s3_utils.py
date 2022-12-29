import boto3
import botocore

import logging
import os

logger = logging.getLogger(__name__)


def s3_exists(s3filepath):
    s3 = boto3.resource("s3")
    s3filepath = s3filepath[5:]  # remove s3://

    bucket_name, key = s3filepath.split("/", 1)

    try:
        s3.Object(bucket_name, key).load()
    except botocore.exceptions.ClientError as e:
        if e.response["Error"]["Code"] == "404":
            return False
        else:
            # Something else has gone wrong.
            raise
    return True


def download_file_from_s3(s3_path: str, target_path: str) -> str:
    logger.info(f"Starting s3 download: {s3_path}")
    # Example tar_file_path = "s3://anyscale-temp/diffusion-demo/checkpoint-demo.tar.gz"

    s3 = boto3.resource("s3")
    # remove s3:// from the path
    s3_path = s3_path[5:]

    bucket_name = s3_path.split("/")[0]
    s3_key = "/".join(s3_path.split("/")[1:])
    s3.Bucket(bucket_name).download_file(s3_key, target_path)
    logger.info(f"Downloaded s3 path to local path: {s3_path} -> {target_path}")
    logger.info(f"Print listdir {os.listdir(os.path.dirname(target_path))}")
    return target_path


def write_to_s3(local_file, s3_target_path):
    s3 = boto3.resource("s3")
    s3_target_path = s3_target_path[5:]  # remove s3://

    bucket_name, key = s3_target_path.split("/", 1)
    s3.Bucket(bucket_name).upload_file(local_file, key)
    print(f"Uploaded {local_file} to {s3_target_path}")
    return s3_target_path


def zip_dir(file_directory, target_path):
    import shutil
    import os

    # if ends with zip, remove from target path
    if target_path.endswith(".zip"):
        target_path = target_path[:-4]
    shutil.make_archive(target_path, "zip", file_directory)
    return target_path + ".zip"


def unzip_dir(zip_path, target_path):
    import shutil
    import os

    shutil.unpack_archive(zip_path, target_path)
    return target_path


def tar_dir(file_directory, target_path):
    import tarfile
    import os

    # if ends with tar.gz, remove from target path
    if target_path.endswith(".tar.gz"):
        target_path = target_path[:-7]
    with tarfile.open(target_path + ".tar.gz", "w:gz") as tar:
        tar.add(file_directory, arcname=os.path.basename(file_directory))
    return target_path + ".tar.gz"


def untar_dir(tar_file_path: str, target_path: str) -> str:
    # Returns absolute path to the extracted folder
    import tarfile

    tar = tarfile.open(tar_file_path, "r:gz")
    tar.extractall(target_path)
    tar.close()
    return target_path
