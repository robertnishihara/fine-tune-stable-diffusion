"""Simple stable diffusion app.

How to run:
    $ cd src/ && uvicorn server:app --reload

How to test:
"""
from typing import Optional
import os
import uuid
import tempfile
from fastapi import FastAPI, UploadFile, HTTPException
from fastapi.logger import logger
from pydantic import Field
from fastapi.responses import Response

from anyscale.sdk.anyscale_client.models import (
    CreateProductionJob,
    ClusterComputeConfig,
    ComputeNodeType,
    CreateProductionService,
    CreateProductionJobConfig,
)
from anyscale import AnyscaleSDK
import logging

# local module
from db_client import DBClient

app = FastAPI()


def s3_exists(s3filepath):
    import boto3
    import botocore

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


def submit_anyscale_job(files, file_directory, job_name):
    sdk = AnyscaleSDK()
    runtime_env = {
        "working_dir": file_directory,
        "upload_path": "s3://anyscale-temp/diffusion-demo/",
    }
    job_config = {
        # IDs can be found on Anyscale Console under Configurations.
        # The IDs below are examples and should be replaced with your own IDs.
        # 'compute_config_id': 'cpt_U8RCfD7Wr1vCD4iqGi4cBbj1',
        "runtime_env": runtime_env,
        # The compute config can also specified as a one-off instead:
        "compute_config": ClusterComputeConfig(
            cloud_id="cld_V1U8Jk3ZgEQQbc7zkeBq24iX",
            region="us-west-2",
            head_node_type=ComputeNodeType(
                name="head",
                instance_type="m5.large",
            ),
            worker_node_types=[],
        ),
        # The id of the cluster env build
        "build_id": "bld_1277XIinoJmiM8Z3gNdcHN",
        # 'runtime_env': {
        #     'working_dir': 's3://my_bucket/my_job_files.zip'
        # },
        "entrypoint": f"python train.py --image-dir {file_directory}",
        "max_retries": 3,
    }

    job = sdk.create_job(
        CreateProductionJob(
            name=job_name,
            description="Stable diffusion training job",
            # project_id can be found in the URL by navigating to the project in Anyscale Console
            project_id="prj_7S7Os7XBvO6vdiVC1J0lgj",
            config=job_config,
        )
    )
    return job


def submit_service(model_id, model_path, local=False):
    from types import SimpleNamespace

    if local:
        import subprocess

        # This is probably all broken rn
        subprocess.run("serve run src.service.serve_model:entrypoint".split(" "))
        result = SimpleNamespace()
        result.url = "http://localhost:8001"
    else:
        assert s3_exists(model_path), f"{model_path} does not exist on s3."
        sdk = AnyscaleSDK()
        # current_dir = os.path.dirname(os.path.abspath(__file__))
        response = sdk.apply_service(
            create_production_service=CreateProductionService(
                name=f"stable-diffusion-{model_id}",
                description="Stable diffusion services",
                # project_id can be found in the URL
                # https://console.anyscale-staging.com/o/anyscale-internal/projects/prj_j2bynt35acxvgtg6riahpzqk
                project_id="prj_j2bynt35acxvgtg6riahpzqk",
                healthcheck_url="/-/healthz",
                config=dict(
                    # https://console.anyscale-staging.com/o/anyscale-internal/configurations/cluster-computes/cpt_v1hkxu5rd61ql5nd268fen83t7
                    compute_config_id="cpt_v1hkxu5rd61ql5nd268fen83t7",
                    # https://console.anyscale-staging.com/o/anyscale-internal/configurations/app-config-details/bld_hu28yb4llwb66fxh3cd9dzh9ty
                    build_id="bld_hu28yb4llwb66fxh3cd9dzh9ty",
                    runtime_env=dict(
                        working_dir="https://github.com/robertnishihara/fine-tune-stable-diffusion/archive/refs/heads/main.zip",
                        env_vars=dict(RANDOM=str(uuid.uuid4()), MODEL_PATH=model_path),
                    ),
                    entrypoint="serve run --non-blocking src.service.serve_model:entrypoint",
                    access="public",
                ),
            )
        )
        result = response.result
        print(f"Service deployment url: {result.url}")

    return result


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.post("/deploy/{model_id}")
async def deploy(model_id: str, model_path: Optional[str] = None):
    # TODO: deploy model to Anyscale
    local = False
    if model_id == "TEST":
        local = True

    if model_path is None:
        with DBClient(model=model_id) as db:
            model_path = db["path"]

    result = submit_service(model_id, model_path, local=local)
    with DBClient(model=model_id) as db:
        db["url"] = result.url
        db["service_id"] = str(result.id)
        db["service_name"] = str(result.name)
        db["token"] = get_service_token(result.id)
        print("URL: ", result.url)
        print("service_id", result.id)
        print("service_name", str(result.name))
        print("Token: ", get_service_token(result.id))
    return {
        "message": "Deployed model successfully!",
        "model_id": model_id,
        "model_url": result.url,
    }


@app.post("/terminate/{model_id}")
async def terminate(model_id: str):
    sdk = AnyscaleSDK()
    with DBClient(model=model_id) as db:
        service_id = db["service_id"]
        response = sdk.terminate_service(service_id)
        db.clear_all()
    return {
        "message": "Terminated model successfully!",
        "model_id": model_id,
    }


def get_service_token(service_id):
    sdk = AnyscaleSDK()
    service = sdk.get_service(service_id).result
    return service.token


def get_service_url(service_id, model_id, cache=True):
    sdk = AnyscaleSDK()
    service = sdk.get_service(service_id).result
    service_url = service.url
    if service_url is None:
        raise ValueError("Service URL is None")
    if cache:
        with DBClient(model=model_id) as db:
            db["url"] = str(service_url)
    return service.url


@app.get(
    "/query",
    responses={200: {"content": {"image/png": {}}}},
    response_class=Response,
)
async def query_model(
    model_id: str,
    prompt: str,
    override_model_url: Optional[str] = None,
    override_token: Optional[str] = None,
):
    if override_model_url is not None:
        model_url = override_model_url
        service_token = override_token
    else:
        with DBClient(model=model_id) as db:
            if not db.has_entries():
                raise HTTPException(
                    status_code=404, detail=f"Model with ID {model_id} not found"
                )
            model_url = db["url"]
            service_token = db["token"]
            service_id = db["service_id"]
            print(model_url, type(model_url))

        if not model_url.startswith("http"):
            logger.info(f"model_url={model_url} for {model_id}, querying for url...")
            model_url = get_service_url(service_id, model_id, cache=True)
            logger.info(f"Got {model_url} for {model_id}")

    import requests
    import urllib

    encoded_prompt = urllib.parse.urlencode({"prompt": prompt, "image_size": 512})
    print(f"Got {model_url} for {model_id}")
    print("Token: ", service_token)
    response = requests.get(
        f"{model_url}/imagine?{encoded_prompt}",
        headers={"Authorization": f"Bearer {service_token}"},
    )
    if response.status_code != 200:
        print(response.text)
        raise HTTPException(status_code=response.status_code, detail=response.text)
    return Response(
        content=response.content,
        media_type="image/png",
        status_code=response.status_code,
    )


# https://fastapi.tiangolo.com/tutorial/request-files/#multiple-file-uploads
@app.post("/train")
async def submit_train_job(
    files: list[UploadFile], captions: list[str], job_name: str = None
):
    # write files to temporary directory on local disk
    captions = [caption.strip() for caption in captions]
    captions_json = {}
    # create temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        # write files to temporary directory
        for file in files:
            with open(os.path.join(temp_dir, file.filename), "wb") as buffer:
                buffer.write(file.file.read())
            captions_json[file.filename] = captions.pop(0)

        with open(os.path.join(temp_dir, "captions.json"), "w") as buffer:
            import json

            json.dump(captions_json, buffer)

        anyscale_job = submit_anyscale_job(files, temp_dir, job_name)
        return {
            "message": "Job submitted successfully!",
            "job_id": anyscale_job.id,
            "job_name": anyscale_job.name,
        }


@app.get("/train_status/{job_id}")
async def get_train_status(job_id: str):
    sdk = AnyscaleSDK()
    job = sdk.get_job(job_id).result
    return {
        "message": "Job status retrieved successfully!",
        "job_id": job_id,
        "job_name": job.name,
        "status": job.status,
    }
