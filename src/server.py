"""Simple stable diffusion app.

How to run:
    $ cd src/ && uvicorn server:app --reload

How to test:
"""
from typing import Optional, List, Union
import os
import uuid
import tempfile
from fastapi import FastAPI, UploadFile, HTTPException, Query
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
from db_client import TrainingDBClient, ServingDBClient
from s3_utils import write_to_s3, s3_exists, zip_dir

app = FastAPI()

s3_dir = "s3://anyscale-temp/diffusion-demo"

AWS_ACCESS_VARS = {
    "AWS_ACCESS_KEY_ID": os.environ["AWS_ACCESS_KEY_ID"],
    "AWS_SECRET_ACCESS_KEY": os.environ["AWS_SECRET_ACCESS_KEY"],
    "AWS_SESSION_TOKEN": os.environ["AWS_SESSION_TOKEN"],
}


def validate_model_path(model_path, post_training=False):
    if not model_path.startswith("s3://"):
        raise HTTPException(
            status_code=400,
            detail="Model path must be an s3 path, e.g. s3://anyscale-temp/diffusion-demo/models/model.pt",
        )
    if post_training and not s3_exists(model_path):
        raise HTTPException(
            status_code=400,
            detail=f"Model path {model_path} does not exist",
        )
    # if doesn't end with zip, fail
    if not model_path.endswith(".zip"):
        raise HTTPException(
            status_code=400,
            detail=f"Model path {model_path} must be a zip file",
        )


def submit_training_job(job_name, data_path, model_path):
    """Submitting a job to Anyscale.

    Example usage:
        submit_training_job(
            job_name="test-job",
            data_path="s3://anyscale-temp/diffusion-demo/data.zip",
            model_path="s3://anyscale-temp/diffusion-demo/model.zip",
        )
    """
    validate_model_path(model_path)
    sdk = AnyscaleSDK()
    import yaml

    with open("job.yaml", "r") as f:
        job_config = yaml.safe_load(f)

    runtime_env = job_config.get("runtime_env", {})

    if "AWS_ACCESS_KEY_ID" not in os.environ:
        raise ValueError(
            "AWS_ACCESS_KEY_ID needs to be set in the environment. "
        )
    runtime_env.update({
        "env_vars": AWS_ACCESS_VARS,
    })

    job_config.update({
        "runtime_env": runtime_env,
        # The id of the cluster env build - why can't we pass in the env name
        "build_id": "bld_hu28yb4llwb66fxh3cd9dzh9ty",
        "entrypoint": f"python src/train.py --image-data-path {data_path} --output {model_path}",
        "max_retries": 3,
    })

    response = sdk.create_job(
        CreateProductionJob(
            name=job_name,
            description="Stable diffusion training job",
            # project_id can be found in the URL by navigating to the project in Anyscale Console
            project_id="prj_j2bynt35acxvgtg6riahpzqk",
            config=job_config,
        )
    )
    return response.result


def submit_service(model_id, model_path, local=False):
    from types import SimpleNamespace

    if local:
        import subprocess

        # This is probably all broken rn
        subprocess.run("serve run src.service.serve_model:entrypoint".split(" "))
        result = SimpleNamespace()
        result.url = "http://localhost:8001"
    else:
        validate_model_path(model_path, post_training=True)
        sdk = AnyscaleSDK()
        # current_dir = os.path.dirname(os.path.abspath(__file__))
        print("Submitting service for model_path: ", model_path)
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
                        working_dir="https://github.com/robertnishihara/fine-tune-stable-diffusion/archive/refs/heads/training-fix.zip",
                        env_vars=dict(
                            RANDOM=str(uuid.uuid4()),
                            MODEL_PATH=model_path,
                            **AWS_ACCESS_VARS
                        ),
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
        with TrainingDBClient(model=model_id) as db:
            model_path = db["path"]

    result = submit_service(model_id, model_path, local=local)
    with ServingDBClient(model=model_id) as db:
        print("URL: ", result.url)
        print("service_id", result.id)
        print("service_name", str(result.name))
        print("Token: ", get_service_token(result.id))

        db["url"] = str(result.url)
        db["service_id"] = str(result.id)
        db["service_name"] = str(result.name)
        db["token"] = get_service_token(result.id)
    return {
        "message": f"Deployed model ({model_path}) successfully!",
        "model_id": model_id,
        "service_id": result.id,
        "model_url": result.url,
    }


@app.post("/terminate/{model_id}")
async def terminate(model_id: str):
    sdk = AnyscaleSDK()
    with ServingDBClient(model=model_id) as db:
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
        with ServingDBClient(model=model_id) as db:
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
        with ServingDBClient(model=model_id) as db:
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
    files: list[UploadFile],
    captions: Union[list[str], None] = Query(default=None),
    job_name: str = None
):
    from datetime import datetime

    if job_name is None:
        # time indexed
        job_name = f"train-{datetime.now().strftime('%Y-%m-%d-%H%M')}"


    print(captions)
    captions_json = {}
    # create temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        # write files to temporary directory
        temp_data_files = os.path.join(temp_dir, "data")
        os.makedirs(temp_data_files, exist_ok=True)
        for file in files:
            with open(os.path.join(temp_data_files, file.filename), "wb") as buffer:
                buffer.write(file.file.read())
            captions_json[file.filename] = captions.pop(0)

        with open(os.path.join(temp_data_files, "captions.json"), "w") as buffer:
            import json

            json.dump(captions_json, buffer)

        # zip fies in tempdir
        zipped_files = zip_dir(
            temp_data_files, os.path.join(temp_dir, "data.zip"))

        # upload zip to s3
        s3_data_key = f"{job_name}/data.zip"
        data_path = os.path.join(s3_dir, s3_data_key)
        write_to_s3(zipped_files, data_path)

    model_path = os.path.join(s3_dir, f"{job_name}/model.zip")
    anyscale_job = submit_training_job(job_name, data_path, model_path)
    print("Job submitted: ", anyscale_job.id, anyscale_job.name)


    with TrainingDBClient(model=job_name) as db:
        db["id"] = anyscale_job.id
        db["name"] = anyscale_job.name
        db["path"] = model_path

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
