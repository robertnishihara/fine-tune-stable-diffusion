"""Simple stable diffusion app.

How to run:
    $ uvicorn server:app --reload

How to test:
"""
import os
# dumb basic storage
import dbm
from fastapi import FastAPI, UploadFile, HTTPException
from fastapi.responses import Response

from anyscale.sdk.anyscale_client.models import (
    CreateProductionJob, ClusterComputeConfig, ComputeNodeType,
    CreateProductionService, CreateProductionJobConfig
)
from anyscale import AnyscaleSDK
import logging

logger = logging.getLogger(__name__)

app = FastAPI()

storage_path = "storage"  # db automatically appended

def submit_anyscale_job(files, file_directory, job_name):
    import yaml

    sdk = AnyscaleSDK()
    runtime_env = {
        "working_dir": file_directory,
        "upload_path": "s3://anyscale-temp/diffusion-demo/"}
    job_config = {
        # IDs can be found on Anyscale Console under Configurations.
        # The IDs below are examples and should be replaced with your own IDs.
        # 'compute_config_id': 'cpt_U8RCfD7Wr1vCD4iqGi4cBbj1',
        "runtime_env": runtime_env,
        # The compute config can also specified as a one-off instead:
        'compute_config': ClusterComputeConfig(
             cloud_id="cld_V1U8Jk3ZgEQQbc7zkeBq24iX",
             region="us-west-2",
             head_node_type=ComputeNodeType(
                  name="head",
                  instance_type="m5.large",
            ),
             worker_node_types=[],
        ),
        # The id of the cluster env build
        'build_id': 'bld_1277XIinoJmiM8Z3gNdcHN',
        # 'runtime_env': {
        #     'working_dir': 's3://my_bucket/my_job_files.zip'
        # },
        'entrypoint': f'python train.py --image-dir {file_directory}',
        'max_retries': 3
        }


    job = sdk.create_job(CreateProductionJob(
        name=job_name,
        description="Stable diffusion training job",
        # project_id can be found in the URL by navigating to the project in Anyscale Console
        project_id='prj_7S7Os7XBvO6vdiVC1J0lgj',
        config=job_config
    ))
    return job

def submit_service(model_id, local=False):
    from types import SimpleNamespace
    if local == True:
        import subprocess
        subprocess.run(["python", "service/serve_model.py", "--test"])
        result = SimpleNamespace()
        result.url = "http://localhost:8001"
    else:
        sdk = AnyscaleSDK()
        current_dir = os.path.dirname(os.path.abspath(__file__))
        response = sdk.apply_service(
            create_production_service=CreateProductionService(
                name=f"stable-diffusion-{model_id}",
                description="Stable diffusion services",
                # project_id can be found in the URL
                # https://console.anyscale-staging.com/o/anyscale-internal/projects/prj_j2bynt35acxvgtg6riahpzqk
                project_id='prj_j2bynt35acxvgtg6riahpzqk',
                healthcheck_url="/healthcheck",
                config=dict(
                    # https://console.anyscale-staging.com/o/anyscale-internal/configurations/cluster-computes/cpt_v1hkxu5rd61ql5nd268fen83t7
                    compute_config_id="cpt_v1hkxu5rd61ql5nd268fen83t7",
                    # https://console.anyscale-staging.com/o/anyscale-internal/configurations/app-config-details/bld_hu28yb4llwb66fxh3cd9dzh9ty
                    build_id="bld_hu28yb4llwb66fxh3cd9dzh9ty",
                    runtime_env=dict(
                        working_dir="https://github.com/robertnishihara/fine-tune-stable-diffusion/archive/refs/heads/main.zip"
                    ),
                    entrypoint="cd src/service && serve run --non-blocking serve_model:entrypoint",
                    access="public"
                )
            )
        )
        result = response.result
        logger.info(result.url)
    return result

@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.post("/deploy/{model_id}")
async def deploy(model_id: str):
    # TODO: deploy model to Anyscale
    local = model_id == "TEST"
    result = submit_service(model_id, local=local)
    with dbm.open(storage_path, "c") as db:
        db[model_id] = str(result.url)
        db["{model_id}_token"] = get_service_token(result.id)
    return {
        "message": "Deployed model successfully!",
        "model_id": model_id,
        "model_url": result.url
    }

def get_service_token(service_id):
    from anyscale import AnyscaleSDK
    sdk = AnyscaleSDK()
    service = sdk.get_service(service_id).result
    return service.token

@app.get(
    "/query",
    responses={200: {"content": {"image/png": {}}}},
    response_class=Response,
)
async def query_model(model_id: str, prompt: str):
    with dbm.open(storage_path, "c") as db:
        if model_id not in db:
            raise HTTPException(
                status_code=404,
                detail=f"Model with ID {model_id} not found"
            )
        model_url = db[model_id].decode() # to str
        service_token = db[f"{model_id}_token"].decode() # to str
    import requests
    import urllib

    encoded_prompt = urllib.parse.urlencode({
        "prompt": prompt,
        "image_size": 512})
    logger.info(f"Got {model_url} for {model_id}")
    response = requests.get(
        f"{model_url}/imagine?{encoded_prompt}",
        headers={"Authorization": f"Bearer {service_token}"})

    return Response(
        content=response.content,
        media_type="image/png",
        status_code=response.status_code)


# https://fastapi.tiangolo.com/tutorial/request-files/#multiple-file-uploads
@app.post("/train")
async def create_upload_files(
        files: list[UploadFile], captions: list[str],
        job_name: str = None):
    # write files to temporary directory on local disk
    import os
    import tempfile

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