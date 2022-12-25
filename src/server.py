"""Simple stable diffusion app.

How to run:
    $ uvicorn server:app --reload

How to test:
"""


from fastapi import FastAPI, File, UploadFile

app = FastAPI()

def submit_anyscale_job(files, file_directory, job_name):
    import yaml
    from anyscale.sdk.anyscale_client.models import CreateProductionJob, ClusterComputeConfig, ComputeNodeType
    from anyscale import AnyscaleSDK

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
        f'entrypoint': 'python train.py',
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


@app.get("/")
async def root():
    return {"message": "Hello World"}

# https://fastapi.tiangolo.com/tutorial/request-files/#multiple-file-uploads
@app.post("/train")
async def create_upload_files(files: list[UploadFile], job_name: str = None):
    # write files to temporary directory on local disk
    import os
    import tempfile

    # create temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        # write files to temporary directory
        for file in files:
            with open(os.path.join(temp_dir, file.filename), "wb") as buffer:
                buffer.write(file.file.read())

        anyscale_job = submit_anyscale_job(files, temp_dir, job_name)
        return {
            "message": "Job submitted successfully!",
            "job_id": anyscale_job.id,
            "job_name": anyscale_job.name,
        }