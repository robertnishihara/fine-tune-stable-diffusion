# Got this from: https://github.com/anyscale/docs_examples/blob/main/stable-diffusion/app.py
import time
import os
from io import BytesIO

from ray import serve
from fastapi import FastAPI
from fastapi.responses import Response
import logging
app = FastAPI()

logger = logging.getLogger("ray.serve")

def download_file_from_s3(tar_file_s3_path: str, target_path: str) -> str:
    logger.info(f"Starting s3 download: {tar_file_s3_path}")
    # Example tar_file_path = "s3://anyscale-temp/diffusion-demo/checkpoint-demo.tar.gz"
    import boto3
    s3 = boto3.resource('s3')
    # remove s3:// from the path
    tar_file_s3_path = tar_file_s3_path[5:]

    bucket_name = tar_file_s3_path.split('/')[0]
    tar_file_path = "/".join(tar_file_s3_path.split('/')[1:])
    s3.Bucket(bucket_name).download_file(tar_file_path, target_path)
    logger.info(f"Downloaded s3 path to local path: {tar_file_s3_path} -> {target_path}")
    logger.info(f"Print listdir {os.listdir(os.path.dirname(target_path))}")
    return target_path

def extract_file_from_tar(tar_file_path: str, target_path: str) -> str:
    logger.info(f"Starting tar extraction: {tar_file_path}")
    # Returns absolute path to the extracted folder
    import tarfile
    tar = tarfile.open(tar_file_path, "r:gz")
    tar.extractall(target_path)
    tar.close()
    logger.info(f"Extracted tar: {tar_file_path} -> {target_path}")
    logger.info(f"Print listdir {os.listdir(os.path.dirname(target_path))}")
    return target_path


@serve.deployment(num_replicas=1, route_prefix="/")
@serve.ingress(app)
class APIIngress:
    def __init__(self, diffusion_model_handle) -> None:
        self.handle = diffusion_model_handle

    @app.get(
        "/imagine",
        responses={200: {"content": {"image/png": {}}}},
        response_class=Response,
    )
    async def generate(self, prompt: str, img_size: int = 512):
        assert len(prompt), "prompt parameter cannot be empty"

        image = await (await self.handle.generate.remote(prompt, img_size=img_size))

        file_stream = BytesIO()
        image.save(file_stream, "PNG")
        return Response(content=file_stream.getvalue(), media_type="image/png")


@serve.deployment(
    ray_actor_options={"num_gpus": 1},
    autoscaling_config={"min_replicas": 0, "max_replicas": 2},
)
class StableDiffusionV3:
    def __init__(self):
        s3_path = os.environ.get("MODEL_PATH")
        if s3_path is None:
            raise RuntimeError("No S3 path found.")
        logger.info(f"Fetched s3 path: {s3_path}")
        download_path = os.path.abspath(
            os.path.join(os.getcwd(), s3_path.split('/')[-1]))
        download_file_from_s3(s3_path, target_path=download_path)

        # model_path = os.path.abspath(download_path.split(".")[0])
        extraction_path = os.path.abspath("./decent")
        extract_file_from_tar(download_path, target_path=extraction_path)
        model_path = os.path.join(extraction_path, "sd-model-finetuned")
        logger.info(f"Loading model into diffusers: {model_path}")

        import torch
        from diffusers import StableDiffusionPipeline
        self.pipe = StableDiffusionPipeline.from_pretrained(
            model_path, torch_dtype=torch.float16)
        self.pipe.to("cuda")


    def generate(self, prompt: str, img_size: int = 512):
        assert len(prompt), "prompt parameter cannot be empty"
        logger.info(f"Received prompt: {prompt}")
        start = time.time()
        image = self.pipe(
            prompt, height=img_size, width=img_size).images[0]
        logger.info(f"Generation took: {time.time() - start} s")
        return image

@serve.deployment
class SimpleDiffusion:
    def __init__(self) -> None:
        logger.info("Hello")

    def generate(self, prompt: str, img_size: int = 512):
        # get a random image from the internet
        import requests
        from PIL import Image
        from io import BytesIO
        logger.info(f"Generating image for {prompt}")
        response = requests.get("https://picsum.photos/200")
        image = Image.open(BytesIO(response.content))
        return image

# print("FORCING TEST MODE")
if os.environ.get("TEST_MODE") == "1":
    entrypoint = APIIngress.bind(SimpleDiffusion.bind())
else:
    entrypoint = APIIngress.bind(StableDiffusionV3.bind())