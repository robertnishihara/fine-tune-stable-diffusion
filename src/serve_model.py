# Got this from: https://github.com/anyscale/docs_examples/blob/main/stable-diffusion/app.py
import time
import os
from io import BytesIO

from ray import serve
from fastapi import FastAPI
from fastapi.responses import Response
import logging

from s3_utils import download_file_from_s3, untar_dir

app = FastAPI()

logger = logging.getLogger("ray.serve")


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
            os.path.join(os.getcwd(), s3_path.split("/")[-1])
        )
        download_file_from_s3(s3_path, target_path=download_path)

        # model_path = os.path.abspath(download_path.split(".")[0])
        extraction_path = os.path.abspath("./decent")
        untar_dir(download_path, target_path=extraction_path)
        model_path = os.path.join(extraction_path, "sd-model-finetuned")
        logger.info(f"Loading model into diffusers: {model_path}")

        import torch
        from diffusers import StableDiffusionPipeline

        self.pipe = StableDiffusionPipeline.from_pretrained(
            model_path, torch_dtype=torch.float16
        )
        self.pipe.to("cuda")

    def generate(self, prompt: str, img_size: int = 512):
        assert len(prompt), "prompt parameter cannot be empty"
        logger.info(f"Received prompt: {prompt}")
        start = time.time()
        image = self.pipe(prompt, height=img_size, width=img_size).images[0]
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
