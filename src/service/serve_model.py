# Got this from: https://github.com/anyscale/docs_examples/blob/main/stable-diffusion/app.py

from io import BytesIO

from ray import serve
from fastapi import FastAPI
from fastapi.responses import Response

app = FastAPI()

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
class StableDiffusionV2:
    def __init__(self):
        import torch
        from diffusers import EulerDiscreteScheduler, StableDiffusionPipeline
        model_id = "stabilityai/stable-diffusion-2"

        scheduler = EulerDiscreteScheduler.from_pretrained(
            model_id, subfolder="scheduler"
        )
        self.pipe = StableDiffusionPipeline.from_pretrained(
            model_id, scheduler=scheduler, revision="fp16", torch_dtype=torch.float16
        )
        self.pipe = self.pipe.to("cuda")

    def generate(self, prompt: str, img_size: int = 512):
        assert len(prompt), "prompt parameter cannot be empty"

        image = self.pipe(prompt, height=img_size, width=img_size).images[0]

        return image

@serve.deployment
class SimpleDiffusion:
    def generate(self, prompt: str, img_size: int = 512):
        # get a random image from the internet
        import requests
        from PIL import Image
        from io import BytesIO

        response = requests.get("https://picsum.photos/200")
        image = Image.open(BytesIO(response.content))
        return image

print("FORCING TEST MODE")
if True:
    entrypoint = APIIngress.bind(SimpleDiffusion.bind())
else:
    entrypoint = APIIngress.bind(StableDiffusionV2.bind())