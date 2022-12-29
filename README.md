# fine-tune-stable-diffusion

Overall application architecture:
 - `train.py` - trains a model on a cluster
 - `load_checkpoint.py` - loads a checkpoint
 - `server.py` - runs a server that coordinates the serving and training
 - `serve_model.py` - creates a ray serve deployment using the checkpoint

How to run the application:
 - go/aws to get your aws credentials -- make sure to set these so that you (and the remote jobs) can access S3
 - `uvicorn server:app --reload` to run the server
 - then go to `localhost:8000/docs` on your browser to get an interface to test the APIs



 TODOs:
  - [ ] Make sure the model is actually being trained
    - [ ] We should consider migrating to Dreambooth instead to get a better personalization demo.
  - [x] Make sure I can query the model properly
    - [x] Querying a fake model locally and rendering the right output
    - [x] Querying a fake model remotely and rendering the right output
    - [x] Querying a real model remotely and rendering the right output
    - [x] Querying any model remotely and rendering the right output
  - [ ] Miscellanous
    - [ ] Right now there's tar and zip; we should just standardize
    - [ ] Move the service.py out of service/ so that we can use the utils
    - [ ] Right now there's no good way of allowing this service to connect to s3 (for training upload)



To run `train.py`
- Use this cluster environment https://console.anyscale-staging.com/o/anyscale-internal/configurations/app-config-details/bld_hu28yb4llwb66fxh3cd9dzh9ty
- Use a g5.8xlarge instance.

That will save the checkpoint to `"/home/ray/default/fine-tune-stable-diffusion/src/sd-model-finetuned"`.

You can then load the checkpoint and run inference with `python load_checkpoint.py`


### Notes
- Stored a checkpoint file with `aws s3 cp src/checkpoint.tar.gz s3://anyscale-temp/diffusion-demo/`, will need to fetch it and untar it and then you should ideally be able to load it with `load_checkpoint.py` pointed to the appropriate location. E.g.,
    - `aws s3 cp s3://anyscale-temp/diffusion-demo/checkpoint-demo.tar.gz .`
    - `tar xvf checkpoint-demo.tar.gz`