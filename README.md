# fine-tune-stable-diffusion

Overall application architecture:
 - `train.py` - trains a model on a cluster
 - `load_checkpoint.py` - loads a checkpoint
 - `server.py` - runs a server that coordinates the serving and training
 - `serve_model.py` - creates a ray serve deployment using the checkpoint

 TODOs:
  - [ ] Make sure the model is actually being trained
  - [ ] Make sure I can query the model properly
    - [x] Querying a fake model locally and rendering the right output
    - [x] Querying a fake model remotely and rendering the right output
    - [ ] Querying a real model remotely and rendering the right output
    - [ ] Querying any model remotely and rendering the right output
  - [ ] Deploying models
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