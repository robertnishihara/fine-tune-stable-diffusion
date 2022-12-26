# fine-tune-stable-diffusion


To run `train.py`
- Use this cluster environment https://console.anyscale-staging.com/o/anyscale-internal/configurations/app-config-details/bld_hu28yb4llwb66fxh3cd9dzh9ty
- Use a g5.8xlarge instance.

That will save the checkpoint to `"/home/ray/default/fine-tune-stable-diffusion/src/sd-model-finetuned"`.

You can then load the checkpoint and run inference with `python load_checkpoint.py`