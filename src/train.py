# Copied from https://github.com/huggingface/diffusers/blob/main/examples/text_to_image/train_text_to_image.py

import math
import os
import random
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from datasets import load_dataset
from diffusers import AutoencoderKL, DDPMScheduler, StableDiffusionPipeline, UNet2DConditionModel
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version
from diffusers.utils.import_utils import is_xformers_available
from huggingface_hub import HfFolder, Repository, whoami
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer

# SOME THINGS THAT NEED CUSTOMIZATION
image_column = "image" #args.image_column
caption_column = "text" #args.caption_column
resolution = 512
# OPTIONALLY CONFIGURE
center_crop = True
random_flip = True
gradient_accumulation_steps = 1
mixed_precision = None
report_to = None #"tensorboard"
train_batch_size = 1 #16
num_train_epochs = 2
lr_scheduler = "constant"
lr_warmup_steps = 500
use_ema = False  # To use this, must copy EMAModel class from https://github.com/huggingface/diffusers/blob/main/examples/text_to_image/train_text_to_image.py#L266
resume_from_checkpoint = False
max_grad_norm = 1.0
max_train_steps = 5

logger = get_logger(__name__)

output_dir = "sd-model-finetuned"
logging_dir = os.path.join(output_dir, "logs")
model_id = "stabilityai/stable-diffusion-2"


tokenizer = CLIPTokenizer.from_pretrained(
    model_id, #args.pretrained_model_name_or_path,
    subfolder="tokenizer",
    revision=None, #revision=args.revision
)

text_encoder = CLIPTextModel.from_pretrained(
    model_id, #args.pretrained_model_name_or_path,
    subfolder="text_encoder",
    revision=None, #args.revision,
)

vae = AutoencoderKL.from_pretrained(
    model_id, #args.pretrained_model_name_or_path,
    subfolder="vae",
    revision=None, #args.revision,
)
unet = UNet2DConditionModel.from_pretrained(
    model_id, #args.pretrained_model_name_or_path,
    subfolder="unet",
    revision=None, #args.revision,
)

vae.requires_grad_(False)
text_encoder.requires_grad_(False)

#optimizer_cls = bnb.optim.AdamW8bit
optimizer_cls = torch.optim.AdamW
optimizer = optimizer_cls(
    unet.parameters(),
    lr=1e-4, #args.learning_rate,
    betas=(0.9, 0.999), #(args.adam_beta1, args.adam_beta2),
    weight_decay=1e-2, #args.adam_weight_decay,
    eps=1e-08, #args.adam_epsilon,
)
noise_scheduler = DDPMScheduler.from_pretrained(
    model_id, #args.pretrained_model_name_or_path,
    subfolder="scheduler"
)


# Data
dataset_name = "lambdalabs/pokemon-blip-captions"
dataset = load_dataset(
    dataset_name, #args.dataset_name,
    None, #args.dataset_config_name,
    cache_dir=None, #args.cache_dir,
)
dataset_name_mapping = {
    "lambdalabs/pokemon-blip-captions": ("image", "text"),
}

# Preprocessing the datasets.
# We need to tokenize inputs and targets.
column_names = dataset["train"].column_names

# 6. Get the column names for input/target.
# THIS SECTION NEEDS CUSTOMIZATION DEPENDING ON THE DATASET
dataset_columns = dataset_name_mapping.get(dataset_name, None)
assert image_column in column_names
assert caption_column in column_names

# Preprocessing the datasets.
# We need to tokenize input captions and transform the images.
def tokenize_captions(examples, is_train=True):
    captions = []
    for caption in examples[caption_column]:
        if isinstance(caption, str):
            captions.append(caption)
        elif isinstance(caption, (list, np.ndarray)):
            # take a random caption if there are multiple
            captions.append(random.choice(caption) if is_train else caption[0])
        else:
            raise ValueError(
                f"Caption column `{caption_column}` should contain either strings or lists of strings."
            )
    inputs = tokenizer(captions, max_length=tokenizer.model_max_length, padding="do_not_pad", truncation=True)
    input_ids = inputs.input_ids
    return input_ids

train_transforms = transforms.Compose(
    [
        transforms.Resize((resolution, resolution), interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(resolution) if center_crop else transforms.RandomCrop(resolution),
        transforms.RandomHorizontalFlip() if random_flip else transforms.Lambda(lambda x: x),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ]
)

def preprocess_train(examples):
    images = [image.convert("RGB") for image in examples[image_column]]
    examples["pixel_values"] = [train_transforms(image) for image in images]
    examples["input_ids"] = tokenize_captions(examples)

    return examples

accelerator = Accelerator(
    gradient_accumulation_steps=gradient_accumulation_steps,
    mixed_precision=mixed_precision,
    # log_with=report_to,
    logging_dir=logging_dir,
)

with accelerator.main_process_first():
    # Set the training transforms
    train_dataset = dataset["train"].with_transform(preprocess_train)

def collate_fn(examples):
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
    input_ids = [example["input_ids"] for example in examples]
    padded_tokens = tokenizer.pad({"input_ids": input_ids}, padding=True, return_tensors="pt")
    return {
        "pixel_values": pixel_values,
        "input_ids": padded_tokens.input_ids,
        "attention_mask": padded_tokens.attention_mask,
    }

train_dataloader = torch.utils.data.DataLoader(
    train_dataset, shuffle=True, collate_fn=collate_fn, batch_size=train_batch_size
)

num_update_steps_per_epoch = math.ceil(len(train_dataloader) / gradient_accumulation_steps)

lr_scheduler = get_scheduler(
    lr_scheduler,
    optimizer=optimizer,
    num_warmup_steps=lr_warmup_steps * gradient_accumulation_steps,
    num_training_steps=max_train_steps * gradient_accumulation_steps,
)

unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
    unet, optimizer, train_dataloader, lr_scheduler
)
accelerator.register_for_checkpointing(lr_scheduler)

weight_dtype = torch.float32
if accelerator.mixed_precision == "fp16":
    weight_dtype = torch.float16
elif accelerator.mixed_precision == "bf16":
    weight_dtype = torch.bfloat16

# Move text_encode and vae to gpu.
# For mixed precision training we cast the text_encoder and vae weights to half-precision
# as these models are only used for inference, keeping weights in full precision is not required.
text_encoder.to(accelerator.device, dtype=weight_dtype)
vae.to(accelerator.device, dtype=weight_dtype)

# if use_ema:
#     ema_unet = EMAModel(unet.parameters())

# We need to recalculate our total training steps as the size of the training dataloader may have changed.
num_update_steps_per_epoch = math.ceil(len(train_dataloader) / gradient_accumulation_steps)
# Afterwards we recalculate our number of training epochs
num_train_epochs = math.ceil(max_train_steps / num_update_steps_per_epoch)

# We need to initialize the trackers we use, and also store our configuration.
# The trackers initializes automatically on the main process.
#if accelerator.is_main_process:
#    accelerator.init_trackers("text2image-fine-tune", config=vars(args))
# TODO: THE FACT THAT I COMMENTED OUT THE ABOVE MAY CAUSE PROBLEMS

# Train!
total_batch_size = train_batch_size * accelerator.num_processes * gradient_accumulation_steps

logger.info("***** Running training *****")
logger.info(f"  Num examples = {len(train_dataset)}")
logger.info(f"  Num Epochs = {num_train_epochs}")
logger.info(f"  Instantaneous batch size per device = {train_batch_size}")
logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
logger.info(f"  Gradient Accumulation steps = {gradient_accumulation_steps}")
logger.info(f"  Total optimization steps = {max_train_steps}")
global_step = 0
first_epoch = 0

# TODO: RESUME FROM CHECKPOINT HERE IF YOU WANT
if resume_from_checkpoint:
    raise NotImplementedError

# Only show the progress bar once on each machine.
progress_bar = tqdm(range(global_step, max_train_steps), disable=not accelerator.is_local_main_process)
progress_bar.set_description("Steps")

for epoch in range(first_epoch, num_train_epochs):
    unet.train()
    train_loss = 0.0
    for step, batch in enumerate(train_dataloader):
        # Skip steps until we reach the resumed step
        # if resume_from_checkpoint and epoch == first_epoch and step < resume_step:
        #     if step % gradient_accumulation_steps == 0:
        #         progress_bar.update(1)
        #     continue

        with accelerator.accumulate(unet):
            # Convert images to latent space
            latents = vae.encode(batch["pixel_values"].to(weight_dtype)).latent_dist.sample()
            latents = latents * 0.18215

            # Sample noise that we'll add to the latents
            noise = torch.randn_like(latents)
            bsz = latents.shape[0]
            # Sample a random timestep for each image
            timesteps = torch.randint(0, noise_scheduler.num_train_timesteps, (bsz,), device=latents.device)
            timesteps = timesteps.long()

            # Add noise to the latents according to the noise magnitude at each timestep
            # (this is the forward diffusion process)
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

            # Get the text embedding for conditioning
            encoder_hidden_states = text_encoder(batch["input_ids"])[0]

            # Get the target for loss depending on the prediction type
            if noise_scheduler.config.prediction_type == "epsilon":
                target = noise
            elif noise_scheduler.config.prediction_type == "v_prediction":
                target = noise_scheduler.get_velocity(latents, noise, timesteps)
            else:
                raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

            # Predict the noise residual and compute loss
            model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample
            loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

            # Gather the losses across all processes for logging (if we use distributed training).
            avg_loss = accelerator.gather(loss.repeat(train_batch_size)).mean()
            train_loss += avg_loss.item() / gradient_accumulation_steps

            # Backpropagate
            accelerator.backward(loss)
            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(unet.parameters(), max_grad_norm)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

        # Checks if the accelerator has performed an optimization step behind the scenes
        if accelerator.sync_gradients:
            if use_ema:
                ema_unet.step(unet.parameters())
            progress_bar.update(1)
            global_step += 1
            # accelerator.log({"train_loss": train_loss}, step=global_step)
            print("train_loss", train_loss)
            train_loss = 0.0

            # if global_step % args.checkpointing_steps == 0:
            #     if accelerator.is_main_process:
            #         save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
            #         accelerator.save_state(save_path)
            #         logger.info(f"Saved state to {save_path}")

        logs = {"step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
        progress_bar.set_postfix(**logs)

        if global_step >= max_train_steps:
            break


"""
if accelerator.is_main_process:
    save_path = os.path.join(output_dir, f"checkpoint-final")
    accelerator.save_state(save_path)
    logger.info(f"Saved state to {save_path}")
"""

# Create the pipeline using the trained modules and save it.
accelerator.wait_for_everyone()
if accelerator.is_main_process:
    unet = accelerator.unwrap_model(unet)
    # if args.use_ema:
    #     ema_unet.copy_to(unet.parameters())

    pipeline = StableDiffusionPipeline.from_pretrained(
        model_id, #args.pretrained_model_name_or_path,
        text_encoder=text_encoder,
        vae=vae,
        unet=unet,
        revision=None, #args.revision,
    )
    pipeline.save_pretrained(output_dir)

    # if args.push_to_hub:
    #     repo.push_to_hub(commit_message="End of training", blocking=False, auto_lfs_prune=True)

accelerator.end_training()



"""
pipeline = StableDiffusionPipeline.from_pretrained(
    model_id, #args.pretrained_model_name_or_path,
    text_encoder=text_encoder,
    vae=vae,
    unet=unet,
    revision=None, #args.revision,
)
pipeline.save_pretrained(output_dir)
"""







"""
scheduler = EulerDiscreteScheduler.from_pretrained(
    model_id, subfolder="scheduler"
)
self.pipe = StableDiffusionPipeline.from_pretrained(
    model_id, scheduler=scheduler, revision="fp16", torch_dtype=torch.float16
)
self.pipe = self.pipe.to("cuda")
"""