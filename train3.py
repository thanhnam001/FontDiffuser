import os
import math
import time
import logging
from tqdm.auto import tqdm

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torchvision import transforms

from diffusers.optimization import get_scheduler

from dataset.font_dataset import FontDataset
from dataset.collate_fn import CollateFN
from configs.fontdiffuser import get_parser
from src import (FontDiffuserModel,
                 ContentPerceptualLoss,
                 build_unet,
                 build_style_encoder,
                 build_content_encoder,
                 build_label_encoder,
                 build_ddpm_scheduler,
                 build_scr)
from utils import (save_args_to_yaml,
                   x0_from_epsilon, 
                   reNormalize_img, 
                   normalize_mean_std)

def get_args():
    parser = get_parser()
    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank
    style_image_size = args.style_image_size
    content_image_size = args.content_image_size
    args.style_image_size = (64, 256)#(style_image_size, style_image_size)
    args.content_image_size = (64, 256)#(content_image_size, content_image_size)

    return args


def setup_logging(output_dir, logging_dir):
    os.makedirs(output_dir, exist_ok=True)
    logging.basicConfig(
        filename=f"{output_dir}/fontdiffuser_training.log",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO)
    return logging_dir


def setup_distributed_training(args):
    dist.init_process_group(backend='nccl')
    torch.cuda.set_device(args.local_rank)
    device = torch.device('cuda', args.local_rank)
    return device


def main():

    args = get_args()
    logging_dir = setup_logging(args.output_dir, args.logging_dir)
    device = setup_distributed_training(args)

    # Set training seed
    if args.seed is not None:
        torch.manual_seed(args.seed)

    # Load model and noise_scheduler
    unet = build_unet(args=args).to(device)
    style_encoder = build_style_encoder(args=args).to(device)
    content_encoder = build_content_encoder(args=args).to(device)
    noise_scheduler = build_ddpm_scheduler(args)
    label_encoder = build_label_encoder(args)
    if args.phase_2:
        unet.load_state_dict(torch.load(f"{args.phase_1_ckpt_dir}/unet.pth", map_location=device))
        style_encoder.load_state_dict(torch.load(f"{args.phase_1_ckpt_dir}/style_encoder.pth", map_location=device))
        content_encoder.load_state_dict(torch.load(f"{args.phase_1_ckpt_dir}/content_encoder.pth", map_location=device))

    model = FontDiffuserModel(
        unet=unet,
        style_encoder=style_encoder,
        label_encoder=label_encoder,
        content_encoder=content_encoder).to(device)

    # Wrap the model with DDP
    model = DDP(model, device_ids=[args.local_rank], output_device=args.local_rank)

    # Build content perceptual Loss
    perceptual_loss = ContentPerceptualLoss().to(device)

    # Load SCR module for supervision
    if args.phase_2:
        scr = build_scr(args=args).to(device)
        scr.load_state_dict(torch.load(args.scr_ckpt_path, map_location=device))
        scr.requires_grad_(False)

    # Load the datasets
    content_transforms = transforms.Compose(
        [transforms.Resize(args.content_image_size, interpolation=transforms.InterpolationMode.BILINEAR),
         transforms.ToTensor(),
         transforms.Normalize([0.5], [0.5])])
    style_transforms = transforms.Compose(
        [transforms.Resize(args.style_image_size, interpolation=transforms.InterpolationMode.BILINEAR),
         transforms.ToTensor(),
         transforms.Normalize([0.5], [0.5])])
    target_transforms = transforms.Compose(
        [transforms.Resize((64, 256), interpolation=transforms.InterpolationMode.BILINEAR),
         transforms.ToTensor(),
         transforms.Normalize([0.5], [0.5])])
    train_font_dataset = FontDataset(
        args=args,
        phase='train',
        transforms=[
            content_transforms,
            style_transforms,
            target_transforms],
        scr=args.phase_2)
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_font_dataset)
    train_dataloader = torch.utils.data.DataLoader(
        train_font_dataset, sampler=train_sampler, batch_size=args.train_batch_size, collate_fn=CollateFN())

    # Build optimizer and learning rate
    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * dist.get_world_size())
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon)
    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.max_train_steps * args.gradient_accumulation_steps)

    # Prepare for training
    model.train()
    global_step = 0
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)
    progress_bar = tqdm(range(args.max_train_steps), disable=args.local_rank != 0)
    progress_bar.set_description("Steps")

    for epoch in range(num_train_epochs):
        train_dataloader.sampler.set_epoch(epoch)
        train_loss = 0.0
        for step, samples in enumerate(train_dataloader):
            content_images = samples["content_image"].to(device)
            style_images = samples["style_image"].to(device)
            target_images = samples["target_image"].to(device)
            nonorm_target_images = samples["nonorm_target_image"].to(device)

            optimizer.zero_grad()
            noise = torch.randn_like(target_images)
            bsz = target_images.shape[0]
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=target_images.device).long()
            noisy_target_images = noise_scheduler.add_noise(target_images, noise, timesteps)

            context_mask = torch.bernoulli(torch.zeros(bsz, device=device) + args.drop_prob)
            # content_images[context_mask == 1] = 1
            style_images[context_mask == 1] = 1

            noise_pred, offset_out_sum = model(
                x_t=noisy_target_images,
                timesteps=timesteps,
                style_images=style_images,
                content_images=content_images,
                content_encoder_downsample_size=args.content_encoder_downsample_size)
            diff_loss = F.mse_loss(noise_pred.float(), noise.float(), reduction="mean")
            offset_loss = offset_out_sum / 2

            pred_original_sample_norm = x0_from_epsilon(
                scheduler=noise_scheduler,
                noise_pred=noise_pred,
                x_t=noisy_target_images,
                timesteps=timesteps)
            pred_original_sample = reNormalize_img(pred_original_sample_norm)
            norm_pred_ori = normalize_mean_std(pred_original_sample)
            norm_target_ori = normalize_mean_std(nonorm_target_images)
            percep_loss = perceptual_loss.calculate_loss(
                generated_images=norm_pred_ori,
                target_images=norm_target_ori,
                device=target_images.device)

            loss = diff_loss + \
                   args.perceptual_coefficient * percep_loss + \
                   args.offset_coefficient * offset_loss

            if args.phase_2:
                neg_images = samples["neg_images"].to(device)
                sample_style_embeddings, pos_style_embeddings, neg_style_embeddings = scr(
                    pred_original_sample_norm,
                    target_images,
                    neg_images,
                    nce_layers=args.nce_layers)
                sc_loss = scr.calculate_nce_loss(
                    sample_s=sample_style_embeddings,
                    pos_s=pos_style_embeddings,
                    neg_s=neg_style_embeddings)
                loss += args.sc_coefficient * sc_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            lr_scheduler.step()

            train_loss += loss.item() / args.gradient_accumulation_steps

            if step % args.log_interval == 0 and args.local_rank == 0:
                logging.info(f"Epoch [{epoch + 1}/{num_train_epochs}], Step [{step + 1}/{len(train_dataloader)}], "
                             f"Loss: {train_loss:.4f}, LR: {lr_scheduler.get_last_lr()[0]:.6f}")

            progress_bar.update(1)
            global_step += 1
            train_loss = 0.0

            if global_step % args.ckpt_interval == 0 and args.local_rank == 0:
                save_dir = f"{args.output_dir}/global_step_{global_step}"
                os.makedirs(save_dir, exist_ok=True)
                torch.save(model.module.unet.state_dict(), f"{save_dir}/unet.pth")
                torch.save(model.module.style_encoder.state_dict(), f"{save_dir}/style_encoder.pth")
                torch.save(model.module.content_encoder.state_dict(), f"{save_dir}/content_encoder.pth")
                torch.save(model.module.label_encoder.state_dict(), f'{save_dir}/label_encoder.pth')
                torch.save(model.state_dict(), f"{save_dir}/total_model.pth")
                logging.info(f"Saved checkpoint at global step {global_step}")

            if global_step >= args.max_train_steps:
                break

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
