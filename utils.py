import os
import cv2
import yaml
import copy
import pygame
import numpy as np
from PIL import Image
from fontTools.ttLib import TTFont

import torch
import torchvision.transforms as transforms

def save_args_to_yaml(args, output_file):
    # Convert args namespace to a dictionary
    args_dict = vars(args)

    # Write the dictionary to a YAML file
    with open(output_file, 'w') as yaml_file:
        yaml.dump(args_dict, yaml_file, default_flow_style=False)


def save_single_image(save_dir, image):
    i = 0
    while os.path.exists(f"{save_dir}/out_single_{i}.jpg"):
        i += 1
    save_path = f"{save_dir}/out_single_{i}.jpg"
    image.save(save_path)


def save_image_with_content_style(save_dir, image, content_image_pil, content_image_path, style_image_path, resolution):
    h,w = 64, 256 # fix this
    new_image = Image.new('RGB', (w*3, h))
    if content_image_pil is not None:
        content_image = content_image_pil
    else:
        content_image = Image.open(content_image_path).convert("RGB").resize((w, h), Image.BILINEAR)
    style_image = Image.open(style_image_path).convert("RGB").resize((w, h), Image.BILINEAR)

    new_image.paste(content_image, (0, 0))
    new_image.paste(style_image, (w, 0))
    new_image.paste(image, (w*2, 0))
    i = 0
    while os.path.exists(f"{save_dir}/out_with_cs_{i}.jpg"):
        i += 1
    save_path = f"{save_dir}/out_with_cs_{i}.jpg"
    new_image.save(save_path)


def x0_from_epsilon(scheduler, noise_pred, x_t, timesteps):
    """Return the x_0 from epsilon
    """
    batch_size = noise_pred.shape[0]
    for i in range(batch_size):
        noise_pred_i = noise_pred[i]
        noise_pred_i = noise_pred_i[None, :]
        t = timesteps[i]
        x_t_i = x_t[i]
        x_t_i = x_t_i[None, :]

        pred_original_sample_i = scheduler.step(
            model_output=noise_pred_i,
            timestep=t,
            sample=x_t_i,
            # predict_epsilon=True,
            generator=None,
            return_dict=True,
        ).pred_original_sample
        if i == 0:
            pred_original_sample = pred_original_sample_i
        else:
            pred_original_sample = torch.cat((pred_original_sample, pred_original_sample_i), dim=0)

    return pred_original_sample


def reNormalize_img(pred_original_sample):
    pred_original_sample = (pred_original_sample / 2 + 0.5).clamp(0, 1)
    
    return pred_original_sample


def normalize_mean_std(image):
    transforms_norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    image = transforms_norm(image)

    return image


def is_char_in_font(font_path, char):
    TTFont_font = TTFont(font_path)
    cmap = TTFont_font['cmap']
    for subtable in cmap.tables:
        if ord(char) in subtable.cmap:
            return True
    return False


def load_ttf(ttf_path, fsize=128):
    pygame.init()

    font = pygame.freetype.Font(ttf_path, size=fsize)
    return font


def ttf2im(font, char, height=128, width=128):
    try:
        surface, _ = font.render(char)
    except:
        print("No glyph for char {}".format(char))
        return
    bg = np.full((height, width), 255)
    imo = pygame.surfarray.pixels_alpha(surface).transpose(1, 0)
    imo = 255 - np.array(Image.fromarray(imo))
    im = copy.deepcopy(bg)
    h, w = imo.shape[:2]
    if h > height:
        h, w = height, round(w * height / h)
        imo = cv2.resize(imo, (w, h))
    if w > width:
        h, w = round(h * width / w), width
        imo = cv2.resize(imo, (w, h))
    x, y = round((width - w) / 2), round((height - h) / 2)
    im[y:h+y, x:x+w] = imo
    pil_im = Image.fromarray(im.astype('uint8')).convert('RGB')
    
    return pil_im
