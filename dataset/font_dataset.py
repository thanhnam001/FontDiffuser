import os
import random
import json
from PIL import Image
from tqdm import tqdm
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

def get_nonorm_transform(resolution):
    nonorm_transform =  transforms.Compose(
            [transforms.Resize(
                # (resolution, resolution), 
                (64, 256),
                               interpolation=transforms.InterpolationMode.BILINEAR), 
             transforms.ToTensor()])
    return nonorm_transform

vocab = {
    'IAM': 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
}

class Tokenizer:
    def __init__(self, dset_name, max_length=10) -> None:
        self.vocab = vocab[dset_name]
        # self.special_chars = {'GO_TOKEN': 0, 'END_TOKEN': 1, 'PAD_TOKEN': 2}
        self.special_chars = {'PAD_TOKEN': 0}
        self.str2idx = {char:(idx+len(self.special_chars)) for idx, char in enumerate(self.vocab)}
        self.idx2str = {v:k for k, v in self.str2idx.items()}
        self.vocab_size = len(self.vocab) + len(self.special_chars.keys())
        self.batch_max_length = max_length #+ 2
        
    def encode(self, text):       
        text = [self.str2idx[char] for char in text]
        # text = [self.special_chars['GO_TOKEN']] + text + [self.special_chars['END_TOKEN']]
        pad_len = self.batch_max_length - len(text)
        if pad_len > 0:
            text = text + [self.special_chars['PAD_TOKEN']]*pad_len
        return text

    def decode(self, ids):
        chars = list()
        for id in ids:
            if id not in self.special_chars.values():
                chars.append(self.idx2str[id])
        return ''.join(chars)
    
class FontDataset(Dataset):
    """The dataset of font generation  
    """
    def __init__(self, args, phase, transforms=None, scr=False):
        super().__init__()
        self.root = args.data_root
        self.phase = phase
        self.scr = scr
        if self.scr:
            self.num_neg = args.num_neg
        
        # Get Data path
        self.image_transcription = args.image_transcription
        self.get_path()
        self.transforms = transforms
        self.nonorm_transforms = get_nonorm_transform(args.resolution)
        self.tokenizer = Tokenizer('IAM', 16)

    def get_path(self):
        self.target_images = []
        # images with related style  
        self.style_to_images = {}
        with open(self.image_transcription) as f:
            transcription = json.load(f)
        for id, obj in tqdm(transcription.items()):
            image = obj['image']
            wid = obj['s_id']
            label = obj['label']
            img_dir = os.path.join(self.root, image)
            self.target_images.append(
                {'image': img_dir,
                 'wid': wid,
                 'label': label
                 }
            )
            if self.style_to_images.get(wid, None) == None:
                self.style_to_images[wid] = []
            self.style_to_images[wid].append(img_dir)
        print('Done read data')
        # target_image_dir = f"{self.root}/{self.phase}/TargetImage"
        # for style in os.listdir(target_image_dir):
        #     images_related_style = []
        #     for img in os.listdir(f"{target_image_dir}/{style}"):
        #         img_path = f"{target_image_dir}/{style}/{img}"
        #         self.target_images.append(img_path)
        #         images_related_style.append(img_path)
        #     self.style_to_images[style] = images_related_style

    def __getitem__(self, index):
        target_image = self.target_images[index]
        target_image_path, style, content = target_image['image'], target_image['wid'], target_image['label']
        # target_image_name = target_image_path.split('/')[-1]
        # style, content = target_image_name.split('.')[0].split('+')
        
        # Read content image
        # content_image_path = f"{self.root}/{self.phase}/ContentImage/arial/{content}.jpg"
        # content_image = Image.open(content_image_path).convert('RGB')
        content_image = self.tokenizer.encode(content)
        content_image = torch.tensor(content_image,dtype=torch.int64).long()

        # Random sample used for style image
        images_related_style = self.style_to_images[style].copy()
        images_related_style.remove(target_image_path)
        style_image_path = random.choice(images_related_style)
        style_image = Image.open(style_image_path).convert("RGB")
        
        # Read target image
        target_image = Image.open(target_image_path).convert("RGB")
        nonorm_target_image = self.nonorm_transforms(target_image)

        if self.transforms is not None:
            # content_image = self.transforms[0](content_image)
            style_image = self.transforms[1](style_image)
            target_image = self.transforms[2](target_image)
        
        sample = {
            "content_image": content_image,
            "style_image": style_image,
            "target_image": target_image,
            "target_image_path": target_image_path,
            "nonorm_target_image": nonorm_target_image}
        
        if self.scr:
            # Get neg image from the different style of the same content
            style_list = list(self.style_to_images.keys())
            style_index = style_list.index(style)
            style_list.pop(style_index)
            choose_neg_names = []
            for i in range(self.num_neg):
                choose_style = random.choice(style_list)
                choose_index = style_list.index(choose_style)
                style_list.pop(choose_index)
                choose_neg_name = f"{self.root}/train/TargetImage/{choose_style}/{choose_style}+{content}.jpg"
                choose_neg_names.append(choose_neg_name)

            # Load neg_images
            for i, neg_name in enumerate(choose_neg_names):
                neg_image = Image.open(neg_name).convert("RGB")
                if self.transforms is not None:
                    neg_image = self.transforms[2](neg_image)
                if i == 0:
                    neg_images = neg_image[None, :, :, :]
                else:
                    neg_images = torch.cat([neg_images, neg_image[None, :, :, :]], dim=0)
            sample["neg_images"] = neg_images

        return sample

    def __len__(self):
        return len(self.target_images)
