import os
import numpy as np
import PIL
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import os, cv2, json
import random

import warnings
warnings.filterwarnings("ignore")

training_templates_smallest = [
    'photo of a sks {}',
]

reg_templates_smallest = [
    'photo of a {}',
]

imagenet_templates_small = [
    'a photo of a {}',
    'a rendering of a {}',
    'a cropped photo of the {}',
    'the photo of a {}',
    'a photo of a clean {}',
    'a photo of a dirty {}',
    'a dark photo of the {}',
    'a photo of my {}',
    'a photo of the cool {}',
    'a close-up photo of a {}',
    'a bright photo of the {}',
    'a cropped photo of a {}',
    'a photo of the {}',
    'a good photo of the {}',
    'a photo of one {}',
    'a close-up photo of the {}',
    'a rendition of the {}',
    'a photo of the clean {}',
    'a rendition of a {}',
    'a photo of a nice {}',
    'a good photo of a {}',
    'a photo of the nice {}',
    'a photo of the small {}',
    'a photo of the weird {}',
    'a photo of the large {}',
    'a photo of a cool {}',
    'a photo of a small {}',
    'an illustration of a {}',
    'a rendering of a {}',
    'a cropped photo of the {}',
    'the photo of a {}',
    'an illustration of a clean {}',
    'an illustration of a dirty {}',
    'a dark photo of the {}',
    'an illustration of my {}',
    'an illustration of the cool {}',
    'a close-up photo of a {}',
    'a bright photo of the {}',
    'a cropped photo of a {}',
    'an illustration of the {}',
    'a good photo of the {}',
    'an illustration of one {}',
    'a close-up photo of the {}',
    'a rendition of the {}',
    'an illustration of the clean {}',
    'a rendition of a {}',
    'an illustration of a nice {}',
    'a good photo of a {}',
    'an illustration of the nice {}',
    'an illustration of the small {}',
    'an illustration of the weird {}',
    'an illustration of the large {}',
    'an illustration of a cool {}',
    'an illustration of a small {}',
    'a depiction of a {}',
    'a rendering of a {}',
    'a cropped photo of the {}',
    'the photo of a {}',
    'a depiction of a clean {}',
    'a depiction of a dirty {}',
    'a dark photo of the {}',
    'a depiction of my {}',
    'a depiction of the cool {}',
    'a close-up photo of a {}',
    'a bright photo of the {}',
    'a cropped photo of a {}',
    'a depiction of the {}',
    'a good photo of the {}',
    'a depiction of one {}',
    'a close-up photo of the {}',
    'a rendition of the {}',
    'a depiction of the clean {}',
    'a rendition of a {}',
    'a depiction of a nice {}',
    'a good photo of a {}',
    'a depiction of the nice {}',
    'a depiction of the small {}',
    'a depiction of the weird {}',
    'a depiction of the large {}',
    'a depiction of a cool {}',
    'a depiction of a small {}',
]

imagenet_dual_templates_small = [
    'a photo of a {} with {}',
    'a rendering of a {} with {}',
    'a cropped photo of the {} with {}',
    'the photo of a {} with {}',
    'a photo of a clean {} with {}',
    'a photo of a dirty {} with {}',
    'a dark photo of the {} with {}',
    'a photo of my {} with {}',
    'a photo of the cool {} with {}',
    'a close-up photo of a {} with {}',
    'a bright photo of the {} with {}',
    'a cropped photo of a {} with {}',
    'a photo of the {} with {}',
    'a good photo of the {} with {}',
    'a photo of one {} with {}',
    'a close-up photo of the {} with {}',
    'a rendition of the {} with {}',
    'a photo of the clean {} with {}',
    'a rendition of a {} with {}',
    'a photo of a nice {} with {}',
    'a good photo of a {} with {}',
    'a photo of the nice {} with {}',
    'a photo of the small {} with {}',
    'a photo of the weird {} with {}',
    'a photo of the large {} with {}',
    'a photo of a cool {} with {}',
    'a photo of a small {} with {}',
]

per_img_token_list = [
    'א', 'ב', 'ג', 'ד', 'ה', 'ו', 'ז', 'ח', 'ט', 'י', 'כ', 'ל', 'מ', 'נ', 'ס', 'ע', 'פ', 'צ', 'ק', 'ר', 'ש', 'ת',
]

import fnmatch

def find_files(directory : str, pattern : str):
        for root, dirs, files in os.walk(directory):
            for basename in files:
                if fnmatch.fnmatch(basename, pattern):
                    filename = os.path.join(root, basename)
                    yield filename


from typing import List, Dict, Tuple
from collections import defaultdict
import csv
from pprint import pprint


def read_csv_as_dict(csv_pathes: List[str]):
    prompts = defaultdict(str)
    
    for csv_path in csv_pathes:
        with open(csv_path, 'r') as csv_file:
            reader = csv.DictReader(csv_file)
            for row in reader:
                prompt = [row['Propmt'].replace('"', '').strip().lower()]
                prompt.extend([tag.replace('"', '').strip().lower() for tag in row[None]])
                
                prompts[row['FileName']] = ','.join(prompt)               

    return prompts


class PersonalizedBase(Dataset):
    def __init__(self,
                 data_root,
                 size=None,
                 repeats=100,
                 interpolation="bicubic",
                 flip_p=0.5,
                 set="train",
                 placeholder_token="dog",
                 per_image_tokens=False,
                 center_crop=False,
                 mixing_prob=0.25,
                 coarse_class_text=None,
                 reg = True
                 ):

        self.images_root = '/home/ubuntu/selected_images'
        self.masks_root = '/home/ubuntu/masks_random_prompt_extended'
        self.images_path =  list(find_files(self.images_root, '*.jpg'))
        self.masks_path = list(find_files(self.images_root, '*.png'))
        self.prompt_dict = read_csv_as_dict(['/home/ubuntu/selected_images/logs_rand_prompt_extended.csv'])
        
        if set == "train":
            self.images_path = self.images_path[:int(len(self.images_path)*0.75)]
        else:
            self.images_path = self.images_path[int(len(self.images_path)*0.75):]
            
        # self._length = len(self.image_paths)
        self.num_images = len(self.images_path)
        self._length = self.num_images 

        self.placeholder_token = placeholder_token

        self.per_image_tokens = per_image_tokens
        self.center_crop = center_crop
        self.mixing_prob = mixing_prob

        self.coarse_class_text = coarse_class_text

        if per_image_tokens:
            assert self.num_images < len(per_img_token_list), f"Can't use per-image tokens when the training set contains more than {len(per_img_token_list)} tokens. To enable larger sets, add more tokens to 'per_img_token_list'."

        if set == "train":
            self._length = self.num_images

        self.size = size
        self.interpolation = {"linear": PIL.Image.LINEAR,
                              "bilinear": PIL.Image.BILINEAR,
                              "bicubic": PIL.Image.BICUBIC,
                              "lanczos": PIL.Image.LANCZOS,
                              }[interpolation]
        self.flip = transforms.RandomHorizontalFlip(p=flip_p)
        self.reg = reg

    def __len__(self):
        return self._length

    def __getitem__(self, i):
        example = {}
        
        image = Image.open(os.path.join(self.images_root,self.images_path[i]))
        try:
            mask = cv2.imread(os.path.join(self.masks_root,self.images_path[i].split('/')[-1][:-3]+'png'))[:,:,0]
            bboxes = torchvision.ops.masks_to_boxes(mask)
        except:
            bboxes = None
            
        if not image.mode == "RGB":
            image = image.convert("RGB")
        img = np.array(image).astype(np.uint8)
        image_h, image_w = img.shape[:2]
        if bboxes is not None:
            for bbox in bboxes:
                bbox[0] -= image_w / (bbox[2] - bbox[0]) * (bbox[2] - bbox[0])
                bbox[0] = max(0,bbox[0])
                bbox[1] -= image_h / (bbox[3] - bbox[1]) * (bbox[3] - bbox[1])
                bbox[1] = max(0,bbox[1])
                bbox[2] += image_w / (bbox[2] - bbox[0]) * (bbox[2] - bbox[0])
                bbox[2] = min(image_w,bbox[2])
                bbox[3] += image_h / (bbox[3] - bbox[1]) * (bbox[3] - bbox[1])
                bbox[3] = min(image_h,bbox[3])
                break
            img = img[bbox[1]:bbox[3],bbox[0]:bbox[2]]
            
        
        example["caption"] = self.prompt_dict.get(self.images_path[i].split('/')[-1], 
        'fire, flame, smoke, warning, alarm, factory, realistic, forest fire, house flame, details, flambe, bright lights, deadly fire, burns, photo of fire,  breaking news report, explosions, warm colors, smolder, smoke, 8 k'
        )
        
        image = Image.fromarray(img)
        if self.size is not None:
            image = image.resize((self.size, self.size), resample=self.interpolation)

        image = np.array(image).astype(np.uint8)
        example["image"] = (image / 127.5 - 1.0).astype(np.float32)
        return example