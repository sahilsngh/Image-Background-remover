import os
import re
import sys
import cv2
import random
import warnings
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from utils.image_manipulations import stack_images, save_image
from model.modnet import MODNet

warnings.filterwarnings("ignore")

if __name__ == '__main__':

    path = f'{os.getcwd()}'.replace('\\', '/')
    model_dir = path + '/model/modnet_image_matting.ckpt'
    input_dir = path + '/inputs/'
    alpha_dir = path + '/alphas/'
    output_dir = path + '/results/'

    # define hyper-parameters
    ref_size = 512

    # define image to tensor transform
    im_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]
    )

    # create MODNet and load the pre-trained ckpt
    modnet = MODNet(backbone_pretrained=False)
    modnet = nn.DataParallel(modnet).cuda()
    modnet.load_state_dict(torch.load(model_dir))
    modnet.eval()
    print('[INFO]...model loaded successfully.')

    # inference images
    im_names = os.listdir(input_dir)
    for im_name in im_names:

        # read image
        im = Image.open(os.path.join(input_dir, im_name))

        # unify image channels to 3
        im = np.asarray(im)
        if len(im.shape) == 2:
            im = im[:, :, None]
        if im.shape[2] == 1:
            im = np.repeat(im, 3, axis=2)
        elif im.shape[2] == 4:
            im = im[:, :, 0:3]

        # convert image to PyTorch tensor
        im = Image.fromarray(im)
        im = im_transform(im)

        # add mini-batch dim
        im = im[None, :, :, :]

        # resize image for input
        im_b, im_c, im_h, im_w = im.shape
        if max(im_h, im_w) < ref_size or min(im_h, im_w) > ref_size:
            if im_w >= im_h:
                im_rh = ref_size
                im_rw = int(im_w / im_h * ref_size)
            elif im_w < im_h:
                im_rw = ref_size
                im_rh = int(im_h / im_w * ref_size)
        else:
            im_rh = im_h
            im_rw = im_w
        
        im_rw = im_rw - im_rw % 32
        im_rh = im_rh - im_rh % 32
        im = F.interpolate(im, size=(im_rh, im_rw), mode='area')

        # inference
        _, _, matte = modnet(im.cuda(), True)

        # resize and save matte
        matte = F.interpolate(matte, size=(im_h, im_w), mode='area')
        matte = matte[0][0].data.cpu().numpy()
        # code for stacking images
        matte_name = im_name.split('.')[0] + '_alpha.png'
        Image.fromarray(((matte * 255).astype('uint8')), mode='L').save(os.path.join(alpha_dir, matte_name))

    print('[INFO]..Alpha images created.')

    # Creating completed images.
    al_names = os.listdir(alpha_dir)
    names = []
    for i in im_names:
        for j in al_names:
            pattern = j.replace('_alpha.png', '')
            match = re.match(pattern, i)
            if match:
                # list -> [image, alpha]
                names.append([i,j])
            else:
                pass
    
    # print(names,'\n')
    # names = random.shuffle(names)
    for name in names:
          
        print('Process image: {0}'.format(name[0]))
        
        alpha = cv2.imread(os.path.join(alpha_dir, name[1]))
        og = cv2.imread(os.path.join(input_dir, name[0]))

        og = cv2.cvtColor(og, cv2.COLOR_BGR2RGB)
        alpha = cv2.cvtColor(alpha, cv2.COLOR_BGR2GRAY)   

        final = stack_images(og, alpha)
        image_name = output_dir + name[0].split('.')[0] + '_bg_removed.png'
        save_image(image_name, final)

    print("[INFO]>> Operation successful!")
