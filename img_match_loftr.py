import sys
import os
sys.path.insert(1, os.path.join(os.getcwd(),'LoFTR'))

import cv2 as cv
import numpy as np
import torch
from src.loftr.loftr import LoFTR
from src.loftr.utils.supervision import spvs_coarse
from src.utils.misc import lower_config
from src.config.default import get_cfg_defaults

def matching_points_LoFTR(img_name1, img_name2):
    config_file_path = 'LoFTR/configs/loftr/indoor/loftr_ds.py'
    pretrained_ckpt_path = 'LoFTR/weights/indoor_ds_new.ckpt'
    
    config = get_cfg_defaults()
    config.merge_from_file(config_file_path)
    config_lower = lower_config(config)
    matcher = LoFTR(config_lower['loftr'])
    state_dict = torch.load(pretrained_ckpt_path, map_location='cpu')['state_dict']
    matcher.load_state_dict(state_dict, strict=True)
    matcher.to('cuda')
    matcher.eval()
    
    img1 = cv.imread(img_name1,cv.IMREAD_GRAYSCALE)/255. # queryImage
    img2 = cv.imread(img_name2,cv.IMREAD_GRAYSCALE)/255. # trainImage
    img1 = img1[np.newaxis, np.newaxis,...]
    img2 = img2[np.newaxis, np.newaxis,...]
    data = {'image0':torch.tensor(img1).float().cuda(), 'image1':torch.tensor(img2).float().cuda()}
    #spvs_coarse(data, config)
    #print(matcher.backbone(data['image0']))
    #print(torch.cat([data['image0'], data['image1']], dim=0).shape)
    matcher(data)

    return data['mkpts0_f'].cpu().numpy(), data['mkpts1_f'].cpu().numpy()

if __name__ == '__main__':
    import cv2 as cv
    from PIL import Image
    import random
    img_name1 = 'test111.jpg'
    img_name2 = 'test222.jpg'
    img1 = cv.imread(img_name1,cv.IMREAD_GRAYSCALE) # queryImage
    img2 = cv.imread(img_name2,cv.IMREAD_GRAYSCALE) # trainImage
    points1, points2 = matching_points_LoFTR(img_name1, img_name2)
    kp1 = []
    kp2 = []
    matches = []
    n_kp = len(points1)
    for i in range(n_kp):
        x1, y1 = points1[i]
        x2, y2 = points2[i]
        kp1.append(cv.KeyPoint(x1, y1, 1))
        kp2.append(cv.KeyPoint(x2, y2, 1))
        matches.append(cv.DMatch(i, i, ((x1-x2)**2+(y1-y2)**2)**0.5))
    img3 = cv.drawMatches(img1,kp1,img2,kp2,matches,None,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    img3 = Image.fromarray(img3)
    img3.save('match_loftr.png')
