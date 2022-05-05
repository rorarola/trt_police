import torch
import torchvision.transforms as transforms

import cv2
import PIL.Image
import json
import time

import math
import os
import numpy as np
import traitlets

from torch2trt import TRTModule
import trt_pose
import trt_pose.coco
import trt_pose.models
from trt_pose.draw_objects import DrawObjects
from trt_pose.parse_objects import ParseObjects

import ipywidgets
from IPython.display import display
import torch2trt
import trt_pose.models



#############################################################


print_div = lambda x: print(f"\n{x}\n")

oom = False
os.environ['CUDA_VISIBLE_DEVICES']='2'

mean = torch.Tensor([0.485, 0.456, 0.406]).cuda()
std = torch.Tensor([0.229, 0.224, 0.225]).cuda()
device = torch.device('cuda')

def preprocess(image):
    global device
    device = torch.device('cuda')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = PIL.Image.fromarray(image)
    image = transforms.functional.to_tensor(image).to(device)
    image.sub_(mean[:, None, None]).div_(std[:, None, None])
    return image[None, ...]

############################################################


print_div("INIT")

with open('human_pose.json', 'r') as f:
   human_pose = json.load(f)

num_parts = len(human_pose['keypoints'])
num_links = len(human_pose['skeleton'])

model = trt_pose.models.resnet18_baseline_att(num_parts, 2 * num_links).cuda().eval()

# 改變標籤結構 : 增加頸部keypoint以及 paf的維度空間

topology = trt_pose.coco.coco_category_to_topology(human_pose)
# 用於解析預測後的 cmap與paf

parse_objects = ParseObjects(topology)
# parse_objects = ParseObjects(topology, cmap_threshold=0.30, link_threshold=0.30)
# 用於將keypoint繪製到圖片上

draw_objects = DrawObjects(topology)

############################################################


print_div("LOAD TENSORRT ENGINE")
WIDTH = 1920
HEIGHT = 1080

data = torch.zeros((1, 3, HEIGHT, WIDTH)).cuda()


model_trt = torch2trt.torch2trt(model, [data], fp16_mode=True, max_workspace_size=1<<25)

OPTIMIZED_MODEL = 'resnet18_baseline_att_224x224_A_epoch_249_trt.pth'

model_trt = TRTModule()
model_trt.load_state_dict(torch.load(OPTIMIZED_MODEL))

###########################################################


print_div("START STREAM")

cap = cv2.VideoCapture("astop.avi")

w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
crop_size = (w-h)//2
print(w, h, crop_size)

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('./output.avi', fourcc, 20.0, (w,  h))

start = time.time()

def execute(frame):
    #add
    image = frame
    image = cv2.flip(image,1)
    data = preprocess(image)
    cmap, paf = model_trt(data)
    cmap, paf = cmap.detach().cpu(), paf.detach().cpu()
    counts, objects, peaks = parse_objects(cmap, paf)

    print('type(objects) = ', type(objects))
    
    draw_objects(image, counts, objects, peaks)
    image = cv2.flip(image,1)

    out.write(image)
    cv2.imwrite('./images/test.jpg',image)


while(True):
    
    t_start = time.time()    
    ret, frame = cap.read()
    if not ret:
        continue

    t_end = time.time()

    print("outputting")


    execute(frame)

    now = time.time()

    #if cv2.waitKey(0) & 0xFF==ord('q'):
    #    break
    if now-start > 9:
        break

cap.release()
out.release()
print("finish")
