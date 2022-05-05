import torch
import torchvision.transforms as transforms

import cv2
import PIL.Image
import json
import time

from torch2trt import TRTModule
import trt_pose
import trt_pose.coco
import trt_pose.models

from trt_pose.draw_objects import DrawObjects
from trt_pose.parse_objects import ParseObjects
import os

os.environ['CUDA_VISIBLE_DEVICES']='2'

with open('human_pose.json', 'r') as f:
    human_pose = json.load(f)

topology = trt_pose.coco.coco_category_to_topology(human_pose)
parse_objects = ParseObjects(topology)
draw_objects = DrawObjects(topology)


# 我定義了兩個副函式，一個是print_div與cvt_trt.py中功能相同只是換了寫法；另一個是preprocess用來將OpenCV的圖片轉換成PIL並且進行正規化
print_div = lambda x: print(f"\n{x}\n")

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

#######################################################################################################################
# 載入TensorRT引擎，通過torch2trt的函式庫可以使用類似PyTorch的方式匯入，更簡單直觀：
print_div("LOAD TENSORRT ENGINE")

OPTIMIZED_MODEL = 'resnet18_baseline_att_224x224_A_epoch_249_trt.pth'

model_trt = TRTModule()
model_trt.load_state_dict(torch.load(OPTIMIZED_MODEL))


#######################################################################################################################
print_div("START STREAM")

cap = cv2.VideoCapture('astop.avi')

w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
crop_size = (w-h)//2
# print(w, h, crop_size)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('./output.avi', fourcc, 20.0, (224,  224))
while True:
    t_start = time.time()
    
    ret, frame = cap.read()
    
    if not ret:
        break

    # 資料前處理的部分除了剛剛撰寫的preprocess之外還需要針對frame裁切成正方形以及縮放大小，若省略這些步驟會導致模型辨識不出結果
    frame = frame[:, crop_size:(w-crop_size)]
    image = cv2.resize(frame, (224,224))
    data = preprocess(image)

    # 接著就是推論的部分，模型丟入資料後會輸出cmap以及paf，透過 parse_objects可以解析出裡面的內容，counts是物件的數量；objects是物件的座標等資訊；peaks用於繪製skeleton。
    cmap, paf = model_trt(data)
    cmap, paf = cmap.detach().cpu(), paf.detach().cpu()
    counts, objects, peaks = parse_objects(cmap, paf)
    print(peaks)
#, cmap_threshold=0.15, link_threshold=0.15)

    draw_objects(image, counts, objects, peaks)

    t_end = time.time()
    cv2.putText(image, f"FPS:{int(1/(t_end-t_start))}", (10,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1,  cv2.LINE_AA)
    #cv2.imshow('pose esimation', image)
    cv2.imwrite('./images/test.jpg',image)
    out.write(image)

    if cv2.waitKey(7) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()