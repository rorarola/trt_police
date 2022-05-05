# https://www.rs-online.com/designspark/nvidia-jetson-nano-pose-estimation-cn
# 首先我們需要將下載下來的PyTorch模型轉換成TensorRT引擎，所以我們先新增一個檔案名為 cvt_trt.py，接下來就一一新增程式碼，首先導入函式庫：
#
import json

import torch2trt
import torch

import trt_pose.coco
import trt_pose.models

def print_div(txt):    
    print(f"\n{txt}\n")


#######################################################################################################################
# 接著載入預訓練模型並且修改最後一層，這邊增加的是 cmap_channels跟 paf_channels，每一個模型會有兩個輸出cmap與paf，cmap(Confidence Map)就是Pose Estimation常見的熱力圖，
# 能透過其找出Keypoint所在；Paf (Part Affinity Fields)則是提供一個向量區域，可以表示人體各個Keypoint、skeleton的關聯性，而paf 由於是一個向量所以會有 x, y的數值，輸出要乘以2。

print_div("LOAD MODEL")

with open('human_pose.json', 'r') as f:
    human_pose = json.load(f)

# 取得 keypoint 數量

num_parts = len(human_pose['keypoints'])
num_links = len(human_pose['skeleton'])

# 修改輸出層

model = trt_pose.models.resnet18_baseline_att(num_parts, 2 * num_links).cuda().eval()

# 載入權重

MODEL_WEIGHTS = 'resnet18_baseline_att_224x224_A_epoch_249_trt.pth'
model.load_state_dict(torch.load(MODEL_WEIGHTS))

#######################################################################################################################
# 進行轉換與儲存引擎：
print_div("COVERTING")

WIDTH, HEIGHT = 224, 224
data = torch.zeros((1, 3, HEIGHT, WIDTH)).cuda()
model_trt = torch2trt.torch2trt(model, [data], fp16_mode=True, max_workspace_size=1<<25)

print_div("SAVING TENSORRT")

OPTIMIZED_MODEL = 'resnet18_baseline_att_224x224_A_epoch_249_trt.pth'
torch.save(model_trt.state_dict(), OPTIMIZED_MODEL)

print_div("FINISH")