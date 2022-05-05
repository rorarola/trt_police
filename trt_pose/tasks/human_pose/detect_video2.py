# https://spyjetson.blogspot.com/2019/12/jetsonnano-human-pose-estimation-using.html
import json
import trt_pose.coco
import trt_pose.models
import torch
import torch2trt
from torch2trt import TRTModule
import time, sys
import cv2
import torchvision.transforms as transforms
import PIL.Image
from trt_pose.draw_objects import DrawObjects
from trt_pose.parse_objects import ParseObjects
import argparse
import csv
import os

os.environ['CUDA_VISIBLE_DEVICES']='2'


######################
#       args         #
######################

KEYPOINT_INDEXES = {
    0: "nose",
    1: "left_eye",
    2: "right_eye",
    3: "left_ear",
    4: "right_ear",
    5: "left_shoulder",
    6: "right_shoulder",
    7: "left_elbow",
    8: "right_elbow",
    9: "left_wrist",
    10:"right_wrist",
    11:"left_hip",
    12:"right_hip",
    13:"left_knee",
    14:"right_knee",
    15:"left_ankle",
    16:"right_ankle",
    17:"neck"
}

csv_folder_root = './pose_coord_3/'
video_folder_root = './video/'





parser = argparse.ArgumentParser(description='TensorRT pose estimation run')
parser.add_argument('--model', type=str, default='resnet', help = 'resnet or densenet' )
args = parser.parse_args()

with open('human_pose.json', 'r') as f:
    human_pose = json.load(f)

topology = trt_pose.coco.coco_category_to_topology(human_pose)
parse_objects = ParseObjects(topology)
draw_objects = DrawObjects(topology)

num_parts = len(human_pose['keypoints'])
num_links = len(human_pose['skeleton'])


if 'resnet' in args.model:
    print('------ model = resnet--------')
    MODEL_WEIGHTS = 'resnet18_baseline_att_224x224_A_epoch_249.pth'
    OPTIMIZED_MODEL = 'resnet18_baseline_att_224x224_A_epoch_249_trt.pth'
    model = trt_pose.models.resnet18_baseline_att(num_parts, 2 * num_links).cuda().eval()
    WIDTH = 224
    HEIGHT = 224

else:    
    print('------ model = densenet--------')
    MODEL_WEIGHTS = 'densenet121_baseline_att_256x256_B_epoch_160.pth'
    OPTIMIZED_MODEL = 'densenet121_baseline_att_256x256_B_epoch_160_trt.pth'
    model = trt_pose.models.densenet121_baseline_att(num_parts, 2 * num_links).cuda().eval()
    WIDTH = 256
    HEIGHT = 256

data = torch.zeros((1, 3, HEIGHT, WIDTH)).cuda()
if os.path.exists(OPTIMIZED_MODEL) == False:
    model.load_state_dict(torch.load(MODEL_WEIGHTS))
    model_trt = torch2trt.torch2trt(model, [data], fp16_mode=True, max_workspace_size=1<<25)
    torch.save(model_trt.state_dict(), OPTIMIZED_MODEL)

model_trt = TRTModule()
model_trt.load_state_dict(torch.load(OPTIMIZED_MODEL))

t0 = time.time()
torch.cuda.current_stream().synchronize()
for i in range(50):
    y = model_trt(data)
torch.cuda.current_stream().synchronize()
t1 = time.time()

print(50.0 / (t1 - t0))

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



# =============================================================================================== #
'''
hnum: 0 based human index
kpoint : keypoints (float type range : 0.0 ~ 1.0 ==> later multiply by image width, height
'''

def get_keypoint(humans, hnum, peaks):
    #check invalid human index
    kpoint = []
    human = humans[0][hnum]
    C = human.shape[0]
    for j in range(C):
        k = int(human[j])
        if k >= 0:
            peak = peaks[0][j][k]   # peak[1]:width, peak[0]:height
            peak = (j, float(peak[0]), float(peak[1]))
            kpoint.append(peak)
            # print('index:%d : success [%5.3f, %5.3f]'%(j, peak[1], peak[2]) )
             # Extract Pose landmarks                    
        else:    
            peak = (j, None, None)
            kpoint.append(peak)
            #print('index:%d : None %d'%(j, k) )

    return kpoint

def parse_image(img, src, t, X_compress, Y_compress, frame_count, csv_coord_filename):
    color = (0, 255, 0)
    data = preprocess(img)
    cmap, paf = model_trt(data)
    cmap, paf = cmap.detach().cpu(), paf.detach().cpu()
    counts, objects, peaks = parse_objects(cmap, paf)#, cmap_threshold=0.15, link_threshold=0.15)
    fps = 1.0 / (time.time() - t)
    print(f'counts[0]={counts[0]}')
    
    # for i in range(counts[0]): # counts[0]: 圖片中有幾個人
    ## 抓第一個人就好了
    keypoints = get_keypoint(objects, 0, peaks)

    csv_xy_row = []
    for peak in keypoints:
        csv_xy_row.extend([peak[1], peak[2]])

    # Export to CSV
    with open(csv_coord_filename, mode='a', newline='') as f:
        csv_xy_row.insert(0, frame_count)
        csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        csv_writer.writerow(csv_xy_row)

    for j in range(len(keypoints)):
        if keypoints[j][1]:
            x = round(keypoints[j][2] * WIDTH * X_compress)
            y = round(keypoints[j][1] * HEIGHT * Y_compress)
            cv2.circle(src, (x, y), 3, color, 2)
            cv2.putText(src , "%d" % int(keypoints[j][0]), (x + 5, y),  cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 1)
            cv2.circle(src, (x, y), 3, color, 2)
    # print("FPS:%f "%(fps))
    #draw_objects(img, counts, objects, peaks)

    cv2.putText(src , "FPS: %f" % (fps), (20, 20),  cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1)
    return src
    






def get_pose_coord(fullpath):
    basename = os.path.basename(fullpath)
    video_name = os.path.splitext(basename)[0]

    cap = cv2.VideoCapture(fullpath)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    out_video = cv2.VideoWriter(f'./tmp/{video_name}.mp4', fourcc, cap.get(cv2.CAP_PROP_FPS), (w, h))
    

    X_compress = w / WIDTH * 1.0
    Y_compress = h / HEIGHT * 1.0


    csv_headers = ['frame']
    for keypoint in KEYPOINT_INDEXES.values():
        csv_headers.extend([keypoint+'_x', keypoint+'_y'])

    csv_coord_filename = csv_folder_root + video_name + '_coord.csv'

    with open(csv_coord_filename, mode='w', newline='') as f:
        csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        csv_writer.writerow(csv_headers) 

    frame_count = 0
    while True:
        t = time.time()
        ret_val, dst = cap.read()
        if ret_val == False:
            print("Camera read Error")
            break

        img = cv2.resize(dst, dsize=(WIDTH, HEIGHT), interpolation=cv2.INTER_AREA)
        src = parse_image(img, dst, t, X_compress, Y_compress, frame_count, csv_coord_filename)
        out_video.write(src)
        frame_count += 1


    cv2.destroyAllWindows()
    out_video.release()
    cap.release()
    





files = os.listdir(video_folder_root)

# 以迴圈處理
for f in files:
    fullpath = os.path.join(video_folder_root, f)
    get_pose_coord(fullpath)