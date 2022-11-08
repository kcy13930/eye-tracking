from __future__ import print_function
import sys, os
import argparse
import numpy as np
import cv2
import time

from PIL import Image
import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import transforms
import torch.backends.cudnn as cudnn
import torchvision

sys.path.append('./gaze_estimation')
sys.path.append('./face_detector')

from L2CS.utils import select_device, draw_gaze
from L2CS.model import L2CS

from retinaface.data import config
from retinaface.layers.functions.prior_box import PriorBox
from retinaface.utils.nms.py_cpu_nms import py_cpu_nms
from retinaface.models.retinaface import RetinaFace
from retinaface.utils.box_utils import decode, decode_landm

import flask

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(
        description='Gaze evalution using model pretrained with L2CS-Net on Gaze360.')
    parser.add_argument(
        '--gpu',dest='gpu_id', help='GPU device id to use [0]',
        default="0", type=str)
    parser.add_argument(
        '--snapshot',dest='snapshot', help='Path of model snapshot.', 
        default='gaze_estimation/L2CS/output/snapshots/L2CS-mpiigaze_1654834358/fold0/_epoch_50.pkl', type=str)
    parser.add_argument(
        '--cam',dest='cam_id', help='Camera device id to use [0]',  
        default=0, type=int)
    parser.add_argument(
        '--arch',dest='arch',help='Network architecture, can be: ResNet18, ResNet34, ResNet50, ResNet101, ResNet152',
        default='ResNet50', type=str)

    args = parser.parse_args()
    return args

def getArch(arch,bins):
    # Base network structure
    if arch == 'ResNet18':
        model = L2CS( torchvision.models.resnet.BasicBlock,[2, 2,  2, 2], bins)
    elif arch == 'ResNet34':
        model = L2CS( torchvision.models.resnet.BasicBlock,[3, 4,  6, 3], bins)
    elif arch == 'ResNet101':
        model = L2CS( torchvision.models.resnet.Bottleneck,[3, 4, 23, 3], bins)
    elif arch == 'ResNet152':
        model = L2CS( torchvision.models.resnet.Bottleneck,[3, 8, 36, 3], bins)
    else:
        if arch != 'ResNet50':
            print('Invalid value for architecture is passed! '
                'The default value of ResNet50 will be used instead!')
        model = L2CS( torchvision.models.resnet.Bottleneck, [3, 4, 6, 3], bins)
    return model

def check_keys(model, pretrained_state_dict):
    ckpt_keys = set(pretrained_state_dict.keys())
    model_keys = set(model.state_dict().keys())
    used_pretrained_keys = model_keys & ckpt_keys
    unused_pretrained_keys = ckpt_keys - model_keys
    missing_keys = model_keys - ckpt_keys
    print('Missing keys:{}'.format(len(missing_keys)))
    print('Unused checkpoint keys:{}'.format(len(unused_pretrained_keys)))
    print('Used keys:{}'.format(len(used_pretrained_keys)))
    assert len(used_pretrained_keys) > 0, 'load NONE from pretrained checkpoint'
    return True


def remove_prefix(state_dict, prefix):
    ''' Old style model is stored with all names of parameters sharing common prefix 'module.' '''
    print('remove prefix \'{}\''.format(prefix))
    f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
    return {f(key): value for key, value in state_dict.items()}


def load_model(model, pretrained_path, load_to_cpu):
    print('Loading pretrained model from {}'.format(pretrained_path))
    if load_to_cpu:
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage)
    else:
        device = torch.cuda.current_device()
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage.cuda(device))
    if "state_dict" in pretrained_dict.keys():
        pretrained_dict = remove_prefix(pretrained_dict['state_dict'], 'module.')
    else:
        pretrained_dict = remove_prefix(pretrained_dict, 'module.')
    check_keys(model, pretrained_dict)
    model.load_state_dict(pretrained_dict, strict=False)
    return model

args = parse_args()
cudnn.enabled = True
arch=args.arch
batch_size = 1
cam = args.cam_id
gpu = select_device(args.gpu_id, batch_size=batch_size)
snapshot_path = args.snapshot

transformations = transforms.Compose([
    transforms.Resize(448),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

gaze_tracker = getArch(arch, 28)
print('Loading snapshot.')
saved_state_dict = torch.load(snapshot_path)
gaze_tracker.load_state_dict(saved_state_dict)
gaze_tracker.cuda(gpu)
gaze_tracker.eval()


softmax = nn.Softmax(dim=1)

# face detection
trained_model = './face_detector/retinaface/weights/mobilenet0.25_Final.pth'
cpu = False
confidence_threshold = 0.02
top_k = 5000
nms_threshold = 0.4
keep_top_k = 750

cfg = config.cfg_mnet
face_detector = RetinaFace(cfg=cfg, phase = 'test')
face_detector = load_model(face_detector, trained_model, cpu)
face_detector.eval()
print('Finished loading model!')
cudnn.benchmark = True
device = torch.device("cpu" if cpu else "cuda")
face_detector = face_detector.to(device)  
resize = 1

idx_tensor = [idx for idx in range(28)]
idx_tensor = torch.FloatTensor(idx_tensor).cuda(gpu)
x=0

app = flask.Flask(__name__)
@app.route("/predict", methods=["POST"])
def predict():
    data = {"success" : False}
    data["predictions"] = []
    if flask.request.method == "POST":
        if flask.request.files.get("image"):
            response = flask.request.files["image"].read()
            encoded_response = np.frombuffer(response, dtype=np.uint8)
            image = cv2.imdecode(encoded_response, cv2.IMREAD_COLOR)
            orin_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            with torch.no_grad():
                while True:
                    start_fps = time.time()

                    img = np.float32(orin_img.copy())
            
                    im_height, im_width, _ = img.shape
                    scale = torch.Tensor([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
                    img -= (104, 117, 123)
                    img = img.transpose(2, 0, 1)
                    img = torch.from_numpy(img).unsqueeze(0)
                    img = img.to(device)
                    scale = scale.to(device)
            
                    loc, conf, landms = face_detector(img)
                    priorbox = PriorBox(cfg, image_size=(im_height, im_width))
                    priors = priorbox.forward()
                    priors = priors.to(device)
                    prior_data = priors.data
                    boxes = decode(loc.data.squeeze(0), prior_data, cfg['variance'])
                    boxes = boxes * scale / resize
                    boxes = boxes.cpu().numpy()
                    scores = conf.squeeze(0).data.cpu().numpy()[:, 1]
                    landms = decode_landm(landms.data.squeeze(0), prior_data, cfg['variance'])
                    scale1 = torch.Tensor([img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                                        img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                                        img.shape[3], img.shape[2]])
                    scale1 = scale1.to(device)
                    landms = landms * scale1 / resize
                    landms = landms.cpu().numpy()
            
                    # ignore low scores
                    inds = np.where(scores > confidence_threshold)[0]
                    boxes = boxes[inds]
                    landms = landms[inds]
                    scores = scores[inds]
            
                    # keep top-K before NMS
                    order = scores.argsort()[::-1][:top_k]
                    boxes = boxes[order]
                    landms = landms[order]
                    scores = scores[order]
            
                    # do NMS
                    faces = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
                    keep = py_cpu_nms(faces, nms_threshold)
                    # keep = nms(dets, args.nms_threshold,force_cpu=args.cpu)
                    faces = faces[keep, :]
                    landms = landms[keep]
            
                    # keep top-K faster NMS
                    faces = faces[:keep_top_k, :]
                    landms = landms[:keep_top_k, :]
            
                    faces = np.concatenate((faces, landms), axis=1)
            
                    if faces is not None:
                        # faces = faces[faces.T[4]>=0.95]
                        faces = faces[faces.T[4]>=0.6]
                        if faces.any():
                        # for face in faces:
                            face = faces[0]
                            box = face[0:4]
                            landmarks = face[5:15].astype(np.int32)
                            score = face[4]
            
                            #if score < .95:
                            if score < .6:
                                continue
            
                            x_min=int(box[0])
                            if x_min < 0:
                                x_min = 0
                            y_min=int(box[1])
                            if y_min < 0:
                                y_min = 0
            
                            x_max=int(box[2])
                            y_max=int(box[3])
                            bbox_width = x_max - x_min
                            bbox_height = y_max - y_min
                            # x_min = max(0,x_min-int(0.2*bbox_height))
                            # y_min = max(0,y_min-int(0.2*bbox_width))
                            # x_max = x_max+int(0.2*bbox_height)
                            # y_max = y_max+int(0.2*bbox_width)
                            # bbox_width = x_max - x_min
                            # bbox_height = y_max - y_min
            
                            # Crop image
                            img = orin_img[y_min:y_max, x_min:x_max]
                            img = cv2.resize(img, (224, 224))
                            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                            im_pil = Image.fromarray(img)
                            img = transformations(im_pil)
                            img  = Variable(img).cuda(gpu)
                            img  = img.unsqueeze(0) 
            
                            # gaze prediction
                            gaze_pitch, gaze_yaw = gaze_tracker(img)
            
            
                            pitch_predicted = softmax(gaze_pitch)
                            yaw_predicted = softmax(gaze_yaw)
                            # Get continuous predictions in degrees.
                            # print('idx : ', idx_tensor)
                            pitch_predicted = torch.sum(pitch_predicted.data[0] * idx_tensor) * 3 - 42
                            yaw_predicted = torch.sum(yaw_predicted.data[0] * idx_tensor) * 3 - 42

                            pitch_predicted = pitch_predicted.cpu().detach().numpy().item(0)
                            yaw_predicted = yaw_predicted.cpu().detach().numpy().item(0)

                            pitch_predicted_pi = pitch_predicted * np.pi/180.0
                            yaw_predicted_pi = yaw_predicted * np.pi/180.0
            
                            draw_gaze(x_min,y_min,bbox_width, bbox_height,orin_img,(pitch_predicted_pi,yaw_predicted_pi),color=(0,0,255))
                            cv2.putText(orin_img, '{:.3f}, {:.3f}'.format(yaw_predicted, pitch_predicted), (x_min, y_min), cv2.FONT_HERSHEY_COMPLEX_SMALL,1,(0, 255, 0),1,cv2.LINE_AA)
                            cv2.rectangle(orin_img, (x_min, y_min), (x_max, y_max), (0,255,0), 1)
            
                            # landms
                            cv2.circle(orin_img, (landmarks[0], landmarks[1]), 1, (0, 0, 255), 4)
                            cv2.circle(orin_img, (landmarks[2], landmarks[3]), 1, (0, 255, 255), 4)
                            cv2.circle(orin_img, (landmarks[4], landmarks[5]), 1, (255, 0, 255), 4)
                            cv2.circle(orin_img, (landmarks[6], landmarks[7]), 1, (0, 255, 0), 4)
                            cv2.circle(orin_img, (landmarks[8], landmarks[9]), 1, (255, 0, 0), 4)

                            # loop over the results and add them to the list of
                            # returned predictions
                            result = {"pitch": pitch_predicted, "yaw": yaw_predicted}
                            data["predictions"].append(result)

                            # indicate that the request was a success
                            data["success"] = True

                    myFPS = 1.0 / (time.time() - start_fps)
                    cv2.putText(orin_img, 'FPS: {:.1f}'.format(myFPS), (10, 20), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 0), 1, cv2.LINE_AA)

                    orin_img = cv2.cvtColor(orin_img, cv2.COLOR_RGB2BGR)
                    
                    cv2.imwrite('./result/orin.png', image)
                    cv2.imwrite('./result/test.png', orin_img)
                    
                    break
    print(data)
	# return the data dictionary as a JSON response
    return flask.jsonify(data)

if __name__ == '__main__':
    if not os.path.isdir('./result'):
        os.mkdir('./result')
    
    print("sever starting")
    app.run(port=5000)
