
import argparse
import os
import platform
import sys
from pathlib import Path

import torch
import numpy as np

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_boxes, scale_segments,
                           strip_optimizer)
from utils.plots import Annotator, colors, save_one_box
from utils.segment.general import masks2segments, process_mask, process_mask_native
from utils.torch_utils import select_device, smart_inference_mode
from utils.augmentations import letterbox


def load_model(
        weights='./best.pt',  # model.pt path(s)
        data=ROOT / 'data/coco128.yaml',  # dataset.yaml path
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference

):

    # Load model
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt

    return model, stride, names, pt


def run(model, img, stride, pt,
        imgsz=(640, 640),  # inference size (height, width)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        half=False,  # use FP16 half-precision inference
        retina_masks=True,
        ):
    imgsz = check_img_size(imgsz, s=stride)  # check image size
    model.warmup(imgsz=(1 if pt else 1, 3, *imgsz))  # warmup

    cal_detect = []
    device = select_device(device)
    names = model.module.names if hasattr(model, 'module') else model.names  # get class names

    # Set Dataloader
    im = letterbox(img, imgsz, stride, pt)[0]

    # Convert
    im = im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
    im = np.ascontiguousarray(im)

    im = torch.from_numpy(im).to(device)
    im = im.half() if half else im.float()  # uint8 to fp16/32
    im /= 255  # 0 - 255 to 0.0 - 1.0
    if len(im.shape) == 3:
        im = im[None]  # expand for batch dim

    pred, proto = model(im, augment=augment)[:2]

    pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det, nm=32)
    # Process detections
    for i, det in enumerate(pred):  # detections per image
        annotator = Annotator(img, line_width=1, example=str(names))
        if len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], img.shape).round()  # rescale boxes to im0 size
            masks = process_mask_native(proto[i], det[:, 6:], det[:, :4], img.shape[:2])  # HWC
            #segments = [
                #scale_segments(img.shape if retina_masks else im.shape[2:], x, img.shape, normalize=True)
                #for x in reversed(masks2segments(masks))]

            # Write results
            ###############################################
            id_list = []
            for id in range(len(det[:, :6])):
                # print(det[id])
                # print(masks[id])
                class_name = names[int(det[:, :6][id][5])]
                #print(det[:, :6][id])
                #print(class_name)
                # if class_name == 'person':
                # id_list.append(id)

            # print(id_list)

            def del_tensor(arr, id_list):
                if len(id_list) == 0:
                    return arr
                elif len(id_list) == 1:
                    arr1 = arr[:id_list[0]]
                    arr2 = arr[id_list[0] + 1:]
                    return torch.cat((arr1, arr2), dim=0)
                else:
                    arr1 = arr[:id_list[0]]
                    arr2 = arr[id_list[0] + 1:id_list[1]]
                    arr1 = torch.cat((arr1, arr2), dim=0)
                    for id_index in range(len(id_list)):
                        arr2 = arr[id_list[id_index - 1] + 1:id_list[id_index]]
                        arr1 = torch.cat((arr1, arr2), dim=0)
                    return arr1

            det = del_tensor(det, id_list)
            masks = del_tensor(masks, id_list)
            ###############################################

            annotator.masks(
                masks,
                colors=[colors(x, True) for x in det[:, 5]],
                im_gpu=torch.as_tensor(img, dtype=torch.float16).to(device).permute(2, 0, 1).flip(
                    0).contiguous() /
                       255 if retina_masks else im[i])

            for j, (*xyxy, conf, cls) in enumerate(reversed(det[:, :6])):
                c = int(cls)  # integer class
                label = f'{names[c]}'
                #lbl = names[int(cls)]
                #print(segments[j])
                #if lbl not in [' Chef clothes',' clothes']:
                    #continue
                #print(masks[j])
                #cv2.imshow('out', masks[j].cpu().numpy()*255)
                #cv2.waitKey(0)
                cal_detect.append([label, xyxy,float(conf)])
    return  cal_detect


def detect():
    model, stride, names, pt = load_model()   # 加载模型
    image = cv2.imread("./images/yaobiao.jpg")   # 读取识别对象
    results = run(model, image, stride, pt)   # 识别， 返回多个数组每个第一个为结果，第二个为坐标位置
    for i in results:
        box = i[1]
        p1, p2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
        cv2.rectangle(image, p1, p2, (0, 0, 255), thickness=3, lineType=cv2.LINE_AA)
        cv2.putText(image, str(i[0]) + ' ' + str(i[2])[:5], (int(box[0]), int(box[1]) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
    image = cv2.resize(image,(0,0),fx=0.4,fy=0.4)
    cv2.imshow('image', image)
    cv2.waitKey(0)


detect()
