#<=============IMPORT STMTS===============>
import os
import sys
import time
import numpy as np
from tqdm.autonotebook import tqdm
import argparse

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import math
import yaml
import json

from torch import nn
import torch.nn.functional as F

import torchvision
from torchvision import datasets
import torchvision.transforms as transforms
from torch.ao.quantization import get_default_qconfig, QConfigMapping
from torch.ao.quantization.quantize_fx import prepare_fx, convert_fx, fuse_fx

import torch.nn as nn
import torch
from torchvision.ops.boxes import nms as nms_torch
from torch.ao.quantization import QuantStub, DeQuantStub
from efficientdet.utils import BBoxTransform, ClipBoxes
from utils.utils import preprocess, invert_affine, postprocess, boolean_string
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from efficientnet import EfficientNet 
from efficientnet.utils import MemoryEfficientSwish, Swish
from efficientdet.dataset import CocoDataset, Resizer, Normalizer, Augmenter, collater
from utils.utils import replace_w_sync_bn, CustomDataParallel, get_last_weights, init_weights, boolean_string
from efficientdet.model import BiFPN, Regressor, Classifier, EfficientNet
from efficientdet.utils import Anchors
from efficientnet.utils_extra import Conv2dStaticSamePadding, MaxPool2dStaticSamePadding

from torchvision.datasets import CocoDetection
from torch.nn.functional import interpolate

# Set up warnings
import warnings
warnings.filterwarnings(
    action='ignore',
    category=DeprecationWarning,
    module=r'.*'
)
warnings.filterwarnings(
    action='default',
    module=r'torch.ao.quantization'
)

# Specify random seed for repeatable results
torch.manual_seed(191009)


#<---------MODEL ARCHITECTURE----------->
class SeparableConvBlock(nn.Module):
    """
    created by Zylo117
    """

    def __init__(self, in_channels, out_channels=None, norm=True, activation=False, onnx_export=False):
        super(SeparableConvBlock, self).__init__()
        if out_channels is None:
            out_channels = in_channels

        # Q: whether separate conv
        #  share bias between depthwise_conv and pointwise_conv
        #  or just pointwise_conv apply bias.
        # A: Confirmed, just pointwise_conv applies bias, depthwise_conv has no bias.
        
        self.depthwise_conv = Conv2dStaticSamePadding(in_channels, in_channels,
                                                      kernel_size=3, stride=1, groups=in_channels, bias=False)
        self.pointwise_conv = Conv2dStaticSamePadding(in_channels, out_channels, kernel_size=1, stride=1)

        self.norm = norm
        if self.norm:
            # Warning: pytorch momentum is different from tensorflow's, momentum_pytorch = 1 - momentum_tensorflow
            self.bn = nn.BatchNorm2d(num_features=out_channels, momentum=0.01, eps=1e-3)

        self.activation = activation
        if self.activation:
            self.swish = MemoryEfficientSwish() if not onnx_export else Swish()
        
        

    def forward(self, x):
        
        x = self.depthwise_conv(x)
        x = self.pointwise_conv(x)

        if self.norm:
            x = self.bn(x)

        if self.activation:
            x = self.swish(x)
       
        return x

class EfficientDetBackbone(nn.Module):
    def __init__(self, num_classes=80, compound_coef=0, load_weights=False, **kwargs):
        super(EfficientDetBackbone, self).__init__()
        
        self.compound_coef = compound_coef

        self.backbone_compound_coef = [0, 1, 2, 3, 4, 5, 6, 6, 7]
        self.fpn_num_filters = [64, 88, 112, 160, 224, 288, 384, 384, 384]
        self.fpn_cell_repeats = [3, 4, 5, 6, 7, 7, 8, 8, 8]
        self.input_sizes = [512, 640, 768, 896, 1024, 1280, 1280, 1536, 1536]
        self.box_class_repeats = [3, 3, 3, 4, 4, 4, 5, 5, 5]
        self.pyramid_levels = [5, 5, 5, 5, 5, 5, 5, 5, 6]
        self.anchor_scale = [4., 4., 4., 4., 4., 4., 4., 5., 4.]
        self.aspect_ratios = kwargs.get('ratios', [(1.0, 1.0), (1.4, 0.7), (0.7, 1.4)])
        self.num_scales = len(kwargs.get('scales', [2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)]))
        conv_channel_coef = {
            # the channels of P3/P4/P5.
            0: [40, 112, 320],
            1: [40, 112, 320],
            2: [48, 120, 352],
            3: [48, 136, 384],
            4: [56, 160, 448],
            5: [64, 176, 512],
            6: [72, 200, 576],
            7: [72, 200, 576],
            8: [80, 224, 640],
        }

        num_anchors = len(self.aspect_ratios) * self.num_scales

        self.bifpn = nn.Sequential(
            *[BiFPN(self.fpn_num_filters[self.compound_coef],
                    conv_channel_coef[compound_coef],
                    True if _ == 0 else False,
                    attention=True if compound_coef < 6 else False,
                    use_p8=compound_coef > 7)
              for _ in range(self.fpn_cell_repeats[compound_coef])])

        self.num_classes = num_classes
        self.regressor = Regressor(in_channels=self.fpn_num_filters[self.compound_coef], num_anchors=num_anchors,
                                   num_layers=self.box_class_repeats[self.compound_coef],
                                   pyramid_levels=self.pyramid_levels[self.compound_coef])
        self.classifier = Classifier(in_channels=self.fpn_num_filters[self.compound_coef], num_anchors=num_anchors,
                                     num_classes=num_classes,
                                     num_layers=self.box_class_repeats[self.compound_coef],
                                     pyramid_levels=self.pyramid_levels[self.compound_coef])

        self.anchors = Anchors(anchor_scale=self.anchor_scale[compound_coef],
                               pyramid_levels=(torch.arange(self.pyramid_levels[self.compound_coef]) + 3).tolist(),
                               **kwargs)

        self.backbone_net = EfficientNet(self.backbone_compound_coef[compound_coef], load_weights)
        
    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def forward(self, inputs):
        

        max_size = inputs.shape[-1]

        _, p3, p4, p5 = self.backbone_net(inputs)

        features = (p3, p4, p5)
        

        features = self.bifpn(features)

        regression = self.regressor(features)
        classification = self.classifier(features)
        anchors = self.anchors(inputs, inputs.dtype)
        

        return features , regression, classification, anchors

#<===========HELPER FUNCTIONS===============>

ap = argparse.ArgumentParser()
ap.add_argument('-p', '--project', type=str, default='coco', help='project file that contains parameters')
ap.add_argument('-c', '--compound_coef', type=int, default=0, help='coefficients of efficientdet')
ap.add_argument('-w', '--weights', type=str, default=None, help='/path/to/weights')
ap.add_argument('--nms_threshold', type=float, default=0.5, help='nms threshold, don\'t change it if not for testing purposes')
ap.add_argument('--cuda', type=boolean_string, default=True)
ap.add_argument('--device', type=int, default=0)
ap.add_argument('--float16', type=boolean_string, default=False)
ap.add_argument('--override', type=boolean_string, default=True, help='override previous bbox results file if exists')
args = ap.parse_args()

compound_coef = args.compound_coef
nms_threshold = args.nms_threshold
use_cuda = args.cuda
gpu = args.device
use_float16 = args.float16
override_prev_results = args.override
project_name = args.project
weights_path = f'/content/FX_QAT_Efficientdet/weights/efficientdet-d0.pth' if args.weights is None else args.weights

#print(f'running coco-style evaluation on project {project_name}, weights {weights_path}...')

params = yaml.safe_load(open(f'/content/FX_QAT_Efficientdet/projects/coco.yml'))
obj_list = params['obj_list']

input_sizes = [512, 640, 768, 896, 1024, 1280, 1280, 1536, 1536]


def evaluate_coco(img_path, set_name, image_ids, coco, model, threshold=0.05):
    results = []

    regressBoxes = BBoxTransform()
    clipBoxes = ClipBoxes()

    for image_id in tqdm(image_ids):
        image_info = coco.loadImgs(image_id)[0]
        image_path = img_path + image_info['file_name']

        ori_imgs, framed_imgs, framed_metas = preprocess(image_path, max_size=input_sizes[compound_coef], mean=params['mean'], std=params['std'])
        x = torch.from_numpy(framed_imgs[0])

        if use_cuda:
            x = x.cuda()
            if use_float16:
                x = x.half()
            else:
                x = x.float()
        else:
            x = x.float()

        x = x.unsqueeze(0).permute(0, 3, 1, 2)
        features, regression, classification, anchors = model(x)

        preds = postprocess(x,
                            anchors, regression, classification,
                            regressBoxes, clipBoxes,
                            threshold, nms_threshold)
        
        if not preds:
            continue

        preds = invert_affine(framed_metas, preds)[0]

        scores = preds['scores']
        class_ids = preds['class_ids']
        rois = preds['rois']

        if rois.shape[0] > 0:
            # x1,y1,x2,y2 -> x1,y1,w,h
            rois[:, 2] -= rois[:, 0]
            rois[:, 3] -= rois[:, 1]

            bbox_score = scores

            for roi_id in range(rois.shape[0]):
                score = float(bbox_score[roi_id])
                label = int(class_ids[roi_id])
                box = rois[roi_id, :]

                image_result = {
                    'image_id': image_id,
                    'category_id': label + 1,
                    'score': float(score),
                    'bbox': box.tolist(),
                }

                results.append(image_result)

    if not len(results):
        raise Exception('the model does not provide any valid output, check model architecture and the data input')
    
    filepath = f'{set_name}_bbox_results.json'
    if os.path.exists(filepath):
        os.remove(filepath)
    json.dump(results, open(filepath, 'w'), indent=4)

def _eval(coco_gt, image_ids, pred_json_path):
    # load results in COCO evaluation tool
    coco_pred = coco_gt.loadRes(pred_json_path)

    # run COCO evaluation
    print('BBox')
    coco_eval = COCOeval(coco_gt, coco_pred, 'bbox')
    coco_eval.params.imgIds = image_ids
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()


class Params:
    def __init__(self, project_file):
        self.params = yaml.safe_load(open(project_file).read())

    def __getattr__(self, item):
        return self.params.get(item, None)
  
def load_model(model_file):
    params = Params(f'/content/FX_QAT_Efficientdet/projects/coco.yml')
    model = EfficientDetBackbone(num_classes=len(params.obj_list),compound_coef=0,ratios=eval(params.anchors_ratios), scales=eval(params.anchors_scales))
    model.load_state_dict(torch.load(model_file, map_location=torch.device('cpu')))
    model.requires_grad_(False)
    return model

def print_size_of_model(model):
    torch.save(model.state_dict(), "temp.p")
    print('Size (MB):', os.path.getsize("temp.p")/1e6)
    os.remove('temp.p')

#<===============DATALOADER AND DATASETS=====================>

def prepare_data_loaders(data_path):
    project_name= "coco"  # also the folder name of the dataset that under data_path folder
    train_set= "train2017"
    val_set= "val2017"
    training_params = {'batch_size': 30,
                       'shuffle': True,
                       'drop_last': True,
                       'collate_fn': collater,
                       'num_workers': 0}

    val_params = {'batch_size': 50,
                  'shuffle': False,
                  'drop_last': True,
                  'collate_fn': collater,
                  'num_workers': 0}

    input_sizes = [512, 640, 768, 896, 1024, 1280, 1280, 1536, 1536]
    compound_coef = 0
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
    dataset = CocoDataset(data_path, set= train_set, transform=transforms.Compose([Normalizer(mean=mean, std=std),Augmenter(),Resizer(input_sizes[compound_coef])]))
    dataset_test = CocoDataset(data_path,set=val_set , transform=transforms.Compose([Normalizer(mean=mean, std=std),Resizer(input_sizes[compound_coef])]))

    data_loader = DataLoader(dataset, **training_params)

    data_loader_test = DataLoader(dataset_test, **val_params)

    return data_loader, data_loader_test

data_path = '/content/FX_QAT_Efficientdet/datasets/coco'
saved_model_dir = '/content/FX_QAT_Efficientdet/weights'
float_model_file = '/efficientdet-d0.pth'
scripted_float_model_file = 'efficientdet_quantization_scripted.pth'
scripted_quantized_model_file = 'efficientdet_quantization_scripted_quantized.pth'
SET_NAME = params['val_set']
VAL_GT = f'/content/FX_QAT_Efficientdet/datasets/coco/annotations/instances_val2017.json'
VAL_IMGS = f'/content/FX_QAT_Efficientdet/datasets/coco/val2017/'
MAX_IMAGES = 10000
coco_gt = COCO(VAL_GT)
image_ids = coco_gt.getImgIds()[:MAX_IMAGES]

data_loader, data_loader_test = prepare_data_loaders(data_path)

criterion = nn.CrossEntropyLoss()
float_model = load_model(saved_model_dir+float_model_file).to('cpu')

if use_cuda:
      float_model.cuda()

      if use_float16:
        float_model.half()
   
float_model.eval()

model_to_quantize = load_model(saved_model_dir+float_model_file).to('cpu')

model_to_quantize.eval()

# The old 'fbgemm' is still available but 'x86' is the recommended default.
qconfig = get_default_qconfig("x86")
qconfig_mapping = QConfigMapping().set_global(qconfig)


prepared_model = prepare_fx(model_to_quantize, qconfig_mapping, image_ids)

print(prepared_model.graph)

def calibrate(model, data_loader):
    model.eval()
    with torch.no_grad():
        for image, target in data_loader:
            model(image)
calibrate(prepared_model, data_loader_test)  # run calibration on sample data

quantized_model = convert_fx(prepared_model)
print(quantized_model)

print("Size of model before quantization")
print_size_of_model(float_model)
print("Size of model after quantization")
print_size_of_model(quantized_model)
if override_prev_results or not os.path.exists(f'{SET_NAME}_bbox_results.json'):
      evaluate_coco(VAL_IMGS, SET_NAME, image_ids, coco_gt, float_model)
    

_eval(coco_gt, image_ids, f'{SET_NAME}_bbox_results.json')

fx_graph_mode_model_file_path = saved_model_dir + "efficientdet_fx_graph_mode_quantized.pth"

# this does not run due to some erros loading convrelu module:
# ModuleAttributeError: 'ConvReLU2d' object has no attribute '_modules'
# save the whole model directly
# torch.save(quantized_model, fx_graph_mode_model_file_path)
# loaded_quantized_model = torch.load(fx_graph_mode_model_file_path)

# save with state_dict
# torch.save(quantized_model.state_dict(), fx_graph_mode_model_file_path)
# import copy
# model_to_quantize = copy.deepcopy(float_model)
# prepared_model = prepare_fx(model_to_quantize, {"": qconfig})
# loaded_quantized_model = convert_fx(prepared_model)
# loaded_quantized_model.load_state_dict(torch.load(fx_graph_mode_model_file_path))

# save with script
torch.jit.save(torch.jit.script(quantized_model), fx_graph_mode_model_file_path)
loaded_quantized_model = torch.jit.load(fx_graph_mode_model_file_path)

if override_prev_results or not os.path.exists(f'{SET_NAME}_bbox_results.json'):
      evaluate_coco(VAL_IMGS, SET_NAME, image_ids, coco_gt, float_model)
    

_eval(coco_gt, image_ids, f'{SET_NAME}_bbox_results.json')
