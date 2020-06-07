#%%
# import packages
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import numpy as np
import argparse
import sys
import json
import time
import copy
from utils import convert_to_wh, loadRes

# TO DO
#   - add arguments for resFile Gt file
#   - add option to subset output

if __name__ == '__main__':

    # load the results file -- see convert_to_json in utils.py for format
    resFile = eval(json.load(open('val2017_pred_notransform.json')))

    # optional - filter for results above certain confidence score
    # resFile = [i for i in resFile if i.get('score') > 0.25]

    # since most models output (x1, y1, x2, y2), convert files to (x1, y1, w, h) for coco object
    # NOTE: NOT NEEDED IF RUNNING convert_xy_to_wh in gen_predictions.py - to be option in arguments
    resFile = convert_to_wh(resFile)

    # set annotation type
    anntype = 'bbox'

    # load ground truth annotations for VAL2017 images
    cocoGt = COCO('Data/annotations/instances_val2017.json')

    # use loadRes to create predicted annotations -- see utils.py for more
    cocoDt = loadRes(resFile, cocoGt)

    # create cocoeval object with ground truth and predicted annotations for bbox
    cocoeval = COCOeval(cocoGt, cocoDt, anntype)

    # evaluate and gather results
    cocoeval.evaluate()
    cocoeval.accumulate()
    cocoeval.summarize()