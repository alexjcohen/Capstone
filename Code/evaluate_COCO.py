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
#   - add option to subset output

# argument parser for input directories
parser = argparse.ArgumentParser(description='Evaluation script using the pycocotools library')

parser.add_argument('--results-file', help='location of model output json file, see gen_predictions.py',
                    default='val2017_pred_notransform.json')

parser.add_argument('--truth-file', help='location of ground truth file',
                    default='Data/annotations/instances_val2017.json')

parser.add_argument('--anntype', help='type of annotations (currently only supports bbox)',
                    default='bbox')

parser.add_argument('--convert-xy', help='Convert bounding box coordinates from (x1,y1)(x2,y2) to (x1,x2)(w,h)',
                    default=False, action='store_true')

if __name__ == '__main__':

    args = parser.parse_args()

    if args.anntype is not 'bbox':
        sys.exit('ERROR: ANNYTYPE MUST BE "BBOX" IN CURRENT IMPLEMENTATION')

    # load the results file -- see convert_to_json in utils.py for format
    resFile = eval(json.load(open(args.results_file)))

    # optional - filter for results above certain confidence score
    # resFile = [i for i in resFile if i.get('score') > 0.25]

    # since most models output (x1, y1, x2, y2), convert files to (x1, y1, w, h) for coco object
    # NOTE: NOT NEEDED IF RUNNING convert_xy_to_wh in gen_predictions.py - to be option in arguments
    if args.convert_xy:
        print('Converting image dimensions')
        resFile = convert_to_wh(resFile)
        print('Converted!')

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
