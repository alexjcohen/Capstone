# class for storing COCO class labels for easy importing
class COCOLabels:
    """
    class to store COCO category labels
    """
    def __init__(self):
        self.labels = ['__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
                        'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
                        'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
                        'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
                        'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
                        'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
                        'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
                        'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
                        'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
                        'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
                        'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
                        'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']


def convert_xy_to_wh(output_dict):
    """
    Covnvert XY coordinates of predicted bbox to W,H coordinates for COCO evaluator
    :param output_dict: output dictionary being created in the results_to_json code
    :return: output_dict with updated coordinates
    """
    output_dict['bbox'][2] = output_dict['bbox'][2] - output_dict['bbox'][0]
    output_dict['bbox'][3] = output_dict['bbox'][3] - output_dict['bbox'][1]
    return output_dict


def results_to_json(pred, img_id, json_dict=None):
    """
    Turn predicted object detection output to dictionary for json dump
    :param pred: predicted output from object detection model with keys 'labels', 'boxes', and 'score'
    :param img_id: image id from coco dataset, passed manually
    :param json_dict: option to pass existing json dictionary (if looping through 2+ predictions)
    :return: return completed json dictionary
    """
    import numpy as np

    for i in range(len(pred['boxes'])):
        output_dict = {'image_id': img_id}
        output_dict['category_id'] = int(pred['labels'][i].detach().cpu().numpy())
        output_dict['bbox'] = list(pred['boxes'][i].detach().cpu().numpy())
        output_dict = convert_xy_to_wh(output_dict)
        output_dict['score'] = np.round(float(pred['scores'][i].detach().cpu().numpy()), 6)
        if json_dict:
            json_dict.append(output_dict)
        else:
            json_dict = [output_dict]

    return json_dict


def convert_to_wh(resFile):
    """
    function to convert all bounding boxes in resFile from (x1, y1, x2, y2) to (x1, y1, w, h) for cocoEval
    :param resFile: input results file
    :return resFile (list): convert resolution file
    """
    import numpy as np
    for i, entry in enumerate(resFile):
        resFile[i].get('bbox')[2] = np.round(resFile[i].get('bbox')[2] - resFile[i].get('bbox')[0], 5)
        resFile[i].get('bbox')[3] = np.round(resFile[i].get('bbox')[3] - resFile[i].get('bbox')[1], 5)
    return resFile


def loadRes(resFile, cocoObj):
    """
    Load result file and return a result api object. Adapted from existing pycocotools COCO class, but updated for
    python3 instead of python2
    :param   resFile (str)     : file name of result file
    :param   cocoObj (obj)     : coco ground truth object
    :return: res (obj)         : result api object
    """

    import sys
    from pycocotools.coco import COCO
    import time
    import json
    import numpy as np
    import copy

    PYTHON_VERSION = sys.version_info[0]

    res = COCO()
    res.dataset['images'] = [img for img in cocoObj.dataset['images']]

    print('Loading and preparing results...')
    tic = time.time()
    if type(resFile) == str or (PYTHON_VERSION == 2 and type(resFile) == bytes):
        anns = json.load(open(resFile))
    elif type(resFile) == np.ndarray:
        anns = cocoObj.loadNumpyAnnotations(resFile)
    else:
        anns = resFile
    assert type(anns) == list, 'results in not an array of objects'
    annsImgIds = [ann['image_id'] for ann in anns]
    assert set(annsImgIds) == (set(annsImgIds) & set(cocoObj.getImgIds())), \
        'Results do not correspond to current coco set'
    if 'caption' in anns[0]:
        imgIds = set([img['id'] for img in res.dataset['images']]) & set([ann['image_id'] for ann in anns])
        res.dataset['images'] = [img for img in res.dataset['images'] if img['id'] in imgIds]
        for id, ann in enumerate(anns):
            ann['id'] = id + 1
    elif 'bbox' in anns[0] and not anns[0]['bbox'] == []:
        res.dataset['categories'] = copy.deepcopy(cocoObj.dataset['categories'])
        for id, ann in enumerate(anns):
            bb = ann['bbox']
            x1, x2, y1, y2 = [bb[0], bb[0] + bb[2], bb[1], bb[1] + bb[3]]
            if not 'segmentation' in ann:
                ann['segmentation'] = [[x1, y1, x1, y2, x2, y2, x2, y1]]
            ann['area'] = bb[2] * bb[3]
            ann['id'] = id + 1
            ann['iscrowd'] = 0
    # elif 'segmentation' in anns[0]:
    #     res.dataset['categories'] = copy.deepcopy(cocoObj.dataset['categories'])
    #     for id, ann in enumerate(anns):
    #         # now only support compressed RLE format as segmentation results
    #         ann['area'] = maskUtils.area(ann['segmentation'])
    #         if not 'bbox' in ann:
    #             ann['bbox'] = maskUtils.toBbox(ann['segmentation'])
    #         ann['id'] = id + 1
    #         ann['iscrowd'] = 0
    elif 'keypoints' in anns[0]:
        res.dataset['categories'] = copy.deepcopy(cocoObj.dataset['categories'])
        for id, ann in enumerate(anns):
            s = ann['keypoints']
            x = s[0::3]
            y = s[1::3]
            x0, x1, y0, y1 = np.min(x), np.max(x), np.min(y), np.max(y)
            ann['area'] = (x1 - x0) * (y1 - y0)
            ann['id'] = id + 1
            ann['bbox'] = [x0, y0, x1 - x0, y1 - y0]
    print('DONE (t={:0.2f}s)'.format(time.time() - tic))

    res.dataset['annotations'] = anns
    res.createIndex()
    return res


def get_sample_image(loader, filename, iters=4):
    """
    Function to get sample image to show transformations
    :param loader: data loader object (with transformations applied)
    :param filename: output filename
    :param iters: number of iterations to get to destination image (default is 4 for bicycle)
    :return: saves image file to specified directory
    """
    # get maptplotlib
    import matplotlib.pyplot as plt

    # define iterator
    it = iter(loader)

    # loop through iterator n times to reach target image
    for i in range(iters):
        next(it)

    # extract image tensor
    img = next(it)['img']

    # show image after permutation
    plt.imshow(img.detach().cpu().squeeze().permute(1, 2, 0))
    plt.axis("off")

    # save image and display in console
    plt.savefig(filename)
    plt.show()
    print('Image saved')
