#%%
# Load packages
import torch
import torchvision
from torchvision import transforms
import json
from utils import COCOLabels, results_to_json
from data_loaders import ValImageLoader, AddNoise, AddBlur
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import warnings

warnings.filterwarnings("ignore")

# TO-DO:
# - add argparse for transform_type and degree, output_path, image path, etc.

if __name__ == '__main__':
    # set category labels and device
    COCO_INSTANCE_CATEGORY_NAMES = COCOLabels().labels
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # set parameters
    transform_type = 'blur'
    degree = 1

    # set output file name
    output_path = 'output_json_revised/'
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    output_file = 'val2017_pred_' + transform_type + str(degree).zfill(2) + '.json'

    output_file = output_path + output_file

    # FUTURE - error handling to be used inside __main__
    if os.path.isfile(output_file):
        # sys.exit('File already exists, please change filename to avoid overwrites')
        print('File already exists, please change filename to avoid overwrites')

    # create dataset object for validation images
    if transform_type == 'blur':
        val = ValImageLoader('Data/val2017',
                             'Data/annotations/instances_val2017.json',
                             device, transform=transforms.Compose([
                                AddBlur(radius=degree)
                             ]))

    else:
        val = ValImageLoader('Data/val2017',
                             'Data/annotations/instances_val2017.json',
                             device,
                             transform=transforms.Compose([
                                 AddNoise(mean=0, std=degree, seed=0)
                             ]))

    # create data loader
    loader = DataLoader(val)

    # load model and set to evaluate
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True).to(device)
    model.eval()

    # set json placeholder
    compiled_json = None

    print('Evaluating images')

    # for data loader:
    for i, data in enumerate(tqdm(loader)):

        # make prediction
        out = model(data['img'])[0]

        # if existing json object:
        if compiled_json:
            # convert output to json and append to json object -- see utils.py
            compiled_json = results_to_json(out, int(data['id']), compiled_json)

        else:
            # overwrite compiled_json=None -- see utils.py
            compiled_json = results_to_json(out, int(data['id']))

    # write json to json file at provided filepath
    with open(output_file, 'w') as outfile:
        json.dump(str(compiled_json), outfile)

    print('Output saved!')
