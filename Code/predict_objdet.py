#%%
# import packages
import torch
import cv2
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from PIL import Image
from utils import COCOLabels

# define category names and device
COCO_INSTANCE_CATEGORY_NAMES = COCOLabels().labels
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

confidence = 0.9

# run on example image - hardcoded for now
# load image and convert to tensor
im = cv2.cvtColor(cv2.imread('Data/val2017/000000174482.jpg'), cv2.COLOR_RGB2BGR)

# add any additional transformations here:

# PLACEHOLDER

# turn to image tensor and move to device
img_t = transforms.functional.to_tensor(Image.fromarray(im)).to(device)

# load model
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True).to(device)
model.eval()

# get model output and corresponding info - predicted object classes, boxes, and scores
pred = model([img_t])[0]
pred_class = [COCO_INSTANCE_CATEGORY_NAMES[i] for i in list(pred['labels'].cpu().detach().numpy())]
pred_boxes = [[(int(i[0]), int(i[1])), (int(i[2]), int(i[3]))] for i in list(pred['boxes'].cpu().detach().numpy())]
pred_scores = [float(i) for i in pred['scores'].detach().cpu().numpy()]

# set text size
text_size = 1
text_th = 1

# draw boxes and labels. plot
for i in range(len(pred_class)):
    # for now - only show objects with a score above 0.9 to reduce noise in output
    if pred_scores[i] > confidence:
        cv2.rectangle(im, pred_boxes[i][0], pred_boxes[i][1], (0, 255, 0), 1)
        cv2.putText(im, pred_class[i], pred_boxes[i][0], cv2.FONT_HERSHEY_SIMPLEX, text_size, (0, 255, 0), thickness=1)

plt.imshow(im)
plt.axis('off')
plt.show()





