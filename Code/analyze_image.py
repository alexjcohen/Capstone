#%%
# import packages
from pycocotools.coco import COCO
import cv2
import matplotlib.pyplot as plt
from PIL import Image

# create COCO object from annotation file
annfile = 'Data/annotations/instances_val2017.json'
coco = COCO(annfile)


# function to get image ids
def get_image_ids(sample=None):
    # get all category ids
    catIds = coco.getCatIds()
    imgIds = []
    # appending image ids of each category
    for ids in catIds:
        imgIds += coco.getImgIds(catIds=ids)

    if sample is None:
        # Returning the entire dataset
        return imgIds
    else:
        # Returning a subset of the dataset
        return imgIds[:sample]


# function to load image info
def load_image_node(ids):

    # get image id
    image_node = coco.loadImgs(ids)[0]

    return image_node


# function to get bounding box coordinates given an image id
def get_bbox_coords(id):
    annIds = coco.getAnnIds(imgIds=id, iscrowd=None)
    anns = coco.loadAnns(annIds)
    coordsList = []
    for a in anns:
        coordsList.append(a['bbox'])

    return coordsList

# load sample image
id1 = get_image_ids()[100]

# get image information and bbox coordinates
node = load_image_node(id1)
bbox = get_bbox_coords(id1)

# create a list of annotations
annids = coco.getAnnIds(imgIds=id1)
anns = coco.loadAnns(annids)

# load image and convert to BGR
test_img = Image.open('Data/val2017/' + node.get('file_name'))
img_cv2 = cv2.cvtColor(cv2.imread('Data/val2017/' + node.get('file_name')), cv2.COLOR_RGB2BGR)

# for each bounding box:
for i in range(len(bbox)):

    # get starting and ending points for rectangle
    startpt = (int(bbox[i][0]), int(bbox[i][1]))
    endpt = (int(bbox[i][2] + bbox[i][0]), int(bbox[i][3] + bbox[i][1]))

    # get category label for bbox
    label = anns[i]['category_id']
    entity = coco.loadCats(label)[0]['name']

    # place rectange and text on image
    cv2.rectangle(img_cv2, startpt, endpt, (0, 255, 0), 2)
    cv2.putText(img_cv2, entity, startpt, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

# show image 
plt.imshow(img_cv2)
plt.title('Ground Truth')
plt.axis('off')
plt.show()
