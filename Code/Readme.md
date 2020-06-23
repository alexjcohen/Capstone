# Code

Code repository for the project _Understanding The Effects Of Distortion On Object Detection Models (Working Title)_ 

### Set Up and Installation

This code can be run on any instance with a GPU but was developed on the following instance sizes:
	
- AWS: _g3.4xlarge_

- GCP: _n1-standard-8 (8 vCPUs, 250 GB Memory) with 1 x NVIDIA Tesla K80_


Once the instance is started, please clone the following repo using the below command to set up the CUDA GPU and install the main machine learning packages used during this project:

> `git clone https://github.com/amir-jafari/Cloud-Computing.git`

And follow the installation instructions found [here](https://github.com/amir-jafari/Cloud-Computing/tree/master/Deep-Learning-Kit-Installation/Shell-Script-Installation), where you can find instructions for your specific version of Ubuntu (I used 18.04) 
- _credit to Dr. Amir Jafari for the above code_

Once installed, run the following code:

```

git clone https://github.com/alexjcohen/Capstone.git; 

cd Code;

pip3 install -r requirements.txt; # <-- under development

```

and you should be good to go!


### Data
The data used for this project comes from the Common Objects in Context dataset, and more information can be found [here](http://cocodataset.org/#home). The data used comes from the 2017 set of images, and can be downloaded using the following commands after completing the installation instructions above and cloning/navigating from this repository:

```
sudo apt-get install zip;
mkdir Data; cd Data;
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip; unzip annotations_trainval2017.zip;
wget http://images.cocodataset.org/zips/train2017.zip; unzip train2017.zip
wget http://images.cocodataset.org/zips/val2017.zip; unzip val2017.zip;
rm annotations_trainval2017.zip; rm train2017.zip; val2017.zip;
cd ..;
```


### File Overview
- `analyze_image.py` - code to get the actual bounding boxes for the COCO validation images (currently set to the 100th image by ID)

- `data_loaders.py` - file with the validation and training data loaders (with/without resizing and batch loading), AddNoise, and AddBlur classes for use in `gen_predictions.py` and `train_fasterRCNN_distort.py`

- `evaluate_coco.py` - code to evaluate predicted output on 2017 COCO validation images. Can be run from the command line using `python3 evaluate_coco.py --results-file /path/to/results`. See `python3 evaluate_coco.py --help` for more information

- `gen_predictions.py` - code to generate all 5000 predictions for the COCO validation images and dump results into a json file for use in the `evaluate_coco.py` script. Must currently be run interactively

- `image_resize.py` - code to find the minimum/maximum widths and heights in the underlying dataset to understand requirements for resizing images during model training. See `python3 image_resize.py --help` for more information

- `predict_objdet.py` - code to generate and display predicted bounding boxes and classes for a single image for interactive visualization

- `train_fasterRCNN_distory.py` - code to train the pytorch Faster R-CNN implementation on distorted data to (hopefully) mitigate distortion effects (under development)

- `utils.py` - utility code with class labels, coordinate conversion, json conversion, json results parsing, and sample image display functionality 
