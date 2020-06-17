#Code

Code repository for the project _Understanding The Effects Of Distortion On Object Detection Models (Working Title)_ 

### Set Up and Installation

This code can be run on any instance with a GPU but was developed on the following instance sizes
	
- AWS: _g3.4xlarge_

- GCP: _n1-standard-8 (8 vCPUs, 250 GB Memory) with 1 x NVIDIA Tesla K80_


Once the instance is started, please clone the following repo using the below command:

> `git clone https://github.com/amir-jafari/Cloud-Computing.git`

And follow the installation instructions found [here for Ubuntu 18.04](https://github.com/amir-jafari/Cloud-Computing/tree/master/Deep-Learning-Kit-Installation/Shell-Script-Installation/Ubuntu-18.040) and [here for Ubuntu 16.04](https://github.com/amir-jafari/Cloud-Computing/tree/master/Deep-Learning-Kit-Installation/Shell-Script-Installation/Ubuntu-16.04) - credit to Dr. Amir Jafari for the code

Once installed, run the following code:

```

git clone https://github.com/alexjcohen/Capstone.git 

cd Code

pip3 install -r requirements.txt

```

and you should be good to go!



###File Overview
- `analyze_image.py` - code to get the actual bounding boxes for the COCO validation images (currently set to the 100th image by ID

- `data_loaders.py` - file with the validation data loader, AddNoise, and AddBlur classes for use in `gen_predictions.py`

- `evaluate_coco.py` - code to evaluate predicted output on 2017 COCO validation images. Can be run from the command line using `python3 evaluate_coco.py --results-file /path/to/results`. See `python3 evaluate_coco.py --help` for more information

- `gen_predictions.py` - code to generate all 5000 predictions for the COCO validation images and dump results into a json file for use in the `evaluate_coco.py` script. Must currently be run interactively

- `predict_objdet.py` - code to generate and display predicted bounding boxes and classes for a single image for interactive visualization

- `utils.py` - utility code with class labels, coordinate conversion, json conversion, json results parsing, and sample image display functionality 