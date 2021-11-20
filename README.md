# birdClassification
Kaggle class competition on bird classification. Dataset used: [CUB-200-2011 dataset](http://www.vision.caltech.edu/visipedia/CUB-200-2011.html).

## Object recognition and computer vision 2021/2022

### Setup environment
If you are using VisualStudio Code you can run the code inside a development container. The folders for the settings are in [.devcontainer](.devcontainer/).

To install the dependencies, run the following in the folder where [setup.py](setup.py) is stored:
```Bash
pip install -e .[dev]
```
If you want to use the segmentor, then you need to modify the [setup.py](setup.py) file with your cuda version. To get it, you can type: 
```Bash
nvcc --version
```
Then, download the package *detectron2*:
```Bash
pip install -f "https://dl.fbaipublicfiles.com/detectron2/wheels/{YOUR_CUDA_VERSION}/torch1.10/index.html"
```

### Dataset
We will be using a dataset containing 200 different classes of birds adapted from the [CUB-200-2011 dataset](http://www.vision.caltech.edu/visipedia/CUB-200-2011.html).
Download the training/validation/test images from [here](https://www.di.ens.fr/willow/teaching/recvis18orig/assignment3/bird_dataset.zip). The test image labels are not provided.

### Training and validating the models
To train a model, use the following command line:
```Bash 
train -e 1
```
Please use **train --help** to see the different options.

### Evaluating the models on the test set

As the model trains, model checkpoints are saved to files such as `model_x.pth` to the current working directory.
You can take one of the checkpoints and run:

```Bash
generate_submission --model [path of the weights] --output_path [path of the output csv file]
```
That generates a file `kaggle.csv` that you can upload to the private kaggle competition website.

### Segmentation part
To generate segmentation maps from a network, use the following:
```Bash
generate_segmentation --model [name of the model] --path_to_segment [path to the images to segment] --path_maps [path where the maps will be stored]
```

To crop images around birds from maps, use the following:
```Bash
crop_from_map --path_to_crop [path to the images to crop] --path_maps [path to the segmentation maps] --path_crops [path where the crops will be stored]
```

### Acknowledgments
The assignment was inspired from Rob Fergus and Soumith Chintala https://github.com/soumith/traffic-sign-detection-homework.<br/>
It was adaptated done by Gul Varol: https://github.com/gulvarol.
