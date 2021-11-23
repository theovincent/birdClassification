# Bird classification - Object recognition and computer vision 2021/2022
Kaggle class competition on bird classification. Dataset used: [CUB-200-2011 dataset](http://www.vision.caltech.edu/visipedia/CUB-200-2011.html).

## Setup environment
If you are using VisualStudio Code you can run the code inside a development container. The folders for the settings are in [.devcontainer](.devcontainer/).

To install the dependencies, run the following in the folder where [setup.py](setup.py) is stored:
```Bash
pip install -e .[dev]
```
If you want to use the segmentor, then you need to modify the [setup.py](setup.py) file with your cuda version. To get it, you can type: 
```Bash
nvcc --version
```
Then, download the package *detectron2* by typing:
```Bash
pip install detectron2 -f "https://dl.fbaipublicfiles.com/detectron2/wheels/{YOUR_CUDA_VERSION}/torch1.10/index.html"
```

## Datasets
We will be using a dataset containing 200 different classes of birds adapted from the [CUB-200-2011 dataset](http://www.vision.caltech.edu/visipedia/CUB-200-2011.html).
Download the training/validation/test images from [here](https://www.di.ens.fr/willow/teaching/recvis18orig/assignment3/bird_dataset.zip). The test image labels are not provided.

To pretrain the networks, the following datasets have also been used:
- [butterfly dataset for classification](https://www.kaggle.com/gpiosenka/butterfly-images40-species).
- [monkey dataset for classification](https://www.kaggle.com/slothkong/10-monkey-species).


## Training and validating the models
To train a model, use the following command line:
```Bash 
train --help
```

## Evaluating the models on the test set
As the model trains, model checkpoints are saved to files such as `model_x.pth` to the current working directory.
You can take one of the checkpoints and run:

```Bash
generate_submission --help
```
That generates a .cvs file that you can upload to the private kaggle competition website.

## Segmentation part
To generate segmentation maps from a network, use the following:
```Bash
generate_segmentation --help
```

To crop images around birds from maps, use the following:
```Bash
crop_from_map --help
```

## Acknowledgments
The assignment was inspired from Rob Fergus and Soumith Chintala https://github.com/soumith/traffic-sign-detection-homework.<br/>
It was adaptated done by Gul Varol: https://github.com/gulvarol.

## Folder and file organisation
```
📦birdClassification
 ┣ 📂.devcontainer
 ┃ ┣ 📜Dockerfile
 ┃ ┗ 📜devcontainer.json
 ┣ 📂bird_dataset
 ┃ ┣ 📂raw_images
 ┃ ┃ ┣ 📂train_images
 ┃ ┃ ┃ ┣ 📂class_1
 ┃ ┃ ┃ ┃ ┣ 📜img1.jpg
 ┃ ┃ ┃ ┃ ┗ 📜...jpg
 ┃ ┃ ┃ ┗ 📂class_...
 ┃ ┃ ┣ 📂val_images
 ┃ ┃ ┃ ┣ 📂class_1
 ┃ ┃ ┃ ┃ ┣ 📜img1.jpg
 ┃ ┃ ┃ ┃ ┗ 📜...jpg
 ┃ ┃ ┃ ┗ 📂class_...
 ┃ ┃ ┗ 📂test_images
 ┃ ┃   ┗ 📂mistery_category
 ┃ ┃     ┣ 📜img1.jpg
 ┃ ┃     ┗ 📜...jpg
 ┃ ┗ 📂[folder corresponding to another specific idea]
 ┃   ┗ 📂...
 ┣ 📂birdClassification.egg-info
 ┃ ┣ 📜PKG-INFO
 ┃ ┣ 📜SOURCES.txt
 ┃ ┣ 📜dependency_links.txt
 ┃ ┣ 📜entry_points.txt
 ┃ ┣ 📜requires.txt
 ┃ ┗ 📜top_level.txt
 ┣ 📂classifier
 ┃ ┣ 📜__init__.py
 ┃ ┣ 📜compute_normalization_coefficients.py
 ┃ ┣ 📜generate_submission.py
 ┃ ┣ 📜generate_submission_vote.py
 ┃ ┣ 📜get_best_model.py
 ┃ ┣ 📜gpu_check.py
 ┃ ┣ 📜loader.py
 ┃ ┣ 📜loss.py
 ┃ ┣ 📜mistakes_frequencies.ipynb
 ┃ ┣ 📜model.py
 ┃ ┣ 📜study_mistakes.py
 ┃ ┣ 📜train.py
 ┃ ┣ 📜validation.py
 ┃ ┣ 📜visualize_data.ipynb
 ┃ ┗ 📜visualize_losses.ipynb
 ┣ 📂output
 ┃ ┣ 📂segmentation_from_gt
 ┃ ┃ ┣ 📜resnet.feather
 ┃ ┃ ┗ 📜resnet_41.pth
 ┃ ┣ 📂[folder corresponding to another specific idea]
 ┃ ┃ ┗ 📜...pth
 ┃ ┗ 📂submission
 ┃   ┣ 📜[files to submit to kaggle].csv
 ┃   ┗ 📜....csv
 ┣ 📂scripts
 ┃ ┣ 📜run_training.sh
 ┃ ┣ 📜run_training_4D.sh
 ┃ ┣ 📜run_training_animals.sh
 ┃ ┗ 📜run_warm_start_training.sh
 ┣ 📂segmentor
 ┃ ┣ 📜crop_from_map.py
 ┃ ┣ 📜generate_segmentation.py
 ┃ ┣ 📜loader.py
 ┃ ┣ 📜model.py
 ┃ ┗ 📜restructure_segmentation_gt.ipynb
 ┣ 📜.gitignore
 ┣ 📜LICENSE
 ┣ 📜README.md
 ┣ 📜github_colab_token.txt [to connect google colab with the github account]
 ┗ 📜setup.py
```