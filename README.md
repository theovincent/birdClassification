# birdClassification
Kaggle class competition on bird classification. Dataset used: [CUB-200-2011 dataset](http://www.vision.caltech.edu/visipedia/CUB-200-2011.html).

## Object recognition and computer vision 2021/2022

### Setup environment
If you are using VisualStudio Code you can run the code inside a development container. The folders for the settings are in [.devcontainer](.devcontainer/).

To install the dependencies, run the following in the folder where [setup.py](setup.py) is stored:
```Bash
pip install -e .[dev]
```

### Dataset
We will be using a dataset containing 200 different classes of birds adapted from the [CUB-200-2011 dataset](http://www.vision.caltech.edu/visipedia/CUB-200-2011.html).
Download the training/validation/test images from [here](https://www.di.ens.fr/willow/teaching/recvis18orig/assignment3/bird_dataset.zip). The test image labels are not provided.

### Training and validating the models
Run the script `main.py` to train your model.

Modify `main.py`, `model.py` and `data.py` for your assignment, with an aim to make the validation score better.

### Evaluating the models on the test set

As the model trains, model checkpoints are saved to files such as `model_x.pth` to the current working directory.
You can take one of the checkpoints and run:

```
python evaluate.py --data [data_dir] --model [model_file]
```

That generates a file `kaggle.csv` that you can upload to the private kaggle competition website.

### Acknowledgments
The assignment was inspired from Rob Fergus and Soumith Chintala https://github.com/soumith/traffic-sign-detection-homework.<br/>
It was adaptated done by Gul Varol: https://github.com/gulvarol.
