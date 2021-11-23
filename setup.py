from setuptools import setup

CUDA_VERSION = "cu102"

setup(
    name="birdClassification",
    version="0.1",
    description="Classifies bird from the Caltech-UCSD Birds-200-2011 dataset.",
    packages=["classifier", "segmentor"],
    requires=["setuptools", "wheel"],
    dependency_links=[
        "https://dl.fbaipublicfiles.com/detectron2/wheels/{CUDA_VERSION}/torch1.10/index.html",
    ],
    install_requires=[
        "torch==1.10",
        "torchvision",
        "tqdm",
        "opencv-python",
        "pandas",
        "ipywidgets",
        "pyarrow",
    ],
    extras_require={
        "dev": ["tqdm", "ipykernel", "black"],
    },
    entry_points={
        "console_scripts": [
            # For the classifier
            "compute_normalization_coefficients=classifier.compute_normalization_coefficients:compute_normalization_coefficients_cli",
            "train=classifier.train:train_cli",
            "study_mistakes=classifier.study_mistakes:study_mistakes_cli",
            "get_best_model=classifier.get_best_model:get_best_model_cli",
            "generate_submission=classifier.generate_submission:generate_submission_cli",
            "generate_submission_vote=classifier.generate_submission_vote:generate_submission_vote_cli",
            # For the segmentor
            "generate_segmentation=segmentor.generate_segmentation:generate_segmentation_cli",
            "crop_from_map=segmentor.crop_from_map:crop_from_map_cli",
        ]
    },
)
