from setuptools import setup

setup(
    name="birdClassification",
    version="0.1",
    description="Classifies bird from the Caltech-UCSD Birds-200-2011 dataset.",
    packages=["classifier"],
    requires=["setuptools", "wheel"],
    install_requires=["torch", "torchvision", "tqdm"],
    extras_require={
        "dev": ["tqdm", "ipykernel", "black"],
    },
    entry_points={
        "console_scripts": [
            "train=classifier.train:train_cli",
            "generate_submission=classifier.generate_submission:generate_submission_cli",
        ]
    },
)
