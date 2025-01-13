from setuptools import setup, find_packages

setup(
    name='chexpert_advanced',
    version='1.0.0',
    description='Advanced chest X-ray analysis system using deep learning',
    packages=find_packages(),
    install_requires=[
        'torch==1.9.0',
        'torchvision==0.10.0',
        'efficientnet-pytorch==0.7.1',
        'timm==0.4.12',
        'opencv-python==4.5.3.56',
        'albumentations==1.0.3',
        'Pillow==8.3.2',
        'scikit-image==0.18.3',
        'numpy==1.21.2',
        'pandas==1.3.3',
        'scikit-learn==0.24.2',
        'matplotlib==3.4.3',
        'seaborn==0.11.2',
        'plotly==5.3.1',
        'PyYAML==5.4.1',
        'tqdm==4.62.3'
    ],
    extras_require={
        'dev': [
            'pytest==6.2.5',
            'black==21.9b0',
            'flake8==3.9.2',
            'jupyter==1.0.0'
        ],
        'prod': [
            'gunicorn==20.1.0',
            'Flask==2.0.1',
            'redis==3.5.3',
            'pydicom==2.2.2'
        ]
    },
    python_requires='>=3.9',
    entry_points={
        'console_scripts': [
            'chexpert-train=core.training.train:main',
            'chexpert-predict=core.deployment.predict:main',
        ],
    },
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Healthcare Industry',
        'Topic :: Scientific/Engineering :: Medical Science Apps.',
        'Programming Language :: Python :: 3.9',
    ],
)
