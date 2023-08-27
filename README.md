# Towards generalizing deep-audio-fake detection networks

This is the supplementary source code for our paper "[Towards generalizing deep-audio fake detection networks](https://arxiv.org/abs/2305.13033)".

![packet vizualization](./img/packet_vizualization.png)

The plot above shows Wavelet packet visualizations of WaveFake-Sample LJ001-0002: "In being comparatively 
modern." A wavelet packet transformed version of the one-second long original recording is
shown on the left. The center plot depicts a full-band melgan re-synthesis. The plot on the right
shows their absolute difference. By leveraging the wavelet-packet and short-time fourier transform, 
we train excellent lightweight detectors that generalize and examine the results in our paper.

## Installation

The latest code can be installed in development mode in a running installation of python 3.10 with:

```shell
git clone git@github.com:gan-police/wavelet-audiodeepfake-detection.git
```
Move to the repository with
```shell
cd wavelet-audiodeepfake-detection
```
and install all requirements with
```shell
pip install -r requirements.txt
```

## Assets

For continuous wavelet computations, we use the:
- [PyTorch-Wavelet-Toolbox: ptwt](https://github.com/v0lta/PyTorch-Wavelet-Toolbox)

We compare our approach to the DCT-LFCC/MFCC-method from:
- [WaveFake: A Data Set to Facilitate Audio DeepFake Detection](https://github.com/RUB-SysSec/WaveFake)

### Datasets

We utilize two datasets that appeared in previous work:

- [LJSpeech 1.1](https://keithito.com/LJ-Speech-Dataset/)
- [WaveFake](https://zenodo.org/record/5642694)
- [ASVSpoof 2021 DeepFake Evaluation Dataset](https://zenodo.org/record/4835108)
- [ASVSpoof 2019 Logical Access Dataset](https://datashare.ed.ac.uk/handle/10283/3336)

### GAN Architectures
We utilize pre-trained models from the following repositories:

- [WaveFake](https://github.com/RUB-SysSec/WaveFake)
- [BigVGAN](https://github.com/NVIDIA/BigVGAN)

We used the inofficial implementation of Avocodo from [commit 2999557](https://github.com/ncsoft/avocodo) to train the avocodo vocoder.

## Reproduction

The following section of the README serves as a guide to reproducing the experiments from our paper.

### Preparation WaveFake

As the WaveFake dataset contains gan generated audios equivalent to the audios of LJSpeech, no further preparation needs to be done to get all audios that are needed. We work with mono-channeled audios of different sizes. Hence, the raw audio needs to be cut into equally sized frames of desired size. We mainly used frames of 1s. The sample rate can be varied as well.

To do this store all audios (original and GAN-generated) in separate subdirectories, i.e. the directory structure should look like this

```
data
  └── cross_test
      ├── B_melgan
      |    ├── LJ001-0001_gen.wav
      |    ├── ...
      |    └── LJ008-0217_gen.wav
      ├── C_hifigan
      |    ├── LJ001-0001_gen.wav
      |    ├── ...
      |    └── LJ008-0217_gen.wav
      └── A_ljspeech
           ├── LJ001-0001.wav
           ├── ...
           └── LJ008-0217.wav
```

The prefixes of the folders are important, since the directories get the labels in lexicographic order of their prefix, i.e. directory `A_...` gets label 0, `B_...` label 1, etc. If you skip certain letters in the alphabet that is okay as well. The labels will be in ascending order beginning from 0 automatically.

Now, to prepare the data sets run `python -m scripts.prepare_ljspeech`. It reads the data set, cuts all audios to pieces of given size, splits them into a training, validation and test set, and stores the resulting audio paths with the frame numbers for each audio as numpy arrays.
Use the parameter `use_only` to specify the name of the directories that should be used from the given data path. E.g. if there are directories `A_ljspeech`, `B_melgan` and `C_hifigan` but you only want to use the first two, set `only_use=["ljspeech", "melgan"]` in the corresponding dataset.

This process could take some time, because it reads the length of all audio files. The results will be saved in the directory specified in `save_path` and hence this process has to only run once for each dataset.

### Preparation ASVSpoof
We are using Logical Access (LA) train and eval sets of ASV Spoof Challenge 2019 and the DeepFake (DF)
1. Download data
1.2. download keys (for DF ASVspoof 21) and unpack them in the folder containing the data of ASVspoof 21
2. unzip/untar data
3. Adjust base_path
4. Run `python -m scripts.split_asvspoof` from repository folder first for ASVspoof 2019
5. Uncomment lines for ASVspoof 2021 and run `python -m scripts.split_asvspoof`
6. Adjust `save_path` and `data_path` in `scripts/prepare_asvspoof.py` so that `save_path` is the folder for the datasets files that will be generated and `data_path` the directory with the structure e.g.
└── cross_test
      ├── A_asv2019real
      |    └── real1.flac...
      ├── B_asv2019fake
      |    └── fake1.flac...
      ├── A_asv2021real
      |    └── real1.flac...
      └── B_asv2021fake
           └── fake1.flac...

### Training the Classifier

Now you should be able to train a classifier using the config file in `scripts/gridsearch_config.py` and the train scripts. The train scripts start the training process with some configuration values that can be changed. These will be loaded into a variable dict named `args` wich is dot accessible (e.g. `args.epochs`). If you run e.g. `scripts/train_multigpu.py` python will run `src.train_classifier` using the grid search functionality. In this case the given training parameters will be overridden if found in the config dict in `scripts/gridsearch_config.py`. There you can also define new training args if you want to use them later in a model or somewhere else in the code. Each parameter expects a list of values with len(list) >= 1. If you only give one value it will run only this one experiment. If you give more than one value the script will run two different experiments, one for each value. If you give more than one value for e.g. two parameters, the script will run `2 * 2 = 4` experiments.

Keep in mind that each experiment will be run for 5 different seeds.

For asvspoof set `config = asv_config` in the gridsearch config, for wavefake set `config = wf_config`.

### Evaluating the Classifier

#### Calculating accuracy and equal error rate (EER)

To test a model that was already trained, set the argument `only_testing` in the initial training script or in `scripts/gridsearch_config.py` to `True`.


## Issues
As we use the Adam optimizer of the python module pytorch, we recommend to use torch 2.0.0, torchaudio 2.0.0 and cuda 11.7.

Important: If training with multiple GPUs be aware of the train, test and val set sizes to be equal to our initial settings to get reproducible results.
They should e.g. be multiples of batch size * number of GPUs to have the same sizes when training with 1, 2, 4, 6, 8 or 16 GPUs.
When training and testing on different GPU hardware than our settings we cannot guarantee equal results.

## Citation
This work is in the public domain. Feel free to use my material, but please cite it properly.
```
@misc{gasenzerwolter2023generalizingadf,
      title={Towards generalizing deep-audio fake detection networks}, 
      author={Konstantin Gasenzer and Moritz Wolter},
      year={2023},
      eprint={2305.13033},
      archivePrefix={arXiv},
      primaryClass={cs.SD}
}
```
