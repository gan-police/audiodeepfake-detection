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

### GAN Architectures
We utilize pre-trained models from the following repositories:

- [WaveFake](https://github.com/RUB-SysSec/WaveFake)
- [BigVGAN](https://github.com/NVIDIA/BigVGAN)

We used the inofficial implementation of Avocodo from [commit 2999557](https://github.com/ncsoft/avocodo) to train the avocodo vocoder.

## Reproduction

The following section of the README serves as a guide to reproducing the experiments from our paper.

### Preparation

As the WaveFake dataset contains gan generated audios equivalent to the audios of LJSpeech, no further preparation needs to be done to get all audios that are needed. We work with mono-channeled audios of different sizes. Hence, the raw audio needs to be cut into equally sized frames of desired size. We mainly used frames of 0.5s and 0.25s. The sample rate can be varied as well.

To do this store all images (original and GAN-generated) in separate subdirectories of one or two directories (depends if you split real and fake), i.e. the directory structure should look like this

```
data
  ├── fake
  │    ├── B_melgan
  │    |    ├── LJ001-0001_gen.wav
  |    |    ├── ...
  │    |    └── LJ008-0217_gen.wav
  │    └── C_hifigan
  │         ├── LJ001-0001_gen.wav
  |         ├── ...
  │         └── LJ008-0217_gen.wav
  ├── real
  |    └── A_ljspeech
  |         ├── LJ001-0001.wav
  |         ├── ...
  |         └── LJ008-0217.wav
  └── fake_test
       └── N_conformer
            ├── 0001_gen.wav
            ├── ...
            └── 2200_gen.wav
```

The prefixes of the folders are important, since the directories get the labels in lexicographic order of their prefix, i.e. directory `A_...` gets label 0, `B_...` label 1, etc.

Now, to prepare the data sets run `src.prepare_dataset` . It reads data set, splits them into a training, validation and test set, cuts all audios to pieces of window_size and stores the result as numpy arrays. 
If you want to only generate test datasets e.g. for cross validation, put them into a seperate folder like `fake_test` and pass them to `prepare_datasets.py` with the option `--testdir`. This could look like this:

```shell
python -m src.prepare_datasets \
    --window-size 22050 \
    --sample-rate 22050 \
    --realdir "${HOME}/data/real/A_ljspeech" \
    --fakedir "${HOME}/data/fake/B_melgan" \
    --directory "${HOME}/data/fake" \
    --target-dir "${HOME}/data/datasets"
```
or
```shell
python -m src.prepare_datasets \
    --window-size 22050 \
    --sample-rate 22050 \
    --realdir "${HOME}/data/real/A_ljspeech" \
    --directory "${HOME}/data/fake"
    --target-dir "${HOME}/data/datasets"
```

The dataset preparation script accepts additional arguments. For example, it is possible to change the sizes of the train, test or validation sets. Important: All .wav files need to have the same sample rate before preparing the datasets. For a list of all optional arguments, open the help page via the `-h` argument.

We recommend to use the scripts `scripts/prepare_single_dataset.sh`, `scripts/prepare_all_dataset.sh` and `scripts/prepare_test_dataset.sh`.

Important: After preparing all folders run `python scripts/clean_up.py` to make sure that all dataset splits contain the same amount of positive and negative labels. Make sure to set `path` to the target directory from above (it should only contain the output folders from the step before ending with `_test`, `_val` and `_all`). Check if the script executed properly with
```shell
for folder in *_train; do echo "$folder: $(find "$folder" -maxdepth 1 -type f | wc -l)"; done
```
Find all folders to contain the same amount of files.

### Training the Classifier

Now you should be able to train a classifier using for example:

```shell
python -m src.train_classifier \
    --data-prefix "${HOME}/data/fake_22050_11025_0.7_melgan" \
    --batch-size 128 \
    --learning-rate 0.0001 \
    --weight-decay 0.0001   \
    --epochs 10 \
    --model "lcnn"  \
    --wavelet "cmor3.3-4.17" \
    --f-min 1000 \
    --features "none" \
    --hop-length 50 \
    --transform packets \
    --calc-normalization \
    --f-max 9500 \
    --num-of-scales 150 \
    --sample-rate 22050 \
    --flattend-size 352 \
    --pbar \
    --tensorboard
```

This trains a cnn classifier using the chosen hyperparameters. The training, validation and test accuracy and loss values are stored in a file placed in a `log` folder. The state dict of the trained model is stored there as well. Using the argument `--adapt-wavelet` will make the wavelet bandwidth and center frequency part of the trainable parameters of the model. For a list of all optional arguments, open the help page via the `-h` argument.

### Evaluating the Classifier

#### Calculating accuracy and equal error rate (EER)

To calculate the accuracy and eer of trained models use `src.eval_models` with varying arguments, e.g.
```shell
python -m src.eval_models \
    --data-prefix "${HOME}/data/fake_22050_11025_0.7" \
    --model-path-prefix ./log/fake_packets_none_100_22050_22050_256_1-11025_0.7_0.0001_0.01_64_2_10e_lcnn_False \
    --model "lcnn"  \
    --batch-size 64 \
    --window-size 22050 \
    --sample-rate 22050 \
    --flattend-size 352 \
    --features none \
    --seed 0 \
    --transform packets \
    --train-gans "melgan"
```

If you want to cross evaluate models against the test sets of other gans, you can use the `--crosseval-gans`, e.g. like this:
```shell
python -m src.eval_models \
    --data-prefix "${HOME}/data/fake_22050_11025_0.7" \
    --model-path-prefix ./log/fake_packets_none_100_22050_22050_256_1-11025_0.7_0.0001_0.01_64_2_10e_lcnn_False \
    --model "lcnn"  \
    --batch-size 64 \
    --window-size 22050 \
    --sample-rate 22050 \
    --flattend-size 352 \
    --features none \
    --seed 0 \
    --transform packets \
    --train-gans "melgan" \
    --crosseval-gans "melgan" "lmelgan" "mbmelgan" "fbmelgan" "hifigan" "waveglow" "pwg"
```
This evaluates the trained melgan model against all the other fake audios with default parameters, which makes it possible to see how well the classifier generalizes to other gan generated audios. All results will be saved in under `log/results/`.

For a list of all arguments, open the help page via the `-h` argument.


### Attributing the Classifier

To get a grip of which parts of the input the classifier uses to discriminate the audios in real and fake, I implemented attribution via integrated gradients. These can be used like this:
```shell
python -m src.integrated_gradients \
    --data-prefix "${HOME}/data/fake_22050_11025_0.7" \
    --plot-path "plots/attribution/" \
    --target-label 1 \
    --times 5056 \
    --model "learndeepnet"  \
    --wavelet "cmor3.3-4.17" \
    --num-of-scales 150 \
    --sample-rate 22050 \
    --flattend-size 21888 \
    --gans "melgan"
```
This plots the saliency, the integrated gradients and the mean over all time of the integrated gradients as min-, max- and absolute histograms for all given frequencies and saves the plots to latex standalone scripts.


## Issues
As we use the Adam optimizer of the python module pytorch, we recommend to use torch 2.0.0, torchaudio 2.0.0 and cuda 11.7.

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
