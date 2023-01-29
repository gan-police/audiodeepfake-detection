# wavelet-audiodeepfake-detection_code

CURRENT:
src/learn_direct_train_classifier.py current trainer, uses LearnWavefakeDataset() and transforms
data with cwt in the corresponding first layer of the chosen model (LearnDeepTestNet, OneDNet).

src/d_learn_direct_train_classifier.py implements the same as above but with direct data loading
from audios. Is reasonably slower.

src/prepare_dataset.py holds mainly utility methods.

src/prep_all.py prepares all datasets nicely.

src/data_loader.py holds costum datasets.

OLD:
src/train_classifier.py uses TransformDataset in src/data_loader.py and loads data directly.
-> buggy

src/new_train_classifier.py uses data from NumpyDataset of preprocessed data from prep_dataset.py
so all transformations are done before training. This is much quicker, but getting a gradient 
of different wavelets is not possible like that. Work in progress.
-> buggy

In prep_dataset.py three path lists for training, testing, validation are generated from given
folders. The audios are loaded one by one, cut to the a specified length and then transformed
with the cwt in wavelet_math.py. Currently all audios need to have a minimum length of the
specified cut length.
-> buggy


Exp. Todo:
- train single cmor3.3-4.17 2 for seed 0; 0,2,5,6 for seed 4; 0,5,6 for 3
- train single shan0.01-0.40 for seeds 1, 2, 3, 4
- train allgans cmor4.6-0.87 on seeds 0

- onednet as many params as LearnDeepTestNet
- train cmor3.3-4.17 on OneDNet for all seeds

- train adaptable on cmor4.6-0.87 on seeds 0-4

- eval a lot

- integrated gradients

- kreuzvalidierung mit leave-1to5-out

- Prep allgans with equal-sized gan test dataset
- Prep all 16000 Hz
- train single cmor3.3-4.17 on 16kHz on seed 0