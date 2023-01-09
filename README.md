# wavelet-audiodeepfake-detection_code

src/train_classifier.py uses TransformDataset in src/data_loader.py and loads data directly.

src/new_train_classifier.py uses data from NumpyDataset of preprocessed data from prep_dataset.py
so all transformations are done before training. This is much quicker, but getting a gradient 
of different wavelets is not possible like that. Work in progress.

In prep_dataset.py three path lists for training, testing, validation are generated from given
folders. The audios are loaded one by one, cut to the a specified length and then transformed
with the cwt in wavelet_math.py. Currently all audios need to have a minimum length of the
specified cut length. 