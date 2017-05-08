## Train you model on flickr30

we try to achieve the image caption model that are proposed by the google using the pytorch

## Overview

The pipeline for the project looks as follows:
- the input is a dataset of images features and 5 snetences captions. In particular, the code is set up
for the dataset for flicker30
- in the train step, the image features are fed as input to LSTM and the LSTM is to predict the sentence,
conditioned on the current word and previous state.
- in the step of predict, the image features are fed to the trained LSTM and the LSTM is to predict to the     sentence.


## Dependencies
The code are requires pyTorch, if you want to run the code, please install the pytorch.
we only tested the code on OSX. so if you're on OSX, you can install pytorch as following:
'$ pip install http://download.pytorch.org/whl/torch-0.1.12.post2-cp27-none-macosx_10_7_x86_64.whl'
'$ pip install torchvision '


## Get started
1. **Get the code.** `$ git clone the repositories`
2. **Get the datset.** you can download the datset [here](http://cs.stanford.edu/people/karpathy/deepimagesent/)
3. **Get the vocab**. `$ python build_vocab.py`
3. **Train model.** `$ python dirve.py`
4. **Evaluate the result.** `$ python eval_image.py`
achieve the image2caption according the google paper
