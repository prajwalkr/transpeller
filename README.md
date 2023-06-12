# Weakly-supervised Fingerspelling Recognition in British Sign Language Videos

This is the official implementation of the paper. The code has been tested with Python version 3.6.8. Pre-trained checkpoint for fingerspelling is also released below. 

## Environment & checkpoints
- `pip install -r requirements.txt`
- Download the pre-trained checkpoint for fingerspelling and also the test set CSV.
  - `cd data/`
  - wget [https://www.robots.ox.ac.uk/~vgg/research/transpeller/transpeller.pth](https://www.robots.ox.ac.uk/~vgg/research/transpeller/transpeller.pth)
  - wget [https://www.robots.ox.ac.uk/~vgg/research/transpeller/bobsl-transpeller-test.csv](https://www.robots.ox.ac.uk/~vgg/research/transpeller/bobsl-transpeller-test.csv)
  - Download the video features: 

## Reproducing the scores on the test set

`python test.py --ckpt_path data/transpeller.pth --builder localizer_ctc --test_csv data/bobsl-transpeller-test.csv --feat_root <path_to_bobsl_features_root>`

The above run should give a CER of `53.1`. You can also turn on the `--full_word_test` flag to compute CER with the full words, which should be `59.9`. 

## Using Video-Swin as a feature extractor

We also release the pre-trained Video-Swin model which is used to extract the features mentioned above. The model has been trained on person-crops of the BOBSL dataset. You can get the pre-trained checkpoint [here](https://www.robots.ox.ac.uk/~vgg/research/transpeller/video-swin-s.pth). Below is a small example how to use it:

```python
from videoswin import SwinTransformer3D, VideoPreprocessing
from utils import load

# BOBSL person-crop video input
model = SwinTransformer3D()
model = load("video-swin-s.pth")[0]

vp = VideoPreprocessing()

clip = # read video clip with size (batch_size, 3, 16, 256, 256)

clip = vp(clip) # (batch_size, 3, 16, 224, 224)

features = model(clip) # (batch_size, 768)
```
