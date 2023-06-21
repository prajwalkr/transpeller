# Weakly-supervised Fingerspelling Recognition in British Sign Language Videos

This is the official implementation of the paper. The code has been tested with Python version 3.6.8. Pre-trained checkpoint for fingerspelling is also released below. 

## Environment & checkpoints
- `pip install -r requirements.txt`
- Download the pre-trained Transpeller checkpoint:
  - `cd data/`
  - wget [https://www.robots.ox.ac.uk/~vgg/research/transpeller/transpeller.pth](https://www.robots.ox.ac.uk/~vgg/research/transpeller/transpeller.pth)
- Get the video features:
  - Follow the instructions on the [BOBSL page](https://www.robots.ox.ac.uk/~vgg/data/bobsl/#data) to get the username and password to access parts of the BOBSL dataset. 
  - `cd features`
  - `sh download_features.sh username password` 
- Get the annotations:
  - `cd data/`
  - `sh download.sh username password`. This is a fast download that obtains the manually verified test annotations and the automatically obtained annotations for the BOBSL episodes. For the automatic annotations, the `?` in the `word` column indicates the fingerspelled word could not be determined by the automatic pseudolabeling method.

## Reproducing the scores on the test set

```bash
python test.py --ckpt_path data/transpeller.pth --builder localizer_ctc --test_csv data/fingerspelling-data-bmvc2022/transpeller-test.csv --feat_root features/video-swin-s_c8697_16f_bs32/
```

The above run should give a CER of `53.1`. You can also turn on the `--full_word_test` flag to compute CER with the full words, which should be `59.9`. 

## Using Video-Swin as a feature extractor

We also release the pre-trained Video-Swin model which is used to extract the features mentioned above. The model has been trained on person-crops of the BOBSL dataset. You can get the pre-trained checkpoint [here](https://www.robots.ox.ac.uk/~vgg/research/transpeller/video-swin-s.pth). Below is a small example of how to use it:

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

License and Citation
----------
The code, models, and the released annotations are bound by the exact same licensing terms stated on the [official BOBSL page](https://www.robots.ox.ac.uk/~vgg/data/bobsl/#data). 

Please cite the following paper if you use this repository:
```
@InProceedings{Prajwal22a,
  author       = "K R Prajwal and Hannah Bull and Liliane Momeni and Samuel Albanie and G{\"u}l Varol and Andrew Zisserman",
  title        = "Weakly-supervised Fingerspelling Recognition in British Sign Language Videos",
  booktitle    = "British Machine Vision Conference",
  year         = "2022",
  keywords     = "sign language, fingerspelling, bsl, bobsl",
}
```