# DL2017Project
This is an initial repo for Deep Learning Project.

This is a Tensorflow Implementation of [Fully Convolutional Networks for Semantic Segmentation, CVPR 2015](https://github.com/shelhamer/fcn.berkeleyvision.org).

To do things a bit differently, we would like to take a [GoogLeNet (Inception v3)](https://github.com/tensorflow/models/tree/master/slim) and do this.

There is also a Tensorflow implementation here: [FCN.tensorflow](https://github.com/shekkizh/FCN.tensorflow).

This project would mostly based on these previous work.

# Edit this on Github Directly is WRONG!

# Update Log
## Major Updates: #1
uploaded modified inception_v3_fcn (generate the model) based on slim/nets/inception_v3.py

uploaded modified inception_FCN (training and visualize script) based on tensorflow.FCN/FCN.py and slim/train_image_classifier.py

uploaded inception_utils.py from slim/nets because I think it's needed

More will be uploaded later if needed

Possible work for next update

cleanup inception_v3_fcn: use slim/nets/inception_v3.py as much as possible and separate upsampling part

minor mod for inception_FCN: i dont know if it will work 


# Things that Yang found interesting
## Here is the presentation ([slides](https://docs.google.com/presentation/d/1VeWFMpZ8XN7OC3URZP4WdXvOGYckoFWGVN7hApoXVnc))given by the authors of the original paper.
http://techtalks.tv/talks/fully-convolutional-networks-for-semantic-segmentation/61606/

## Notes from this presentation
- Step 1: reinterpret fully connected layer as conv layers with 1x1 output. (No weight changing)
- Step 2: add conv layer at the very end to do upsample.
- Step 3: put a pixelwise loss in the end

			along the way we have stack of features.

			closer to the input - higher resolution - shallow, local - where

			closer to the output - lower resolution - deep, global - what
- Step 4: skip to fuse layers. interpolate and sum.
- Step 5: Fine tune on per-pixel dataset, PASCAL

			I stopped at 8:30 in the video

## This is about CONVERT fully connected layer to convolutional layer:
http://cs231n.github.io/convolutional-networks/#convert

## Some links about previous people asking about this but with no success. LOL:
http://stackoverflow.com/questions/38536202/how-to-use-inception-v3-as-a-convolutional-network
http://stackoverflow.com/questions/38565497/tensorflow-transfer-learning-implementation-semantic-segmentation

## Things that look promising and useful in the future
(I never thought that this could be a huge project. CIFAR10 conceived me.)
### CONVERT: make MS COCO usable in tenserflow (tfRecord)
The first thing we need to ask is MS COCO or PASCAL?

#### PASCAL (used in original FCN)
This guy's [Blog](http://warmspringwinds.github.io/blog/) and his [TensorFlow Image Segmentation](https://github.com/warmspringwinds/tf-image-segmentation) can be useful. 

Blog posts worth mentioning are: (some of this can also be found by the end of his project README)

[TFrecords Guide](http://warmspringwinds.github.io/tensorflow/tf-slim/2016/12/21/tfrecords-guide/)

[Convert Classification Network to FCN](http://warmspringwinds.github.io/tensorflow/tf-slim/2016/10/30/image-classification-and-segmentation-using-tensorflow-and-tf-slim/)

[His Implementation on FCN](http://warmspringwinds.github.io/tensorflow/tf-slim/2017/01/23/fully-convolutional-networks-(fcns)-for-image-segmentation/)

[About Upsampling](http://warmspringwinds.github.io/tensorflow/tf-slim/2016/11/22/upsampling-and-image-segmentation-with-tensorflow-and-tf-slim/)

Another implemetation called [Train Deeplab](https://github.com/martinkersner/train-DeepLab). They were not using TensorFlow but they were doing PASCAL with **less classes**. Could be useful. This could be developed as an extra feature. I don't know if this makes sense.

#### MS COCO
There is a [Python API for MS COCO](https://github.com/pdollar/coco), functionality unknown.

This [Tensorflow Annex](https://github.com/rwightman/tensorflow-annex#tensorflow-annex) thingy claims to do the conversion with no validation...

This [Show and Tell](https://github.com/tensorflow/models/tree/master/im2txt) example used MS COCO and did some convertion. But they include some "caption" as "labels". That's not us.

### Back Hand
[VGG in TensorFlow](https://www.cs.toronto.edu/~frossard/post/vgg16/)

