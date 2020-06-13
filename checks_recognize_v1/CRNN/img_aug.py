from imgaug import augmenters as iaa 
from PIL import Image
import numpy as np 

class RandomElastic(object):

    def __init__(self):
        pass 

    def __call__(self, img):
        augmenter = iaa.ElasticTransformation(alpha=50, sigma=9)
        img = Image.fromarray(augmenter.augment_image(np.array(img)))
        return img

    def __repr__(self):
        return self.__class__.__name__

class RandomAddToHueAndSaturation(object):

    def __init__(self):
        pass 

    def __call__(self, img):
        augmenter = iaa.AddToHueAndSaturation((-60, 60))
        img = Image.fromarray(augmenter.augment_image(np.array(img)))
        return img

    def __repr__(self):
        return self.__class__.__name__

class RandomCoarseDropout(object):

    def __init__(self):
        pass 

    def __call__(self, img):
        augmenter = iaa.CoarseDropout(p=0.08, size_percent=(0.2, 0.2))
        img = Image.fromarray(augmenter.augment_image(np.array(img)))
        return img

    def __repr__(self):
        return self.__class__.__name__

class RandomMotionBlur(object):

    def __init__(self):
        pass 

    def __call__(self, img):
        augmenter = iaa.MotionBlur(k=8)
        img = Image.fromarray(augmenter.augment_image(np.array(img)))
        return img

    def __repr__(self):
        return self.__class__.__name__

class RandomGussBlur(object):

    def __init__(self):
        pass 

    def __call__(self, img):
        augmenter = iaa.GaussianBlur(sigma=(0.0, 1.2))
        img = Image.fromarray(augmenter.augment_image(np.array(img)))
        return img

    def __repr__(self):
        return self.__class__.__name__


class RandomPerspective(object):

    def __init__(self):
        pass 

    def __call__(self, img):
        augmenter = iaa.PerspectiveTransform(scale=0.05, cval=0.1, keep_size=True)
        img = Image.fromarray(augmenter.augment_image(np.array(img)))
        return img

    def __repr__(self):
        return self.__class__.__name__

class RandomAffine(object):

    def __init__(self):
        pass 

    def __call__(self, img):
        augmenter = iaa.Affine(scale=1, shear=(-50, 50), mode='edge')
        img = Image.fromarray(augmenter.augment_image(np.array(img)))
        return img

    def __repr__(self):
        return self.__class__.__name__
