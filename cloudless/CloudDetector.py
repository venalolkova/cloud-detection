import copy
import os
import numpy as np

from scipy.ndimage.filters import convolve
from skimage.morphology import disk, dilation
import pickle




class CloudDetector:

    def __init__(self, threshold=0.4, average_over=1, dilation_size=1, model_filename=None):

        self.threshold = threshold
        self.average_over = average_over
        self.dilation_size = dilation_size

        if model_filename is not None:
            package_dir = os.path.dirname(__file__)
            model_filename = os.path.join(package_dir, 'models', model_filename)
        self.model_filename = model_filename
        print(self.model_filename)
        loaded_model = pickle.load(open(self.model_filename, "rb"))
        self.classifier = loaded_model
        if average_over > 0:
            self.conv_filter = disk(average_over) / np.sum(disk(average_over))

        if dilation_size > 0:
            self.dilation_filter = disk(dilation_size)

           

    def get_cloud_probability_maps(self, X):
  
        if X.shape[3] != 4:
            raise ValueError("Numper of channel!= 4")
        if len(X.shape) != 4:
            raise ValueError('Array of input images has to be a 4-dimensional array of shape'
                             '[n_images, n_pixels_y, n_pixels_x, n_bands]')
        pixels = X.reshape(-1,4)
        probabilities = self.classifier.predict_proba(pixels)
        return probabilities[:,1].reshape( X.shape[0], X.shape[1], X.shape[2])


    def get_cloud_masks(self, X,threshold=None):
    
        cloud_probs = self.get_cloud_probability_maps(X)
        
        threshold = self.threshold if threshold is None else threshold
        if self.average_over:
            cloud_masks = np.asarray([convolve(cloud_prob, self.conv_filter) > threshold
                                      for cloud_prob in cloud_probs], dtype=np.int8)
        else:
            cloud_masks = (cloud_probs > threshold).astype(np.int8)

        if self.dilation_size:
            cloud_masks = np.asarray([dilation(cloud_mask, self.dilation_filter) for cloud_mask in cloud_masks], dtype=np.int8)
        return cloud_masks




