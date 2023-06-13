"""
Module implementing pixel-based classifier
"""
import numpy as np
class PixelClassifier:

    def __init__(self, classifier):
 
        self._check_classifier(classifier)
        self.classifier = classifier

    @staticmethod
    def _check_classifier(classifier):
        """
        Checks if the classifier is of correct type or if it implements predict and predict_proba methods
        """

        predict = getattr(classifier, 'predict', None)
        if not callable(predict):
            raise ValueError('Classifier does not have a predict method!')

        predict_proba = getattr(classifier, 'predict_proba', None)
        if not callable(predict_proba):
            raise ValueError('Classifier does not have a predict_proba method!')

    @staticmethod
    def extract_pixels(X):
        """ Extracts pixels from array X

        :param X: Array of images to be classified.
        :type X: numpy array, shape = [n_images, n_pixels_y, n_pixels_x, n_bands]
        :return: Reshaped 2D array
        :rtype: numpy array, [n_samples*n_pixels_y*n_pixels_x,n_bands]
        :raises: ValueError is input array has wrong dimensions
        """
        if len(X.shape) != 4:
            raise ValueError('Array of input images has to be a 4-dimensional array of shape'
                             '[n_images, n_pixels_y, n_pixels_x, n_bands]')

        new_shape  = X.shape[0] * X.shape[2] * X.shape[3], X.shape[1]
        pixels = X.reshape(-1,4)
        return pixels


    def image_predict_proba(self, X, **kwargs):
        """
        Predicts class probabilities for the entire image.

        :param X: Array of images to be classified.
        :type X: numpy array, shape = [n_images, n_pixels_y, n_pixels_x, n_bands]
        :param kwargs: Any keyword arguments that will be passed to the classifier's prediction method
        :return: classification probability map
        :rtype: numpy array, [n_samples, n_pixels_y, n_pixels_x]
        """
        pixels = self.extract_pixels(X)

        probabilities = self.classifier.predict_proba(pixels, **kwargs)
        print(probabilities.shape)
        return probabilities[:,1].reshape( X.shape[0], X.shape[1], X.shape[2])
