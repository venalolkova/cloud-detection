# Cloud Detection Using U-Net 
This repository containes code for Skoltech Machine Learning course final project completed by Ilya Chichkanov, Tamara Tsakhilova, Elena Volkova, Evgeny Guzovsky and Boris Arseniev.

It is an implementation of a U-Net for cloud detection trained on Landsat 8 high resolution satellite images https://earthexplorer.usgs.gov/ 
## Description of contents
Image_preprocessing.ipynb contains image segmentstion and snow/ice mask subtraction step

Subpixel_detection.ipynb contains training of conventional ML models and their metrics assessment

file models/ contains the resulting models from the previous ipynb

u_net.ipynb contains a PyTorch implementation of U-Net and experiments
