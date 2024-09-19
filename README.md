# sign-classification
(Maybe) The best traffic speed sign classifier ever.

# Requirements:

- Needs to be able to determine the speed limit from the images in the dataset.
- All images are taken at an angle < 45 degrees from the sign. It shouldn't have to work when viewing signs from the side.

# Software plan: 

The software should follow the structure as taught in class:
- Image acquisition
- Preprocessing
- Segmentation
- Feature extraction
- Classification

As image acquisition is already done and is a manual task, it is excluded from our software. This means that for each part
of the pipeline we have a python file. These are found in the 'src' folder. The __main__.py file should call functions from
each part of the pipeline. 

# Extra info:

- The dataset should be moved into the 'dataset' folder in this repo.
- For access to the dataset contact one of the contributors.
