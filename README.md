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


# How to run the program

Run the following command to download the dependencies:

```
./venv
```

Run the following scripts to get the results from the dataset

```
python3 src/feature_extraction.py
python3 src/preprocessing.py
python3 src/classification.py
```

Run the following command to run a video with it

```
python3 src/__main__.py --video=(path to your video)
```

Run the following command to run a single image with it

```
python3 src/__main__.py --image=(path to your image)
```

# Extra info:

- The dataset should be moved into the 'dataset' folder in this repo.
- For access to the dataset contact one of the contributors.

# Data set link
The following link is to the dataset that we have used.

https://hannl-my.sharepoint.com/personal/m_sterk_student_han_nl/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fm%5Fsterk%5Fstudent%5Fhan%5Fnl%2FDocuments%2FML%20Photos

# Test set

The test set is used especially for the test_cases.

Example: python3 src/__main__.py --image=test_set/1727344296913505264.png

