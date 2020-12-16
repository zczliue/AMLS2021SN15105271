
# This is a rewritten version of lab2_landamarks.py provided in AMLS Week 6 lab.
# The main changes are focused on the function extract_features_labels,
# which now allows label and feature extraction from any dataset folders 


import os
import numpy as np
from keras.preprocessing import image
import cv2
import dlib
from collections import OrderedDict

# PATH TO ALL IMAGES
global basedir, image_paths, target_size
basedir = 'Datasets'
labels_filename = 'labels.csv'

detector = dlib.get_frontal_face_detector()
#predictor = dlib.shape_predictor('A2/shape_predictor_68_face_landmarks.dat')
predictor = dlib.shape_predictor(os.path.join('A2', 'shape_predictor_68_face_landmarks.dat'))


# This is the dictionary contains dlib's 68-point facial landmark detector:
# copyright: https://github.com/jrosebr1/imutils/blob/master/imutils/face_utils/helpers.py
FACIAL_LANDMARKS_68_IDXS = OrderedDict([
	("mouth", (48, 68)),
	("inner_mouth", (60, 68)),
	("right_eyebrow", (17, 22)),
	("left_eyebrow", (22, 27)),
	("right_eye", (36, 42)),
	("left_eye", (42, 48)),
	("nose", (27, 36)),
	("jaw", (0, 17))
])



# how to find frontal human faces in an image using 68 landmarks.  These are points on the face such as the corners of the mouth, along the eyebrows, on the eyes, and so forth.

# The face detector we use is made using the classic Histogram of Oriented
# Gradients (HOG) feature combined with a linear classifier, an image pyramid,
# and sliding window detection scheme.  The pose estimator was created by
# using dlib's implementation of the paper:
# One Millisecond Face Alignment with an Ensemble of Regression Trees by
# Vahid Kazemi and Josephine Sullivan, CVPR 2014
# and was trained on the iBUG 300-W face landmark dataset (see https://ibug.doc.ic.ac.uk/resources/facial-point-annotations/):
#     C. Sagonas, E. Antonakos, G, Tzimiropoulos, S. Zafeiriou, M. Pantic.
#     300 faces In-the-wild challenge: Database and results.
#     Image and Vision Computing (IMAVIS), Special Issue on Facial Landmark Localisation "In-The-Wild". 2016.


def shape_to_np(shape, dtype="int"):
    # initialize the list of (x, y)-coordinates
    coords = np.zeros((shape.num_parts, 2), dtype=dtype)

    # loop over all facial landmarks and convert them
    # to a 2-tuple of (x, y)-coordinates
    for i in range(0, shape.num_parts):
        coords[i] = (shape.part(i).x, shape.part(i).y)

    # return the list of (x, y)-coordinates
    return coords



def rect_to_bb(rect):
    # take a bounding predicted by dlib and convert it
    # to the format (x, y, w, h) as we would normally do
    # with OpenCV
    x = rect.left()
    y = rect.top()
    w = rect.right() - x
    h = rect.bottom() - y

    # return a tuple of (x, y, w, h)
    return (x, y, w, h)


def run_dlib_shape(image):
    # in this function we load the image, detect the landmarks of the face, and then return the image and the landmarks
    # load the input image, resize it, and convert it to grayscale
    resized_image = image.astype('uint8')

    gray = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
    gray = gray.astype('uint8')

    # detect faces in the grayscale image
    rects = detector(gray, 1)
    num_faces = len(rects)

    if num_faces == 0:
        return None, resized_image

    face_areas = np.zeros((1, num_faces))
    face_shapes = np.zeros((136, num_faces), dtype=np.int64)

    # loop over the face detections
    for (i, rect) in enumerate(rects):
        # determine the facial landmarks for the face region, then
        # convert the facial landmark (x, y)-coordinates to a NumPy
        # array
        temp_shape = predictor(gray, rect)
        temp_shape = shape_to_np(temp_shape)

        # convert dlib's rectangle to a OpenCV-style bounding box
        # [i.e., (x, y, w, h)],
        #   (x, y, w, h) = face_utils.rect_to_bb(rect)
        (x, y, w, h) = rect_to_bb(rect)
        face_shapes[:, i] = np.reshape(temp_shape, [136])
        face_areas[0, i] = w * h
    # find largest face and keep
    dlibout = np.reshape(np.transpose(face_shapes[:, np.argmax(face_areas)]), [68, 2])

    return dlibout, resized_image



# This function is rewritten as follows:
# the input image_folder must be a string that specifies 'celeba' or 'cartoon_set'
# the input label_name must be a string that specifies the feature name, i.e. 'gender', 'smiling' as in labels.csv
def extract_features_labels(image_folder, label_name):
    """
    This funtion extracts the landmarks features for all images in the folder 'dataset/celeba'.
    It also extracts the gender label for each image.
    :return:
        landmark_features:  an array containing 68 landmark points for each image in which a face was detected
        gender_labels:      an array containing the gender label (male=0 and female=1) for each image in
                            which a face was detected
    """

    #image_dir is loaded under each subfolder 'img'
    #skip DS_Store file in Mac environment
    images_dir = os.path.join(basedir, image_folder, 'img')
    image_paths = []
    for l in os.listdir(images_dir):
        if l != '.DS_Store':
            image_paths.append(os.path.join(images_dir, l))

    target_size = None


    #label.csv direction under each image_folder:
    labels_file = open(os.path.join(basedir, image_folder, labels_filename), 'r')
    lines = labels_file.readlines()


    #label_headings  = lines[0].strip().split('\t')
    label_headings = lines[0].replace('\n', '').split('\t')
    #find index of label_name
    label_idx = label_headings.index(label_name)


    #extract corresponding index of the label in lines
    #start from line[1:] other wise the first line will be missing since
    #Datasets has a different format from datasets in lab2
    label_contents = {line.replace('\n', '').split('\t')[0] : int(line.replace('\n', '').split('\t')[label_idx]) for line in lines[1:]}

    if os.path.isdir(images_dir):
        all_features = []
        all_labels = []

        no_features_sets = []
        
        for img_path in image_paths:
            #file_name= img_path.split('/')[-1].split('.')[0]
            file_name= (os.path.split(img_path)[1]).split('.')[0]

            # load image
            img = image.img_to_array(
                image.load_img(img_path,
                               target_size=target_size,
                               interpolation='bicubic'))
            features, _ = run_dlib_shape(img)
            if features is not None:
                all_features.append(features)
                all_labels.append(label_contents[file_name])
            else:
                #this is the collection where detector cannot extract useful information 
                no_features_sets.append(file_name)
                



    landmark_features = np.array(all_features)
    label_contents = (np.array(all_labels) + 1)/2 # simply converts the -1 into 0, so male=0 and female=1
    return landmark_features, label_contents, no_features_sets

