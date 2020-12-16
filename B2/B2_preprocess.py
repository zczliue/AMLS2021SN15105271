
# This is a rewritten version of lab2_landamarks.py provided in AMLS Week 6 lab.
# The main changes are focused on the function extract_features_labels,
# which now allows label and feature extraction from any dataset folders 


import os
import numpy as np
from keras.preprocessing import image
import cv2
import dlib
from collections import OrderedDict
import math

# PATH TO ALL IMAGES
global basedir, image_paths, target_size
basedir = 'Datasets'
#image_dir needs to be manually input according to tasks
#images_dir = os.path.join(basedir,'celeba')
labels_filename = 'labels.csv'

detector = dlib.get_frontal_face_detector()
#predictor = dlib.shape_predictor('B2/shape_predictor_68_face_landmarks.dat')
predictor = dlib.shape_predictor(os.path.join('B2', 'shape_predictor_68_face_landmarks.dat'))


# This is the dictionary contains dlib's 68-point facial landmark detector:
# copyright: https://github.com/jrosebr1/imutils/blob/master/imutils/face_utils/helpers.py
FACIAL_LANDMARKS_68_IDXS = OrderedDict([
	("mouth", (48, 68)),
	("inner_mouth", (60, 68)),
	("right_eyebrow", (17, 22)),
	("left_eyebrow", (22, 27)),
	("right_eye", (36, 41)),
	("left_eye", (42, 47)),
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

def get_righteye_roi(features):


    #only requires 4 outer points to estimate a square region that contains the right eye
    xleft = features[36][0]
    xright = features[39][0]
    ytop = features[37][1]
    ybottom = features[39][1]

    return xleft, xright, ytop, ybottom

# This function is rewritten as follows:
# the input image_folder must be a string that specifies 'celeba' or 'cartoon_set'
# the input label_name must be a string that specifies the feature name, i.e. 'gender', 'smiling' as in labels.csv
def extract_features_labels_and_dominant_colour(image_folder, label_name, margin, n_colors):
    """
    This funtion extracts the landmarks features for all images in the folder 'dataset/celeba'.
    It also extracts the gender label for each image.
    :return:
        landmark_features:  an array containing 68 landmark points for each image in which a face was detected
        gender_labels:      an array containing the gender label (male=0 and female=1) for each image in
                            which a face was detected
    """

    #image_dir is loaded under each subfolder 'img'
    images_dir = os.path.join(basedir, image_folder, 'img')
    image_paths = []
    for l in os.listdir(images_dir):
        if l != '.DS_Store':
            image_paths.append(os.path.join(images_dir, l))
 
    #image_paths = [os.path.join(images_dir, l) for l in os.listdir(images_dir)]
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
        identified_file_sets = []

        all_dominant_rgb = []
        
        for img_path in image_paths:


            #file_name= img_path.split('/')[-1].split('.')[0]
            file_name= (os.path.split(img_path)[1]).split('.')[0]

            # load image
            img_org =  cv2.imread(img_path)
            img = image.img_to_array(img_org)
            #img = image.img_to_array(
            #    image.load_img(img_path,
            #                   target_size=target_size,
            #                   interpolation='bicubic'))
            features, _ = run_dlib_shape(img)


            if features is not None:

                identified_file_sets.append(file_name)
                all_features.append(features)
                all_labels.append(label_contents[file_name])

                #extract 4 points at landamrks 36, 37, 39
                #which corresponds to the righteye region
                xleft, xright, ytop, ybottom = get_righteye_roi(features)


                #crop the original image, leaving only right eye area 
                #enlarge and resample to improve the accuracy of circle detection
                #resized_img = crop_and_resample(xleft, xright, ytop, ybottom, margin, img)
                resized_img = crop_and_resample(xleft, xright, ytop, ybottom, margin, img_org)

                #cv2.imshow("resized_img", resized_img)
                #cv2.waitKey(0)


                #perform circle detection within the resized image
                #the average circle detected should normally fit the eye socket 
                avgcenter, avgradius, src = circle_detection(resized_img)

                #To avoid distortions from other area, we continue to crop the image.
                #For simplicity, select the largest square within the circular region. 
                square_roi = crop_largest_square_within_circle(avgcenter, avgradius, src)

                #extract the dominant colour in the square roi
                dominant_rgb = extract_dominant_colour(square_roi,  n_colors)

                all_dominant_rgb.append(dominant_rgb)


            else:
                #this is the collection where detector cannot extract useful information 
                no_features_sets.append(file_name)
                
    dominant_rgbs = np.array(all_dominant_rgb)
    landmark_features = np.array(all_features)
    label_contents = np.array(all_labels) 
    return landmark_features, label_contents, no_features_sets, identified_file_sets, dominant_rgbs 



def crop_and_resample(xleft, xright, ytop, ybottom, margin, img):

    crop_img = img[(ybottom - margin):(ytop+margin), (xleft-margin):(xright+margin)]
    crop_height, crop_width, channels = crop_img.shape 
    #enlarge and resample
    resized_img = cv2.resize(crop_img, (crop_width*50, crop_height*50), interpolation = cv2.INTER_AREA)

    return resized_img



def circle_detection(src):

    #print(src)
    resized_height, resized_width, channels = src.shape 

    #read in gray scale
    gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)
    gray = np.array(gray , dtype=np.uint8)
    #print(type(gray))


    rows = gray.shape[0]
    #assume that the eye now occupies the center and majority of the image
    #the radius of iris should be close to either resized_height/2 or resized_width/2 
    #so use them as the detection boundary
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, rows / 8,
                               param1=100, param2=30,
                               minRadius=int(resized_height/2), maxRadius=int(resized_width/2))

    #if circles detected
    if circles is not None:
        circles = np.uint16(np.around(circles))
        circles  = np.array(circles)

        #there might be multiple circles detected
        center_xlist  = []
        center_ylist  = []
        radius_list = []

        for i in circles[0, :]:
            center = (i[0], i[1])
            radius = i[2]

            center_xlist.append(i[0])
            center_ylist.append(i[1])
            radius_list.append(i[2])

        #if multiple circles detected, we select the average centre and radius as the eventual result
        avgcenter = (int(np.average(center_xlist)), int(np.average(center_ylist)))
        avgradius = int(np.average(radius_list))
        #add a circle to highlight the region
        #cv2.circle(src,  avgcenter , avgradius, (255, 0, 255), 3)

        #cv2.imshow("detected circles", src)
        #cv2.waitKey(0)

    # if circle not detected in the region, use the centre of the entire image
    # However the avgcentre is slightly place at the bottom of the image 
    # to avoid the distortions from eyebrow and eyelash colour
    # The circle radius is also kept small to avoid distortion from pupil
    # The choice of constant ratio parameters can be arbitrary
    else: 
        avgcenter = (int(resized_width/2), int(2*resized_height/3))
        #avgradius = min(int(resized_height/2)-10, int(resized_width/2)-10)
        avgradius = min(int(resized_height/3), int(resized_width/3))

        #cv2.circle(src,  avgcenter , avgradius, (255, 0, 255), 3)
        #print('No circle')
        #cv2.imshow("detected circles", src)
        #cv2.waitKey(0)

    
    return avgcenter, avgradius, src



def crop_largest_square_within_circle(avgcenter, avgradius, src):

    square_range = int(avgradius/math.sqrt(2))
    roi_xc = avgcenter[0]
    roi_yc = avgcenter[1]
    
    #use min and max to remove the detected circle area that is out of the image
    roi_yup = max(0, roi_yc - square_range)
    roi_ydown = min(src.shape[0]-1, roi_yc + square_range)
    roi_xleft = max(0, roi_xc - square_range)
    roi_xright = min(src.shape[1]-1, roi_xc + square_range )

    square_roi = src[roi_yup : roi_ydown, roi_xleft :roi_xright]

    return square_roi


# Extract dominant colours in the image
# credit: https://stackoverflow.com/questions/43111029/how-to-find-the-average-colour-of-an-image-in-python-with-opencv
def extract_dominant_colour(square_roi,  n_colors):

    roi_pixels = np.float32(square_roi.reshape(-1, 3))
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, .1)
    flags = cv2.KMEANS_RANDOM_CENTERS 

    #n_colors is the number of dominint colors to be detected
    _, labels, palette = cv2.kmeans(roi_pixels, n_colors, None, criteria, 10, flags)
    _, counts = np.unique(labels, return_counts=True) 

    dominant = palette[np.argmax(counts)]
    #to get value in the order of rgb, use reverse     
    dominant_rgb = list(reversed(dominant))
    #print(palette)
    #print(counts)

    return dominant_rgb























def get_righteye_roi_from_all_landmarks(landmark_features):

    #righteye roi
    rt_righteye_roi = []

    for item in landmark_features:
        #(i,j) = FACIAL_LANDMARKS_68_IDXS["right_eye"]
        #pts = item[i:j]

        #only requires 4 outer points to estimate a square region that contains the right eye
        xleft = item[36][0]
        xright = item[39][0]
        ytop = item[37][1]
        ybottom = item[39][1]
        roi = [xleft, xright, ytop, ybottom]
        rt_righteye_roi.append(roi)

        #rt_righteye_landmarks.append(pts)

    righteye_roi = np.array(rt_righteye_roi)

    return righteye_roi



