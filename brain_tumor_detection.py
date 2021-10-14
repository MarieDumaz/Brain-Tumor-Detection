# Brain Tumor Detection

###############################################################################
''' import libraries '''
import cv2
import numpy as np
import skimage.measure
import glob
import os
from skimage.feature import greycomatrix, greycoprops
from scipy.stats import kurtosis, skew
import skimage
from skimage import measure
from skimage.measure import label, regionprops
import pandas as pd
import random
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn import neighbors, svm
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import precision_recall_curve, confusion_matrix


###############################################################################
''' Setting Randomness to create reproducible results '''
seed_value = 1792

random.seed(seed_value)
np.random.seed(seed_value)

###############################################################################
''' Set features to extract '''

# Texture Features
cr = []
cn = []
en = []
ho = []

kur = []
sk = []
mean = []
std = []
Class=[]

###############################################################################
'''Parameters'''

resize = 1 # If not 0, pictures are resized
skull_stripped = 1 # If not 0, pictures are skull stripped
save = 0 # If not 0, images after pre-processing are saved
median_filter = 1 # If not 0, a median filter is applied after skull stripping

###############################################################################
''' For all pictures with a tumor '''

# Set path
path = "./original_images/yes"

# Get all directories 
dirs = os.listdir(path)
ls =[]
for i in dirs:
    d = glob.glob(path+'/'+i+'/*.png')
    for j in d:
        ls.append(j)
        
        
count = 0
# For each image
for x in ls:
    
    c=str(count)
    #read the image
    image = cv2.imread(x);
    
    # Resize
    if resize != 0:
        image = cv2.resize(image, (256,256))
    
    # Transform image to grayscale
    grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Skull Stripping
    if skull_stripped !=0 :
        
        # Convert to binary image
        ret,thresh1 = cv2.threshold(grayImage,150,255,cv2.THRESH_BINARY)
        # Find connected components
        labeled_image = skimage.measure.label(thresh1, connectivity=2, return_num=True)
        # Create mask
        mask= labeled_image[0]<1
        # Get the original image without skull
        no_skull = grayImage*(mask)
    
        BLUR = 21
        CANNY_THRESH_1 = 10 #10
        CANNY_THRESH_2 = 200 #200
        MASK_DILATE_ITER = 5 #5
        MASK_ERODE_ITER = 20 #20
        MASK_COLOR = 0
        # Detect edges
        edges = cv2.Canny(no_skull, CANNY_THRESH_1, CANNY_THRESH_2)
        # Dilatation
        edges = cv2.dilate(edges, None)
        # Erosion
        edges = cv2.erode(edges, None)
        # Find contours
        contour_info = []
        contours, r = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        for i in contours:
            contour_info.append((
                    i,
                    cv2.isContourConvex(i),
                    cv2.contourArea(i),
                    ))
            contour_info = sorted(contour_info, key=lambda c: c[2], reverse=True)
            
        # Create mask
        max_contour = contour_info[0]
        mask = np.zeros(edges.shape)
        cv2.fillConvexPoly(mask, max_contour[0], (255))
        mask = cv2.dilate(mask, None, iterations=MASK_DILATE_ITER)
        mask = cv2.erode(mask, None, iterations=MASK_ERODE_ITER)
        mask = cv2.GaussianBlur(mask, (BLUR, BLUR), 0)
        mask = mask>0
        no_back = no_skull*mask
    
        # Median Filter
        if median_filter != 0:
            median = cv2.medianBlur(no_back,5)
            im = cv2.GaussianBlur(median,(5,5),0)
        else:
            im = no_back
    
    else:
        im = grayImage
    
    if save != 0:    
        cv2.imwrite("./preprocessedImages/"+c+"_after_preprocessing.jpg", im)
    
    count+=1
    
    
    # GLCM extraction
    glcm = greycomatrix(im, [5], [0], symmetric=True, normed=True)
    cr.append(greycoprops(glcm, 'correlation')[0,0])
    cn.append(greycoprops(glcm, 'contrast')[0,0])
    en.append(greycoprops(glcm, 'energy')[0,0])
    ho.append(greycoprops(glcm, 'homogeneity')[0,0])

    # Shape features
    m = skimage.measure.moments(im)
    c = m[0,1] / m[0,0]
    cc = m[1,0] / m[0,0]
    measure.moments_central(im, center=(c, cc), order=3)
    label_img = label(im, connectivity=im.ndim)
    props = regionprops(label_img, coordinates='xy')

    kur.append(kurtosis(im, axis=None))
    sk.append(skew(im, axis=None))
    mean.append(im.mean())
    std.append(im.std())
    
    # All images with a tumor get assigned label 1
    Class.append(1)


###############################################################################
''' For all pictures without a tumor '''

# Set path
path = "./original_images/no"

# Get all directories 
dirs = os.listdir(path)
ls =[]
for i in dirs:
    d = glob.glob(path+'/'+i+'/*.png')
    for j in d:
        ls.append(j)
        
        
count = 0
# For each image
for x in ls:
    
    c=str(count)
    #read the image
    image = cv2.imread(x);
    
    # Resize
    if resize != 0:
        image = cv2.resize(image, (256,256))

    # Transform image to grayscale
    grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Skull Stripping
    if skull_stripped !=0 :

        # Convert to binary image
        ret,thresh1 = cv2.threshold(grayImage,150,255,cv2.THRESH_BINARY)
        # Find connected components
        labeled_image = skimage.measure.label(thresh1, connectivity=2, return_num=True)
        # Create mask
        mask= labeled_image[0]<1
        # Get the original image without skull
        no_skull = grayImage*(mask)
    
        BLUR = 21
        CANNY_THRESH_1 = 10
        CANNY_THRESH_2 = 200
        MASK_DILATE_ITER = 5
        MASK_ERODE_ITER = 20
        MASK_COLOR = 0
        # Detect edges
        edges = cv2.Canny(no_skull, CANNY_THRESH_1, CANNY_THRESH_2)
        # Dilatation
        edges = cv2.dilate(edges, None)
        # Erosion
        edges = cv2.erode(edges, None)
        # Find contours
        contour_info = []
        contours, r = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        for i in contours:
            contour_info.append((
                    i,
                    cv2.isContourConvex(i),
                    cv2.contourArea(i),
                    ))
            contour_info = sorted(contour_info, key=lambda c: c[2], reverse=True)
            
        # Create mask
        max_contour = contour_info[0]
        mask = np.zeros(edges.shape)
        cv2.fillConvexPoly(mask, max_contour[0], (255))
        mask = cv2.dilate(mask, None, iterations=MASK_DILATE_ITER)
        mask = cv2.erode(mask, None, iterations=MASK_ERODE_ITER)
        mask = cv2.GaussianBlur(mask, (BLUR, BLUR), 0)
        mask = mask>0
        no_back = no_skull*mask
    
        # Median Filter
        if median_filter != 0:
            median = cv2.medianBlur(no_back,5)
            im = cv2.GaussianBlur(median,(5,5),0)
        else:
            im = no_back
    
    else:
        im = grayImage
    
    if save != 0:    
        cv2.imwrite("./preprocessedImages/"+c+"_after_preprocessing.jpg", im)
    
    count+=1
    
    
    # GLCM extraction
    glcm = greycomatrix(im, [5], [0], symmetric=True, normed=True)
    cr.append(greycoprops(glcm, 'correlation')[0,0])
    cn.append(greycoprops(glcm, 'contrast')[0,0])
    en.append(greycoprops(glcm, 'energy')[0,0])
    ho.append(greycoprops(glcm, 'homogeneity')[0,0])

    # Shape features
    m = skimage.measure.moments(im)
    c = m[0,1] / m[0,0]
    cc = m[1,0] / m[0,0]
    measure.moments_central(im, center=(c, cc), order=3)
    label_img = label(im, connectivity=im.ndim)
    props = regionprops(label_img, coordinates='xy')

    kur.append(kurtosis(im, axis=None))
    sk.append(skew(im, axis=None))
    mean.append(im.mean())
    std.append(im.std())
    
    # All images without a tumor get assigned label 0
    Class.append(0)

    
    
###############################################################################
''' Normalization '''

# Create dataframe
df_features = pd.DataFrame({'correlation': cr,
                    'contrast': cn,
                    'energy': en,
                    'homogeneity': ho,
                    'kurtosis': kur,
                    'skew': sk,
                    'standard deviation': std,
                    'mean': mean,
                    'class': Class})

# Normalize all values but Class to a range of [-1;1]    
feature = df_features.drop(['class'],1)
labels =df_features['class'] 
names = feature.columns
scaler = preprocessing.MinMaxScaler(feature_range=(-1,1))
feature[feature.columns] = scaler.fit_transform(feature[feature.columns])
scaled_df = pd.concat([feature, labels], axis=1)

# Create training instances
Y=scaled_df['class']
X=scaled_df.drop(['class'],1)

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.25, random_state=seed_value)

###############################################################################
''' Determining K to optimize accuracy (KNN parameter)'''
# Creating odd list K for KNN
n = list(range(1,50,2))
# Empty list that will hold cv scores
cv_scores = [ ]
# Perform 10-fold cross-validation
for K in n:
    knn_test = neighbors.KNeighborsClassifier(n_neighbors = K)
    scores = cross_val_score(knn_test,x_train,y_train,cv = 10,scoring ="accuracy")
    cv_scores.append(scores.mean())
# Changing to misclassification error
mse = [1-x for x in cv_scores]
# Determining best k
optimal_k = n[mse.index(min(mse))]
print("The optimal no. of neighbors is {}".format(optimal_k))
def plot_accuracy(knn_list_scores):
    pd.DataFrame({"K":[i for i in range(1,50,2)], "Accuracy":knn_list_scores}).set_index("K").plot.bar(figsize= (9,6),ylim=(0.7,0.9),rot=0)
    plt.show()
 
plot_accuracy(cv_scores)

###############################################################################
''' Determining c to optimize accuracy (SVM parameter)'''
# Creating list of C
n = list(range(10,500,10))
# Empty list that will hold cv scores
cv_scores = [ ]
# Perform 10-fold cross-validation
for c in n:
    svm_test = svm.SVC(kernel = 'poly', C=c, gamma='scale', probability=True)
    scores = cross_val_score(svm_test, x_train, y_train, cv = 10, scoring ="accuracy")
    cv_scores.append(scores.mean())
    print(c)
# Changing to misclassification error
mse = [1-x for x in cv_scores]
# Determining best c
optimal_c = n[mse.index(min(mse))]
print("The optimal c is {}".format(optimal_c))
def plot_accuracy(svm_list_scores):
    pd.DataFrame({"C":[i for i in range(10,500,10)], "Accuracy":svm_list_scores}).set_index("C").plot.bar(figsize= (9,6),ylim=(0.7,0.9),rot=0)
    plt.show()
 
plot_accuracy(cv_scores)

###############################################################################
''' K-Nearest Neighbors Classifier '''

knn = neighbors.KNeighborsClassifier(n_neighbors=optimal_k)
knn.fit(x_train, y_train)

# Probabilities
knn_probs = knn.predict_proba(x_test)
knn_probs = knn_probs[:, 1]

knn_cv_scores = cross_val_score(knn, x_train, y_train, cv=10, scoring='accuracy')


###############################################################################
''' SVM classifier '''

SVM = svm.SVC(kernel = 'poly', C=optimal_c, gamma='scale', probability=True)
SVM.fit(x_train,y_train)

# Probabilities
svm_probs = SVM.predict_proba(x_test)
svm_probs = svm_probs[:, 1]

# Cross-Validation
svm_cv_scores = cross_val_score(SVM, x_train, y_train, cv=10, scoring='accuracy')

###############################################################################
''' Determining threshold to optimize accuracy '''

threshold = np.linspace(0,1,21)
svm_t = []
knn_t = []
svm_optimal_t = 0
knn_optimal_t = 0
for t in threshold:
    
    # SVM accuracy
    svm_predicted=np.zeros(svm_probs.shape)
    svm_predicted[svm_probs>t]=1
    svm_cm = confusion_matrix(y_test, svm_predicted)
    svm_total = sum(sum(svm_cm))
    svm_accuracy = (svm_cm[0,0]+svm_cm[1,1])/svm_total
    svm_t.append(svm_accuracy)
    if(svm_accuracy == max(svm_t)):
        svm_optimal_t = t
    
    # KNN accuracy 
    knn_predicted=np.zeros(knn_probs.shape)
    knn_predicted[knn_probs>t]=1
    knn_cm = confusion_matrix(y_test, knn_predicted)
    knn_total = sum(sum(knn_cm))
    knn_accuracy = (knn_cm[0,0]+knn_cm[1,1])/knn_total
    knn_t.append(knn_accuracy)
    if(knn_accuracy == max(knn_t)):
        knn_optimal_t = t

print("The optimal threshold for SVM is {}".format(svm_optimal_t))
print("The optimal threshold for KNN is {}".format(knn_optimal_t))

plt.plot(threshold, knn_t, 'o-', label='KNN')
plt.plot(threshold, svm_t, 'o-', label='SVM')
plt.legend()

###############################################################################
''' Making predictions with best parameters '''

# KNN Predictions 
knn_predicted=np.zeros(knn_probs.shape)
knn_predicted[knn_probs>knn_optimal_t]=1

# Create a confusion matrix and calculate accuracy
knn_cm = confusion_matrix(y_test, knn_predicted)
knn_total = sum(sum(knn_cm))
knn_accuracy = (knn_cm[0,0]+knn_cm[1,1])/knn_total
knn_sensitivity = knn_cm[0,0]/(knn_cm[0,0]+knn_cm[0,1])
knn_specificity = knn_cm[1,1]/(knn_cm[1,0]+knn_cm[1,1])

# Predictions
svm_predicted=np.zeros(svm_probs.shape)
svm_predicted[svm_probs>svm_optimal_t]=1

# Create a confusion matrix and calculate accuracy
svm_cm = confusion_matrix(y_test, svm_predicted)
svm_total = sum(sum(svm_cm))
svm_accuracy = (svm_cm[0,0]+svm_cm[1,1])/svm_total
svm_sensitivity = svm_cm[0,0]/(svm_cm[0,0]+svm_cm[0,1])
svm_specificity = svm_cm[1,1]/(svm_cm[1,0]+svm_cm[1,1])

###############################################################################
''' Precision Recall Curve '''

fig=plt.figure()
svm_precision, svm_recall, _ = precision_recall_curve(y_test, svm_probs)
knn_precision, knn_recall, _ = precision_recall_curve(y_test, knn_probs)
plt.plot(svm_recall, svm_precision, 'r.-', label='SVM')
plt.plot(knn_recall, knn_precision, 'b.-', label='KNN')
plt.legend(loc='lower left', prop = {'size':10})

###############################################################################
''' Accuracy, Sensitivity, Specificity histogram '''

width=0.4
r1 = np.arange(4)
r2 = [x + width for x in r1]
fig = plt.figure(figsize=(8,8))
rects1 = plt.bar(r1, [float('%.3f'%(knn_sensitivity*100)), float('%.3f'%(knn_specificity*100)), float('%.3f'%(knn_accuracy*100)), float('%.3f'%(knn_cv_scores.mean()*100))], color='royalblue', width = width, edgecolor='black', label = 'KNN (%)')
rects2 = plt.bar(r2, [float('%.3f'%(svm_sensitivity*100)), float('%.3f'%(svm_specificity*100)), float('%.3f'%(svm_accuracy*100)), float('%.3f'%(svm_cv_scores.mean()*100))], color='orange', width = width, edgecolor='black', label = 'SVM (%)')
plt.xticks([r + 0.5*width for r in range(4)], ["Sensitivity", "Specificity", "Accuracy", "10-fold \n Cross-Validation \n Mean accuracy"], size=15)
plt.ylim(0,110)
plt.legend( prop = {'size':10})

# Function to annotate bars
def autolabel(rects):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        plt.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', size=11)


autolabel(rects1)
autolabel(rects2)

###############################################################################
#ROC curve

#fig = plt.figure()
#knn_fpr, knn_tpr, t = roc_curve(y_test, knn_probs)
#svm_fpr, svm_tpr, t = roc_curve(y_test, svm_probs)
#plt.plot(svm_fpr, svm_tpr, 'r.-', label='SVM')
#plt.plot(knn_fpr, knn_tpr, 'b.-', label='KNN')
#plt.legend(loc='lower right', prop = {'size':10})

