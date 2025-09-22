#Step 1:
"""
Read video and downsample frames using modulo to obtain multiple frames
nth apart from each other uisng cv2.VideoCapture

"""
import cv2, os
import numpy as np

my_vid = cv2.VideoCapture("Minecraft_stitch_test.mp4") #if you pass 0 it will open webcam lol
assert my_vid.isOpened() #to make sure the video is opened otherwise don't start the script

step = 5
frames = []
i =0
while True:
    """
    .read() is a method that returns a boolean true of false if fram exists
    also returns frame width/height and color channels (BGR in OpenCv)
    """
    ok,frame = my_vid.read()
    if not ok:
        break
    if i % step ==0: #only keep frames that are step away from each other
        frames.append(frame)
    i+=1

my_vid.release()
print(len(frames),"kept")

#Step 2:
"""
Using ORB and  SIFT to try feature detection and matching
first on two frames then pick which one performs better 
to scale to n-frames
"""

"ORB (Oriented Fast and Rotated BRIEF) approach"
img0 = frames[0]
img1= frames[1]

orb = cv2.ORB_create(nfeatures = 4000)
"creates an ORB detector obj with max number of keypoints to detect 4000"
"More keypoints = more potential matches, but also slower"

k0, d0= orb.detectAndCompute(cv2.cvtColor(img0,cv2.COLOR_BGR2GRAY),None)
"""
cv2.cvtColor() this converts frame img0 from color to grayscale
feature detection works better on grayscale because it focuses 
on internsity patterns, not color

orb.detectAndCompute() this detects keypoints k0(the locations of interesting features)
compute d0 descriptors which are numerical vectors that describe what each keypoint looks like

Returns two things:
1.k0 A list of keypoint objects with x,y coordinates, orientation, etc
2.d0 A Numpy array where each row is a descriptor
"""
k1,d1 = orb.detectAndCompute(cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY),None)

bf = cv2.BFMatcher(cv2.NORM_HAMMING,crossCheck=False)
"""
Creates a Brute Force Matcher
Brute force means it compares every descriptor in image 0 to every
descriptor in image 1 to find matches

cv2.NORM_HAMMING the distance metric to compare descriptors 
ORB descriptors are binary, so Hammming distance is ideal

crossCheck = False a looser matching rule
when False: a match only has to be good in one direction, not both
this usually gives more matches but need to filter later
"""

pairs = bf.knnMatch(d0,d1,k=2)
"finds the k best matches for each descriptor (k=2 top 2 matches per descriptor)"

good = [m for m,n in pairs if m.distance < 0.75* n.distance]
"""
This is the Lowe's ratio test
for each pair of matches (m,n):
m is the best match
n is the second best match 
we keep m only if its much better than n
the threshold is 0.75 
if the best match is at least 25% better than the second best match, then its good
This helps eliminate false positives where 2 areas look similar 
but aren't the same point
End Result
good: is a filteres list of high-confidence matches between the two imgs
"""

print(len(good)) # ~= 2189

"Let's try SIFT now!"
sift = cv2.SIFT_create()
k0,d0 = sift.detectAndCompute(cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY),None)
k1,d1 = sift.detectAndCompute(cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY),None)

FLANN_INDEX_KDTREE =1
"""
SIFT gives float descriptors.To match a lot of them fast, you use FLANN
its an approximate nearest neighbor search. Instead of checking every pair like
a brute-force matcher, FLANN builds a search structure (trees) so it can find
close matches quickly with almost same quality

The index 1 selects KD Tree as the algorithm to be used for searching
KD Tree is k dimensional tree which is a type of binary search tree
but instead of just splitting numbers on one axis, it splits points is space 
based on multiple dimensions
"""
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees =5)
"""Build 5 KD trees. More trees usually means better recall but 
more memory and slower build
"""
search_params = dict(checks=100)
"""During querying, FLANN limits how many leaf nodes it visits. More checks means better matches 
but slower queries. 32-128 is a common range
"""
flann = cv2.FlannBasedMatcher(index_params, search_params)
"Creates the matcher with those index and search params"

pairs = flann.knnMatch(d0,d1, k=2)
"""For each descriptor in image 0, find its 2 nearest neighbors in image1
"""
good = [m for m,n in pairs if m.distance < 0.7 * n.distance]
"Lowe's ratio test but 0.7 is stricter then 0.75 so keep fewer but cleaner matches"

print("SIFT good matches:", len(good)) #15640

#Lets do a geometric check to make sure we only kept the best!!

pts0 = np.float32([k0[m.queryIdx].pt for m in good]).reshape(-1,1,2)
pts1 = np.float32([k1[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

#Two choices
"1- Homography: works for general perspective transformations (footage with rotation altitude changes etc)"
"2- Affine Partial 2d: simpler. assumes only scaling, rotation and translation"

#using findHomography
H, mask_H = cv2.findHomography(pts0, pts1, cv2.RANSAC,ransacReprojThreshold=5.0)
"""H is a 3 by 3 homography matrix describing how to warp frame 0 to align with frame 1
mask is A binary array where 1 means the match is an inlier (fits the model)
and o means its an outlier
"""
#using estimateAffinePartial2d
M,mask_M = cv2.estimateAffinePartial2D(pts0,pts1,method= cv2.RANSAC, ransacReprojThreshold=5.0)
"""M is the 2 by 3 affine matrix
"""

inlier_matches = [m for m, keep in zip(good, mask_H.ravel()) if keep == 1]
print("Inliers after RANSAC:", len(inlier_matches))
# Before RANSAC
img_before = cv2.drawMatches(frames[0], k0, frames[1], k1, good, None,
                             flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

# After RANSAC
img_after = cv2.drawMatches(frames[0], k0, frames[1], k1, inlier_matches, None,
                            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

# cv2.imshow("Before RANSAC", img_before)
# cv2.imshow("After RANSAC", img_after)

#Using the Built in stitcher cus feeling eepy

stitcher = cv2.Stitcher_create(cv2.Stitcher_SCANS)
status, pano = stitcher.stitch(frames)
if status != cv2.STITCHER_OK:
    raise RuntimeError(f"Stitch failed with code {status}")
#save the panorama
cv2.imwrite("panorama.jpg",pano)