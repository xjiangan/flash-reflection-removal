import cv2 as cv
import numpy as np

MAX_FEATURES = 500
GOOD_MATCH_PERCENT = 0.15
def align_imgs(im1,im2):

    # Convert images to grayscale
    im1Gray = cv.cvtColor(im1, cv.COLOR_RGB2GRAY)
    im2Gray = cv.cvtColor(im2, cv.COLOR_RGB2GRAY)

    # Detect ORB features and compute descriptors.
    orb = cv.ORB_create(MAX_FEATURES)
    keypoints1, descriptors1 = orb.detectAndCompute(im1Gray, None)
    keypoints2, descriptors2 = orb.detectAndCompute(im2Gray, None)

    # Match features.
    matcher = cv.DescriptorMatcher_create(cv.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
    matches = matcher.match(descriptors1, descriptors2, None)

    # Sort matches by score
    matches.sort(key=lambda x: x.distance, reverse=False)

    # Remove not so good matches
    numGoodMatches = int(len(matches) * GOOD_MATCH_PERCENT)
    matches = matches[:numGoodMatches]

    # Draw top matches
    imMatches = cv.drawMatches(im1, keypoints1, im2, keypoints2, matches, None)
    cv.imwrite("matches.jpg", imMatches)

    # Extract location of good matches
    points1 = np.zeros((len(matches), 2), dtype=np.float32)
    points2 = np.zeros((len(matches), 2), dtype=np.float32)

    for i, match in enumerate(matches):
        points1[i, :] = keypoints1[match.queryIdx].pt
        points2[i, :] = keypoints2[match.trainIdx].pt

    # Find homography
    h, mask = cv.findHomography(points1, points2, cv.RANSAC)

    # Use homography
    height, width, channels = im2.shape
    im1Reg = cv.warpPerspective(im1, h, (width, height))

    return im1Reg, h


##################    tests   ################################

# def test_align():
#     ls=["IMG_20210619_135712","IMG_20210619_135732","IMG_20210619_135743"]
#     ls=[osp.join('calculus','jpg',x+'.jpg') for x in ls]
#     imgs=[imageio.imread(x) for x in ls]
#     # plt.figure()
#     plt.imshow(imgs[0])
#     plt.draw()
#     input("Press Enter to continue.")
#     # plt.figure()
#     plt.imshow(imgs[1])
#     plt.draw()
#     input("Press Enter to continue.")
#     sdal,h=align_imgs(imgs[1],imgs[0])
#     # plt.figure()
#     plt.imshow(sdal)
#     plt.draw()
#     input("Press Enter to continue.")
