from sklearn.decomposition import PCA
import numpy as np
from skimage.feature import hog
from skimage.color import rgb2gray


def extract_hog_features(images, pixels_per_cell=(8, 8), cells_per_block=(2, 2), orientations=9):
    hog_features = []
    for image in images:
        gray_image = rgb2gray(image)
        features = hog(
            gray_image,
            orientations=orientations,
            pixels_per_cell=pixels_per_cell,
            cells_per_block=cells_per_block,
            block_norm='L2-Hys'
        )
        hog_features.append(features)
    return np.array(hog_features)


def pca_for_rgb(img, n_components):
    red = img[:, :, 0]
    green = img[:, :, 1]
    blue = img[:, :, 2]

    try:
        pca = PCA(n_components=n_components)

        red_transformed = pca.fit_transform(red)
        red_inverted = pca.inverse_transform(red_transformed)

        green_transformed = pca.fit_transform(green)
        green_inverted = pca.inverse_transform(green_transformed)

        blue_transformed = pca.fit_transform(blue)
        blue_inverted = pca.inverse_transform(blue_transformed)
        img_compressed = (
            np.dstack(
                (red_inverted,
                 green_inverted,
                 blue_inverted))).astype(
            np.uint8)
        return img_compressed
    except BaseException:
        print("Kindly put valid number of components")
