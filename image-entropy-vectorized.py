
import numpy as np

from skimage.measure import shannon_entropy
from skimage.color import rgb2gray


def generate_max_entropy_image():
    matrix = []

    for x in range(200):
        row = []
        for y in range(300):
            pixel = np.random.randint(low=0, high=255, size=3)
            row.append(pixel)

        matrix.append(row)
    return matrix


frames_back_log = []

for f in range(30):
    frames_back_log.append(generate_max_entropy_image())

np_frames_back_log = np.array(frames_back_log)

frame1 = generate_max_entropy_image()
np_frame1 = np.array(frame1)

frames_diff = np_frames_back_log - np_frame1
frames_diff_abs = np.absolute(frames_diff)

frames_diff_mask = frames_diff_abs > [50, 50, 50]
pixels_diff_mask = frames_diff_mask[:, :, :, 0] | frames_diff_mask[:, :, :, 1] | frames_diff_mask[:, :, :, 2]

mask_summed = pixels_diff_mask.sum(0)

prob_matrix = np.absolute((mask_summed - 30.0) / 30.0)

#log_matrix = np.log2(prob_matrix)

#element_wise_entropy = np.multiply(log_matrix, prob_matrix)

#entropys = np.sum(element_wise_entropy)

grayscale = rgb2gray(np_frame1)

e = shannon_entropy(grayscale)

print(e)

#print(np_frames_back_log)