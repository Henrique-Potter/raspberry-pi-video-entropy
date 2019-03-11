import pandas as pd
import numpy as np
import math


from collections import deque as dq


def generate_max_entropy_image():
    matrix = []

    for x in range(200):
        row = []
        for y in range(300):
            pixel = np.random.randint(low=0, high=255, size=3)
            row.append(pixel)

        matrix.append(row)

    return matrix


def set_first_frame(first_matrix):
    past_frame_pixels = []

    for row in first_matrix:
        p_row = []
        for cur_pixel in row:
            pixel_deque = dq([])
            pixel_deque.append(cur_pixel)
            p_row.append(pixel_deque)

        past_frame_pixels.append(p_row)

    return past_frame_pixels


def calculate_entropy(past_frame_pixels, current_matrix, tolerance, back_history_size):

    total_entropy = 0

    height = len(current_matrix)
    width = len(current_matrix[0])

    for r_i in range(height):

        for h_i in range(width):

            equal_counter = 0.0
            for s_i in range(len(past_frame_pixels[r_i][h_i])):

                diff = np.array(past_frame_pixels[r_i][h_i][s_i]) - np.array(current_matrix[r_i][h_i])
                diff_abs = np.absolute(diff)
                over_diff = diff_abs > tolerance

                if not over_diff.any():
                    equal_counter += 1.0

            pixel_probability = (equal_counter+1)/(len(past_frame_pixels[0][0])+1)
            temp = math.log(pixel_probability, 2) * pixel_probability
            total_entropy += temp

            past_frame_pixels[r_i][h_i].append(current_matrix[r_i][h_i])
            if len(past_frame_pixels[r_i][h_i]) > back_history_size:
                past_frame_pixels[r_i][h_i].popleft()

    return total_entropy * -1


frame1 = generate_max_entropy_image()

frame_back_log = set_first_frame(frame1)

print (pd.DataFrame(frame_back_log))

frame2 = generate_max_entropy_image()

entropy = calculate_entropy(frame_back_log, frame2, 20, 30)

print(entropy)


for frames in range(3000):

    frame = generate_max_entropy_image()
    entropy = calculate_entropy(frame_back_log, frame, 20, 30)

    print (entropy)

#print pd.DataFrame(frame_back_log)








