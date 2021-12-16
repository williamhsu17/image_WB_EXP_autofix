import cv2
import math
import numpy as np
import sys

def apply_mask(matrix, mask, fill_value):
    masked = np.ma.array(matrix, mask=mask, fill_value=fill_value)
    return masked.filled()

def apply_threshold(matrix, low_value, high_value):
    low_mask = matrix < low_value
    matrix = apply_mask(matrix, low_mask, low_value)

    high_mask = matrix > high_value
    matrix = apply_mask(matrix, high_mask, high_value)

    return matrix

def simplest_cb(img, percent):
    assert img.shape[2] == 3
    assert percent > 0 and percent < 100

    half_percent = percent / 200.0

    channels = cv2.split(img)

    out_channels = []
    for channel in channels:
        assert len(channel.shape) == 2
        # find the low and high precentile values (based on the input percentile)
        height, width = channel.shape
        vec_size = width * height
        flat = channel.reshape(vec_size)

        assert len(flat.shape) == 1

        flat = np.sort(flat)

        n_cols = flat.shape[0]

        low_val  = flat[math.floor(n_cols * half_percent)]
        high_val = flat[math.ceil( n_cols * (1.0 - half_percent))]

        print("Lowval: " + str(low_val))
        print("Highval: " + str(high_val))

        # saturate below the low percentile and above the high percentile
        thresholded = apply_threshold(channel, low_val, high_val)
        # scale the channel
        normalized = cv2.normalize(thresholded, thresholded.copy(), 0, 255, cv2.NORM_MINMAX)
        out_channels.append(normalized)

    return cv2.merge(out_channels)

if __name__ == '__main__':
    img1 = cv2.imread('202112132248.jpg')
    out1 = simplest_cb(img1, 1)
    vis1 = np.concatenate((img1, out1), axis=1)

    img2 = cv2.imread('202112140850.jpg')
    out2 = simplest_cb(img2, 1)
    vis2 = np.concatenate((img2, out2), axis=1)

    img3 = cv2.imread('202112150222.jpg')
    out3 = simplest_cb(img3, 1)
    vis3 = np.concatenate((img3, out3), axis=1)

    output = np.concatenate((vis1, vis2, vis3), axis=0)

    cv2.imshow("output", output)
    cv2.imwrite('output.jpg', output)
    cv2.waitKey(0)