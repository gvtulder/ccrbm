# Utility script to plot images.
#
# Copyright (c) 2016 Gijs van Tulder / Erasmus MC, the Netherlands
# This code is licensed under the MIT license. See LICENSE for details.
import matplotlib.pyplot as plt
import numpy as np

# (filters, height, width)
def plot_filters(d, normalise_pixel=False, normalise_filter=False, groupdims=False):
    if groupdims and d.ndim == 5:
        # (hidden_maps, offsets, visible_maps, filter_height, filter_width)
        d_orig_shape = d.shape
        d = d.reshape([d.shape[0]*d.shape[1]*d.shape[2], d.shape[3], d.shape[4]])

        cols_per_offset = int(np.floor(np.ceil(np.sqrt(d_orig_shape[0]*d_orig_shape[1]))/d_orig_shape[1]))
        if cols_per_offset < 1:
          cols_per_offset = 1
        maps_per_col = int(np.ceil(float(d_orig_shape[0]) / cols_per_offset))

        h, w = d.shape[1], d.shape[2]

        if normalise_pixel:
            d = (d - np.reshape(np.mean(d, axis=0), (1,h,w))) / np.reshape(np.std(d, axis=0), (1,h,w))

        img = np.zeros([d_orig_shape[0] * h, d_orig_shape[1] * w])
        img = np.zeros([maps_per_col * h, d_orig_shape[1] * cols_per_offset * w])
        for hmap in xrange(d_orig_shape[0]):
            for offset in xrange(d_orig_shape[1]):
                i = hmap * d_orig_shape[1] + offset
                if normalise_filter:
                    d[i] = (d[i] - np.mean(d[i])) / np.std(d[i])
                r = hmap % maps_per_col
                c = offset * cols_per_offset + (hmap / maps_per_col)
                img[(r*h):((r+1)*h), (c*w):((c+1)*w)] = d[i]

        plt.imshow(img, cmap=plt.cm.gray, interpolation='nearest')
        plt.yticks(np.array(range(0, maps_per_col)) * h - 0.5,
                   range(0, maps_per_col))
        plt.xticks(np.array(range(0, d_orig_shape[1])) * cols_per_offset * w - 0.5,
                   range(0, d_orig_shape[1]))
        plt.grid(color='#ff0000')
        plt.ylabel('filters')
        plt.xlabel('offsets')
        return


    if d.ndim == 4:
        # (hidden_maps, visible_maps, filter_height, filter_width)
        group_size = d.shape[0]
        d = np.swapaxes(d, 0, 1)
        d = d.reshape([d.shape[0]*d.shape[1], d.shape[2], d.shape[3]])
    elif d.ndim == 5:
        # (hidden_maps, offsets, visible_maps, filter_height, filter_width)
        group_size = d.shape[0] * d.shape[1]
        d = np.swapaxes(d, 0, 2)
        d = np.swapaxes(d, 1, 2)
        d = d.reshape([d.shape[0]*d.shape[1]*d.shape[2], d.shape[3], d.shape[4]])
    else:
        group_size = None

    sq = int(np.ceil(np.sqrt(d.shape[0])))
    h, w = d.shape[1], d.shape[2]

    if normalise_pixel:
        d = (d - np.reshape(np.mean(d, axis=0), (1,h,w))) / np.reshape(np.std(d, axis=0), (1,h,w))

    img = np.zeros([sq * h, sq * w])
    for r in xrange(sq):
        for c in xrange(sq):
            i = r+c*sq
            if i < d.shape[0]:
                if normalise_filter:
                    d[i] = (d[i] - np.mean(d[i])) / np.std(d[i])
                img[(r*h):((r+1)*h), (c*w):((c+1)*w)] = d[i]

    plt.imshow(img, cmap=plt.cm.gray, interpolation='nearest', extent=(0,sq*h,sq*h,0))
    plt.yticks(np.array(range(0,sq)) * h, range(0,sq))
    plt.xticks(np.array(range(0,sq)) * w, range(0,sq))
    plt.grid(color='#ff0000')

    for group in range(0, d.shape[0] / group_size):
        group_y = (group * group_size) % sq
        group_x = (group * group_size) / sq
        plt.hlines(group_y * h + 0.1, group_x * h + 0.1, (group_x + 1) * h, colors='g')
        plt.vlines(group_x * h + 0.1, group_y * h + 0.1, (group_y + 1) * h, colors='g')
        plt.text(group_x * h + 0.1, group_y * h + 0.1, 'v%d' % group, verticalalignment='top', size='small')


# (filters, height, width)
def filter_dot_distance(d):
    if d.ndim == 4:
        # (hidden_maps, visible_maps, filter_height, filter_width)
        d = d.reshape([d.shape[0]*d.shape[1], d.shape[2], d.shape[3]])
    elif d.ndim == 5:
        # (hidden_maps, offsets, visible_maps, filter_height, filter_width)
        d = d.reshape([d.shape[0]*d.shape[1]*d.shape[2], d.shape[3], d.shape[4]])

    for i in xrange(d.shape[0]):
        d[i] = d[i] / np.linalg.norm(d[i])

    s = 0
    for i in xrange(d.shape[0]):
        for j in xrange(d.shape[0]):
            s += sum(sum(d[i] * d[j]))
    return s

