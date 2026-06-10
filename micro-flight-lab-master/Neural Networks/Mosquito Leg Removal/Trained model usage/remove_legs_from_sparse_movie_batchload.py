import tensorflow as tf
import imageio
import numpy as np
import scipy
import matplotlib.pyplot as plt
import scipy.io as sio
from scipy.sparse import csr_matrix
import os


def downsize_image(image, size_row, size_col):
    image_downsize = np.ones([size_row, size_col]) * (np.power(2, 8) - 1)
    ind_label = np.where(image < np.power(2, 8) - 1)
    cropped_image = image[np.min(ind_label[0]):np.max(ind_label[0]), np.min(ind_label[1]):np.max(ind_label[1])]
    [row, col] = cropped_image[:, :].shape
    min_row = np.min(ind_label[0]) - int(size_row / 2 - row / 2)
    min_col = np.min(ind_label[1]) - int(size_col / 2 - col / 2)

    if (min_row + 350 > 799):
        row_offset = (min_row + 350) - 799
        min_row -= row_offset
    else:
        row_offset = 0
    if (min_col + 350 > 1279):
        col_offset = (min_col + 350) - 1279
        min_row -= col_offset
    else:
        col_offset = 0
    image_downsize[(int(size_row / 2 - row / 2) + row_offset):(int(size_row / 2 + row / 2) + row_offset),
    (int(size_col / 2 - col / 2) + col_offset):(int(size_col / 2 + col / 2) + col_offset)] = cropped_image

    return image_downsize, min_row, min_col


def upsize_image(image, min_row, min_col, size_row, size_col):
    [row_downsize, col_downsize] = image.shape
    image_upsize = np.zeros([size_row, size_col])
    image_upsize[min_row:(min_row + row_downsize), min_col:(min_col + col_downsize)] = (
        (image) / (np.power(2, 8) - 1) * (np.power(2, 16) - 1))
    return image_upsize


with tf.Session() as sess:
    new_saver = tf.train.import_meta_graph('E:/saved_model_epoch8/segnet_model.meta')
    new_saver.restore(sess, tf.train.latest_checkpoint('E:/saved_model_epoch8/'))

    graph = tf.get_default_graph()
    # names=[tensor.name for tensor in tf.get_default_graph().as_graph_def().node]
    # matching = [s for s in some_list if "abc" in s]
    inp_tensor = graph.get_tensor_by_name("data/X_placeholder:0")
    wkk = graph.get_tensor_by_name("Reshape:0")
    keepy = graph.get_tensor_by_name("dropout:0")

    row_CNN_inp = 350  # number of rows in CNN input patch
    col_CNN_inp = 350  # number of cols in CNN input patch
    mov = 5  # number of mov
    path = 'E:/Igal/2018_09_05_igal_cutleg/with_legs_A/mov2/'  # path of files with sparse files of the cine movie

    frames_per_batch = 30  # number of frames to enter the CNN in a batch (end_fr-start_fr)/frames_per_batch=int

    start_fr = 526
    end_fr = 536
    sparse_array = np.empty((end_fr - start_fr + 2), dtype=object)
    list_of_filenames = ['E:/Igal/2018_09_05_igal_cutleg/with_legs_A/mov2/mov2_cam2_sparse_array.mat',
                         'E:/Igal/2018_09_05_igal_cutleg/with_legs_A/mov2/mov2_cam3_sparse_array.mat',
                         'E:/Igal/2018_09_05_igal_cutleg/with_legs_A/mov2/mov2_cam4_sparse_array.mat']
    for file_name in list_of_filenames:
        mat_contents = sio.loadmat(file_name)
        sparse_array[0] = mat_contents['sparse_array'][0][0]
        min_row_col = np.empty(([end_fr - start_fr + 1, 2]), dtype=object)
        array_image = []
        entire_array_image = []
        segmentation_array = np.empty((0, row_CNN_inp, col_CNN_inp), int)

        for frame_ind in np.arange(start_fr, end_fr + 1):
            load_im = mat_contents['sparse_array'][frame_ind][0].toarray()
            load_im = load_im - np.min(load_im.flatten())
            load_im = load_im / np.max(load_im.flatten()) * (np.power(2, 8) - 1)
            load_im[load_im == 0] = np.power(2, 8) - 1
            # load_im[load_im == 0] = np.power(2, 16)

            orig_row, orig_col = load_im.shape
            # load_im = imageio.imread('C:/Users/noamler/Downloads/trymebig.png')
            inp_im, min_row, min_col = downsize_image(load_im, 350, 350)
            min_row_col[frame_ind - start_fr, :] = [min_row,
                                                    min_col]  # save the location of the patch on the original image

            array_image.append(inp_im[:, :, np.newaxis])
            entire_array_image.append(inp_im)

            if (frame_ind - start_fr+1) % frames_per_batch == 0:
                # when the size of image array is the size of frames_per_batch calculate the heat map from the CNN
                # and get the mask of the wings (im_softmax>leg_thresh)
                im_softmax = sess.run(tf.nn.softmax(wkk), {inp_tensor: array_image, keepy: 1})
                leg_thresh = 0.5
                segmentation = (im_softmax[:, 1] > leg_thresh).reshape(
                    int(im_softmax.shape[0] / (row_CNN_inp * col_CNN_inp)), row_CNN_inp, col_CNN_inp)
                segmentation_array = np.append(segmentation_array, np.array(segmentation), axis=0)
                array_image = []
        if len(array_image) > 0:
            im_softmax = sess.run(tf.nn.softmax(wkk), {inp_tensor: array_image, keepy: 1})
            leg_thresh = 0.5
            segmentation = (im_softmax[:, 1] > leg_thresh).reshape(
                int(im_softmax.shape[0] / (row_CNN_inp * col_CNN_inp)), row_CNN_inp, col_CNN_inp)
            segmentation_array = np.append(segmentation_array, np.array(segmentation), axis=0)
            array_image = []

        for ind_frame in range(0, segmentation_array.shape[0]):
            # locate the mask patch in its original position on the image, save the mask as sparse and in an array
            inp_im=entire_array_image[ind_frame]
            inp_im[segmentation_array[ind_frame, :, :]] = np.power(2, 8) - 1
            inp_im[inp_im == (np.power(2, 8) - 1)] = 0
            out_im = upsize_image(inp_im, min_row_col[ind_frame, 0],
                                  min_row_col[ind_frame, 1], orig_row, orig_col)
            sparse_array[ind_frame+1] = csr_matrix(out_im)
        # save the array of masks and images in the movie path-------
        nameOfFolder = "mov%d_iniframe%d" % (mov, start_fr)
        if not os.path.exists(str(path + nameOfFolder)):
            os.makedirs(path + nameOfFolder)
        sio.savemat(os.path.splitext(os.path.split(file_name)[1])[0] + '_LR' +
                    os.path.splitext(os.path.split(file_name)[1])[1], {'sparse_array': sparse_array})
