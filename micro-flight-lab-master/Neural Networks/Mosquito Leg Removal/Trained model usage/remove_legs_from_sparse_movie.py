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

    if (min_row+350>799):
        row_offset=(min_row+350)-799
        min_row-=row_offset
    else:
        row_offset=0
    if (min_col+350>1279):
        col_offset=(min_col+350)-1279
        min_row-=col_offset
    else:
        col_offset=0
    image_downsize[(int(size_row / 2 - row / 2)+row_offset):(int(size_row / 2 + row / 2)+row_offset),
    (int(size_col / 2 - col / 2)+col_offset):(int(size_col / 2 + col / 2)+col_offset)] = cropped_image

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

    start_fr = 526
    end_fr = 560
    sparse_array = np.empty((end_fr - start_fr + 2), dtype=object)
    list_of_filenames = ['E:/Igal/2018_09_05_igal_cutleg/with_legs_A/mov2/mov2_cam2_sparse_array.mat',
                         'E:/Igal/2018_09_05_igal_cutleg/with_legs_A/mov2/mov2_cam3_sparse_array.mat',
                         'E:/Igal/2018_09_05_igal_cutleg/with_legs_A/mov2/mov2_cam4_sparse_array.mat']
    for file_name in list_of_filenames:
        mat_contents = sio.loadmat(file_name)
        sparse_array[0] = mat_contents['sparse_array'][0][0]

        for frame_ind in np.arange(start_fr, end_fr + 1):
            load_im = mat_contents['sparse_array'][frame_ind][0].toarray()
            load_im = load_im - np.min(load_im.flatten())
            load_im = load_im / np.max(load_im.flatten()) * (np.power(2, 8) - 1)
            load_im[load_im == 0] = np.power(2, 8) - 1
            # load_im[load_im == 0] = np.power(2, 16)

            orig_row, orig_col = load_im.shape
            # load_im = imageio.imread('C:/Users/noamler/Downloads/trymebig.png')
            inp_im, min_row, min_col = downsize_image(load_im, 350, 350)

            image_shape = inp_im.shape
            im_softmax = sess.run(tf.nn.softmax(wkk), {inp_tensor: [inp_im[:, :, np.newaxis]], keepy: 1})

            leg_thresh = 0.5
            im_softmax = im_softmax[:, 1].reshape(image_shape[0], image_shape[1])
            segmentation = (im_softmax > leg_thresh).reshape(image_shape[0], image_shape[1], 1)
            # segmentation = (im_softmax > 0.3).reshape(image_shape[0], image_shape[1], 1)

            inp_im[segmentation[:, :, 0]] = np.power(2, 8) - 1
            inp_im[inp_im == (np.power(2, 8) - 1)] = 0
            out_im = upsize_image(inp_im, min_row, min_col, orig_row, orig_col)
            sparse_array[frame_ind - start_fr + 1] = csr_matrix(out_im)
        sio.savemat(os.path.splitext(os.path.split(file_name)[1])[0] + '_LR' +
                    os.path.splitext(os.path.split(file_name)[1])[1], {'sparse_array': sparse_array})
        # mask = np.dot(segmentation, np.array([[221, 28, 199, 127]]))
        # mask = scipy.misc.toimage(mask, mode="RGBA")
        # stacked_img = np.stack((inp_im[:, :],) * 3, -1)
        # street_im = scipy.misc.toimage(stacked_img)
        # street_im.paste(mask, box=None, mask=mask)
        # plt.figure()
        # plt.imshow(street_im)
        # plt.show()
