import tensorflow as tf
import os
from model import DeblurModel
import data
import cv2
from tqdm import tqdm
import time
import numpy as np

@tf.function
def call_network(model, images_blurred, training):
    images = image_augmentation(images_blurred)
    results = [model.network(image, training=training)[0] for image in images]
    return image_deaugmentation(results)


def image_augmentation(image):
    # kernel is ksxksxinxout
    flip_lr = image[:, :, :, ::-1, :]
    rotate90 = tf.transpose(image, [0, 1, 3, 2, 4])[:, :, :, ::-1, :]
    rotate180 = image[:, :, ::-1, ::-1, :]
    rotate270 = tf.transpose(image, [0, 1, 3, 2, 4])[:, :, ::-1, :, :]
    flip_td = image[:, :, ::-1, :, :]
    return [image, flip_lr, rotate90, rotate180, rotate270, flip_td]


def image_deaugmentation(images):
    flip_lr = images[1][:, :, ::-1, :]
    derot90 = tf.transpose(images[2][:, :, ::-1, :], [0, 2, 1, 3])
    derot180 = images[3][:, ::-1, ::-1, :]
    derot270 = tf.transpose(images[4][:, ::-1, :, :], [0, 2, 1, 3])
    flip_td = images[5][:, ::-1, :, :]
    return tf.reduce_mean(tf.stack([images[0], flip_lr, derot90, derot180, derot270, flip_td], axis=0), axis=0)

class Inference(object):
    def __init__(self, FLAGS):
        self.FLAGS = FLAGS
        self.build_model()
        self.data_path = {
        "train": FLAGS.data_path_train,
        "val": FLAGS.data_path_val,
        "test": FLAGS.data_path_test,
    }

    def build_model(self):
        self.model = DeblurModel(use_gan=self.FLAGS.use_gan,
                            use_reconstruction=self.FLAGS.use_reconstruction,
                            use_vgg=self.FLAGS.use_vgg, num_steps_video=3, max_global_stride=16,
                            disc_type="discriminator")

        self.output_path = os.path.join("logs", self.FLAGS.model_name)
        self.ckpt_name = "model_max_{}"
        self.model.load(os.path.join(".", self.output_path), expect_partial=True)

    #@tf.function
    def tf_sharpen(self, images):
        tf_images = []
        for x in range(len(images)):
            tf_img = tf.io.read_file(images[x])
            tf_img = tf.io.decode_image(tf_img, expand_animations=False)
            tf_images.append(tf_img)

        tf_images = tf.expand_dims(tf.stack(tf_images, axis=0), axis=0)
        tf_images = tf.cast(tf_images, dtype=tf.float32)
        tf_images /= 127.5
        tf_images -= 1.0
        start = time.time()
        tf_sharp_image = call_network(self.model, tf_images, training=False)
        total = time.time() - start
        tf_sharp_image += 1.0
        tf_sharp_image *= 127.5
        sharp_image = tf.cast(tf.round(tf_sharp_image), dtype=tf.uint8)
        return sharp_image, total

    def sharpen_dataset(self):
        dataset_dir = "datasets/REDS_MotionBlur"
        output_dir = os.path.join(self.output_path, "outputs/REDS_MotionBlur")
        output_dir_coda_lab = os.path.join(self.output_path, "outputs/coda_submission")
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(output_dir_coda_lab, exist_ok=True)
        dataset = data.DeblurDataset(self.data_path[self.FLAGS.data_split],
                                           data_split=self.FLAGS.data_split,
                                           num_steps=1,
                                           phase="test",
                                           augment_images=False,
                                           shuffle=False)
        blurred_filenames_by_videos = dataset.blurred_filenames_by_video

        runtimes = []
        for video in tqdm(blurred_filenames_by_videos.keys()):
            #os.makedirs(video.replace(dataset_dir, output_dir), exist_ok=True)
            for idx in range(len(blurred_filenames_by_videos[video])):
                #image_front1 = blurred_filenames_by_videos[video][max(0, idx - 2)]
                image_front2 = blurred_filenames_by_videos[video][max(0, idx-1)]
                image_center = blurred_filenames_by_videos[video][idx]
                image_back1 = blurred_filenames_by_videos[video][min(99, idx+1)]
                #image_back2 = blurred_filenames_by_videos[video][min(99, idx+2)]

                images = [image_front2, image_center, image_back1]
                video_id = os.path.split(image_center)[-2][-3:]
                image_id = os.path.split(image_center)[-1]
                filename = os.path.join(output_dir, video_id + "_" + image_id)
                sharp_image, time = self.tf_sharpen(images)
                sharp_image = sharp_image[0].numpy()
                runtimes.append(time)

                cv2.imwrite(filename=filename, img=sharp_image[:,:,::-1])
                if idx % 10 == 9:
                    # write coda lab output
                    filename_coda = os.path.join(output_dir_coda_lab, video_id + "_" + image_id)
                    cv2.imwrite(filename=filename_coda, img=sharp_image[:, :, ::-1])
        print("average runtime per image: {}".format(np.mean(runtimes)))


