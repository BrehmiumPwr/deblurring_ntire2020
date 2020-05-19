import tensorflow as tf
import os
from model import DeblurModel
import data
import cv2
from tqdm import tqdm
import time
import numpy as np
from model_stage1 import AtrousNet_Test2


@tf.function
def call_network_stage2(model, images_blurred, training):
    images_blurred /= 127.5
    images_blurred -= 1.0
    images_blurred = tf.expand_dims(images_blurred, axis=0)
    images = image_augmentation(images_blurred)
    results = [model.network(image, training=training)[0] for image in images]
    tf_sharp_image = image_deaugmentation(results)
    tf_sharp_image = tf.clip_by_value(tf_sharp_image, clip_value_min=-1.0, clip_value_max=1.0)
    tf_sharp_image += 1.0
    tf_sharp_image *= 127.5
    return tf.cast(tf.round(tf_sharp_image), dtype=tf.uint8)


@tf.function
def call_network_stage1(model, images_blurred):
    images_blurred /= 127.5
    images_blurred -= 1.0
    images_blurred = model(images_blurred)
    images_blurred = tf.clip_by_value(images_blurred, clip_value_min=-1.0, clip_value_max=1.0)
    images_blurred += 1.0
    images_blurred *= 127.5
    return tf.round(images_blurred)

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
        self.model_stage1 = AtrousNet_Test2(num_blocks=20,
                                            pad_to_fit_global_stride=True,
                                            d_mult=64,
                                            activation=tf.nn.leaky_relu,
                                            atrousDim=[1, 2, 3, 4])
        self.model_stage1.load_weights("logs/AtrousNet_Test2/AtrousNet_Test2").expect_partial()
        self.model_stage2 = DeblurModel(use_gan=self.FLAGS.use_gan,
                                        use_reconstruction=self.FLAGS.use_reconstruction,
                                        use_vgg=self.FLAGS.use_vgg, num_steps_video=3, max_global_stride=16,
                                        disc_type="discriminator")

        self.output_path = os.path.join("logs", self.FLAGS.model_name)
        self.ckpt_name = "model_max_{}"
        self.model_stage2.load(os.path.join(".", self.output_path), expect_partial=True)

    def tf_sharpen(self, images):
        tf_images = []
        for x in range(len(images)):
            tf_img = tf.io.read_file(images[x])
            tf_img = tf.io.decode_image(tf_img, expand_animations=False)
            tf_images.append(tf_img)
        tf_images = tf.stack(tf_images, axis=0)
        tf_images = tf.cast(tf_images, dtype=tf.float32)

        tf_images_stage1 = call_network_stage1(self.model_stage1, tf_images)
        tf_images_stage2 = call_network_stage2(self.model_stage2, tf_images_stage1, training=False)
        return tf_images_stage1 , tf_images_stage2

    def sharpen_dataset(self):
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

        gt_available = False
        if self.FLAGS.data_split in ["train", "val"]:
            # we have gt available -> calculate metrics
            sharp_filenames_by_videos = dataset.sharp_filenames_by_video
            gt_available = True

        stepsize = 1

        #runtimes = []
        psnrs_stage1 = []
        ssims_stage1 = []
        psnrs_stage2 = []
        ssims_stage2 = []
        for video in tqdm(blurred_filenames_by_videos.keys()):
            # os.makedirs(video.replace(dataset_dir, output_dir), exist_ok=True)
            for idx in tqdm(range(0, len(blurred_filenames_by_videos[video]), stepsize)):
                # image_front1 = blurred_filenames_by_videos[video][max(0, idx - 2)]
                image_front2 = blurred_filenames_by_videos[video][max(0, idx - 1)]
                image_center = blurred_filenames_by_videos[video][idx]
                image_back1 = blurred_filenames_by_videos[video][min(99, idx + 1)]
                # image_back2 = blurred_filenames_by_videos[video][min(99, idx+2)]

                images = [image_front2, image_center, image_back1]
                video_id = os.path.split(image_center)[-2][-3:]
                image_id = os.path.split(image_center)[-1]
                filename = os.path.join(output_dir, video_id + "_" + image_id)
                stage1_out, stage2_out = self.tf_sharpen(images)
                if gt_available:
                    gt = tf.io.read_file(sharp_filenames_by_videos[video.replace("blur", "sharp")][idx])
                    gt = tf.io.decode_image(gt, expand_animations=False)
                    ssims_stage1.append(tf.image.ssim(img1=tf.cast(stage1_out[1], dtype=tf.uint8), img2=gt, max_val=255))
                    psnrs_stage1.append(tf.image.psnr(a=tf.cast(stage1_out[1], dtype=tf.uint8), b=gt, max_val=255))
                    ssims_stage2.append(tf.image.ssim(img1=stage2_out, img2=gt, max_val=255))
                    psnrs_stage2.append(tf.image.psnr(a=stage2_out, b=gt, max_val=255))
                sharp_image = stage2_out[0].numpy()
                #runtimes.append(time)

                cv2.imwrite(filename=filename, img=sharp_image[:, :, ::-1])
                if idx % 10 == 9:
                    # write coda lab output
                    filename_coda = os.path.join(output_dir_coda_lab, video_id + "_" + image_id)
                    cv2.imwrite(filename=filename_coda, img=sharp_image[:, :, ::-1])
        #print("average runtime per image: {}".format(np.mean(runtimes)))
        if gt_available:
            ssim_stage1 = tf.reduce_mean(tf.stack(ssims_stage1, axis=0))
            psnr_stage1 = tf.reduce_mean(tf.stack(psnrs_stage1, axis=0))
            ssim_stage2 = tf.reduce_mean(tf.stack(ssims_stage2, axis=0))
            psnr_stage2 = tf.reduce_mean(tf.stack(psnrs_stage2, axis=0))
            print("SSIM: {:.2f}, PSNR: {:.2f}".format(ssim_stage1, psnr_stage1), flush=True)
            print("SSIM: {:.2f}, PSNR: {:.2f}".format(ssim_stage2, psnr_stage2), flush=True)

