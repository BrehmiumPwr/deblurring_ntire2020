import tensorflow as tf
from glob import glob
from os import path
import numpy as np


class DeblurDataset(object):
    def __init__(self, data_path, data_split, num_steps, batch_size=10, augment_images=True, repeat=True,
                 shuffle=True, phase="train", step_size=1, random_crop=False, crop_size=320):
        self.data_path = data_path
        self.data_split = data_split
        self.num_steps = num_steps
        assert self.num_steps % 2 == 1
        self.batch_size = batch_size
        self.augment_images = augment_images
        self.repeat = repeat
        self.shuffle = shuffle
        self.phase = phase
        self.step_size = step_size
        self.random_crop = random_crop
        self.crop_size = crop_size

        # config augmentations
        self.brightness_delta = .2
        self.hue_delta = .1

        sharp_folder = "{}_sharp".format(self.data_split)
        blurred_folder = "{}_blur".format(self.data_split)
        self.path_to_sharp_images = path.join(self.data_path, sharp_folder, data_split, sharp_folder)
        self.path_to_blurred_images = path.join(self.data_path, blurred_folder, data_split, blurred_folder)

        self.sharp_videos = glob(path.join(self.path_to_sharp_images, "*", ""))
        self.blurred_videos = glob(path.join(self.path_to_blurred_images, "*", ""))

        # assert len(self.sharp_videos) == len(self.blurred_videos)
        print(" [*] Found {} videos".format(len(self.sharp_videos)), flush=True)

        self.sharp_filenames_by_video = {}
        self.blurred_filenames_by_video = {}
        for video in self.blurred_videos:
            blurred_files = glob(path.join(video, "*.png"))
            sharp_files = [x.replace(self.path_to_blurred_images, self.path_to_sharp_images) for x in blurred_files]
            self.sharp_filenames_by_video[
                video.replace(self.path_to_blurred_images, self.path_to_sharp_images)] = sorted(sharp_files,
                                                                                                key=lambda x: int(
                                                                                                    x[:-4].split(
                                                                                                        path.sep)[-1]))
            self.blurred_filenames_by_video[video] = sorted(blurred_files,
                                                            key=lambda x: int(x[:-4].split(path.sep)[-1]))
            print(" [**] Found {} frames in video {}".format(len(self.blurred_filenames_by_video[video]), video),
                  flush=True)

        self.data_generator = self.normal_data_generator
        # self.data_generator = self.warped_data_generator
        if self.phase.lower() == "train":
            self.dataset = self.train_dataset()
        else:
            self.dataset = self.val_dataset()

    def get_dataset(self):
        return self.dataset

    def normal_data_generator(self):
        examples_sharp = []
        examples_blurred = []
        videos = list(self.sharp_filenames_by_video.keys())
        for x in range(len(videos)):
            video = videos[x]
            filenames_sharp = np.array(self.sharp_filenames_by_video[video])
            filenames_blurred = np.array(
                self.blurred_filenames_by_video[video.replace(self.path_to_sharp_images, self.path_to_blurred_images)])
            num_files = filenames_sharp.shape[0]
            select_idxs = np.arange(start=0, stop=num_files, step=self.step_size)
            for idx in select_idxs:
                start = idx - self.num_steps // 2
                end = 1 + idx + self.num_steps // 2
                sequence_idxs = np.arange(start=start, stop=end, step=1, dtype=np.int)
                sequence_idxs = np.clip(sequence_idxs, a_min=0, a_max=num_files - 1)
                sequence_center = sequence_idxs.shape[0] // 2
                center_idx = [sequence_idxs[sequence_center]]
                examples_sharp.append(filenames_sharp[center_idx])
                examples_blurred.append(filenames_blurred[sequence_idxs])

        # return examples_sharp, examples_blurred
        for x in range(len(examples_sharp)):
            yield examples_sharp[x], examples_blurred[x]

    def warped_data_generator(self):
        examples_sharp = []
        examples_blurred = []
        videos = list(self.sharp_filenames_by_video.keys())
        for x in range(len(videos)):
            video = videos[x]
            filenames_sharp = np.array(self.sharp_filenames_by_video[video])
            filenames_blurred = np.array(
                self.blurred_filenames_by_video[video.replace(self.path_to_sharp_images, self.path_to_blurred_images)])
            num_files = filenames_sharp.shape[0]
            select_idxs = np.arange(start=0, stop=num_files, step=self.step_size)
            # seq = ['a', 'b', 'c', 'd', 'e']
            seq = ['b', 'c', 'd']
            for idx in select_idxs:
                filenames_seq_blur = filenames_blurred[idx]
                filenames_seq_blur = [filenames_seq_blur.replace(".png", "_" + x + ".png") for x in seq]
                filenames_seq_sharp = filenames_sharp[idx]
                examples_sharp.append([filenames_seq_sharp])
                examples_blurred.append(filenames_seq_blur)

        # return examples_sharp, examples_blurred
        for x in range(len(examples_sharp)):
            yield examples_sharp[x], examples_blurred[x]

    def val_dataset(self):
        # every x-th sequence from all videos
        dataset = tf.data.Dataset.from_generator(self.data_generator, output_types=(tf.string, tf.string))
        dataset = dataset.map(self.load_single_example, num_parallel_calls=1)
        dataset = dataset.cache()
        dataset = dataset.batch(self.batch_size, drop_remainder=False)
        dataset = dataset.prefetch(buffer_size=1)
        return dataset

    def train_dataset(self):
        # random video
        # def data_generator():
        #    videos = list(self.sharp_filenames_by_video.keys())
        #    for x in range(len(videos)):
        #        video = videos[x]
        #        yield self.sharp_filenames_by_video[video], self.blurred_filenames_by_video[video.replace(self.path_to_sharp_images, self.path_to_blurred_images)]

        dataset = tf.data.Dataset.from_generator(self.data_generator, output_types=(tf.string, tf.string))
        if self.shuffle:
            dataset = dataset.shuffle(buffer_size=len(self.sharp_filenames_by_video.keys()))
        if self.repeat:
            dataset = dataset.repeat()

        # choose_image
        dataset = dataset.map(self.load_single_example, num_parallel_calls=16)

        dataset = dataset.batch(self.batch_size)
        dataset = dataset.prefetch(buffer_size=1)
        return dataset

    def get_image(self, filename):
        image = tf.io.read_file(filename)
        image = tf.image.decode_png(image)
        image = tf.cast(image, dtype=tf.float32)
        image /= 255
        image.set_shape((720, 1280, 3))
        return image

    def augment_image(self, sharp_image, blurred_image):
        # randomize brightness
        delta = tf.random.uniform(shape=(), minval=-self.brightness_delta, maxval=self.brightness_delta)
        sharp_image = tf.image.adjust_brightness(sharp_image, delta)
        blurred_image = tf.image.adjust_brightness(blurred_image, delta)

        # randomize hue
        delta = tf.random.uniform(shape=(), minval=-self.hue_delta, maxval=self.hue_delta)
        sharp_image = tf.image.adjust_hue(sharp_image, delta)
        blurred_image = tf.image.adjust_hue(blurred_image, delta)

        # randomize saturation
        delta = tf.random.truncated_normal(shape=(), mean=1.0, stddev=.2, dtype=tf.float32)
        sharp_image = tf.image.adjust_saturation(sharp_image, saturation_factor=delta)
        blurred_image = tf.image.adjust_saturation(blurred_image, saturation_factor=delta)

        ## randomize contrast
        delta = tf.random.truncated_normal(shape=(), mean=1.0, stddev=.2, dtype=tf.float32)
        sharp_image = tf.image.adjust_contrast(sharp_image, contrast_factor=delta)
        blurred_image = tf.image.adjust_contrast(blurred_image, contrast_factor=delta)

        # randomize gamma
        # delta = tf.random.truncated_normal(shape=(), mean=1.0, stddev=.1)
        # sharp_image = tf.image.adjust_gamma(sharp_image, gamma=delta, gain=1.0)
        # blurred_image = tf.image.adjust_gamma(blurred_image, gamma=delta, gain=1.0)

        # random flip
        do_flip = tf.cast(tf.random.uniform(shape=(), minval=0, maxval=2, dtype=tf.int32), dtype=tf.bool)

        def flip(sharp_image, blurred_image):
            sharp_image = tf.reverse(sharp_image, [2])
            blurred_image = tf.reverse(blurred_image, [2])
            return sharp_image, blurred_image

        sharp_image, blurred_image = tf.cond(do_flip, lambda: flip(sharp_image, blurred_image),
                                             lambda: (sharp_image, blurred_image))

        sharp_image = tf.clip_by_value(sharp_image, clip_value_min=0.0, clip_value_max=1.0)
        blurred_image = tf.clip_by_value(blurred_image, clip_value_min=0.0, clip_value_max=1.0)

        return sharp_image, blurred_image

    def crop(self, sharp_image, blurred_image):
        im_shape = tf.shape(sharp_image)
        start_x = tf.random.uniform(shape=(), minval=0, maxval=im_shape[2] - self.crop_size, dtype=tf.int32)
        end_x = start_x + self.crop_size
        start_y = tf.random.uniform(shape=(), minval=0, maxval=im_shape[1] - self.crop_size, dtype=tf.int32)
        end_y = start_y + self.crop_size

        sharp_image = sharp_image[:, start_y:end_y, start_x:end_x, :]
        blurred_image = blurred_image[:, start_y:end_y, start_x:end_x, :]
        return sharp_image, blurred_image

    def select_sequence_range(self, num_files):
        random_idx = tf.random.uniform(shape=(), minval=0, maxval=num_files, dtype=tf.int32)
        start = random_idx - self.num_steps // 2
        end = 1 + random_idx + self.num_steps // 2
        sequence_idxs = tf.range(start=start, limit=end, delta=1, dtype=tf.int32)
        sequence_idxs = tf.clip_by_value(sequence_idxs, clip_value_min=0, clip_value_max=num_files - 1)
        return sequence_idxs

    def load_single_example(self, pathes_sharp, pathes_blurred):
        # if self.phase.lower() == "train":
        #    num_files = tf.shape(pathes_sharp)[0]
        #    sequence_idxs = self.select_sequence_range(num_files)
        #    center_idx = sequence_idxs[tf.shape(sequence_idxs)[0]//2]
        #    filename_sharp = tf.gather(pathes_sharp, [center_idx])
        #    filename_blurred = tf.gather(pathes_blurred, sequence_idxs)
        # else:
        filename_sharp = pathes_sharp
        filename_blurred = pathes_blurred

        image_sharp = tf.map_fn(self.get_image, elems=filename_sharp, dtype=tf.float32)
        image_blurred = tf.map_fn(self.get_image, elems=filename_blurred, dtype=tf.float32)

        if self.random_crop:
            image_sharp, image_blurred = self.crop(image_sharp, image_blurred)
        if self.augment_images:
            image_sharp, image_blurred = self.augment_image(image_sharp, image_blurred)

        image_sharp = (image_sharp * 2.0) - 1.0
        image_blurred = (image_blurred * 2.0) - 1.0

        datapoint = {}
        datapoint["sharp_filenames"] = filename_sharp
        datapoint["sharp_images"] = image_sharp

        datapoint["blurred_images"] = image_blurred
        datapoint["blurred_filenames"] = filename_blurred
        # datapoint["id_video"] =
        # if self.phase.lower() == "train":
        #    datapoint["ids_images"] = sequence_idxs
        return datapoint
