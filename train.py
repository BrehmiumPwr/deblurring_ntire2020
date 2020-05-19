import data
import os
from tqdm import tqdm
from model import DeblurModel
import tensorflow as tf
from skimage.measure import compare_ssim, compare_psnr
import numpy as np

gpus = tf.config.experimental.list_physical_devices('GPU')
[tf.config.experimental.set_memory_growth(gpu, True) for gpu in gpus]


@tf.function
def call_network(model, images_blurred, training, ensemble=True):
    if ensemble:
        images = image_augmentation(images_blurred)
    else:
        images = [images_blurred]
    results = [model.network(image, training=training)[0] for image in images]

    if ensemble:
        return image_deaugmentation(results)
    else:
        return results[0]


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


def validate(model, dataset_val, ensemble=True):
    ssims = []
    psnrs = []
    outputs = []
    gts = []
    for val_datapoint in tqdm(dataset_val, desc="validating"):
        # images_blurred = flatten_sequence_dim(val_datapoint["blurred_images"])
        # images_sharp = flatten_sequence_dim(val_datapoint["sharp_images"])
        images_blurred = val_datapoint["blurred_images"]
        images_sharp = val_datapoint["sharp_images"][:, 0, :, :, :]
        # output = model.network(images_blurred, training=False)
        output = call_network(model, images_blurred=images_blurred, training=False, ensemble=ensemble)

        if isinstance(output, list):
            output = output[-1]
        output = tf.clip_by_value(output, clip_value_min=-1, clip_value_max=1)
        # np_out = output.numpy()
        # np_target = images_sharp.numpy()
        outputs.append(output)
        gts.append(images_sharp)

    # outputs = tf.concat(outputs, axis=0)
    # gts = tf.concat(gts, axis=0)
    for x in tqdm(range(len(outputs)), desc="calculating scores"):
        ssims.append(tf.image.ssim(img1=outputs[x], img2=gts[x], max_val=2.0))
        # ssims.append(compare_ssim(X=np_target[x], Y=np_out[x], multichannel=True, data_range=2))
        psnrs.append(tf.image.psnr(a=outputs[x], b=gts[x], max_val=2.0))

    ssim = tf.reduce_mean(tf.concat(ssims, axis=0))
    # ssim = tf.math.(ssims)
    psnr = tf.reduce_mean(tf.concat(psnrs, axis=0))
    return ssim, psnr


def build_model(FLAGS):
    data_path = {
        "train": FLAGS.data_path_train,
        "val": FLAGS.data_path_val,
    }
    num_video_frames = 3
    if FLAGS.phase == "train":
        dataset_train_g = data.DeblurDataset(data_path["train"], data_split="train", num_steps=num_video_frames,
                                             batch_size=2,
                                             phase="train", random_crop=True, crop_size=320)
        dataset_train_g = dataset_train_g.get_dataset()
    # dataset_train_d = data.DeblurDataset("datasets/REDS_MotionBlur", data_split="train", num_steps=1, batch_size=2,
    #                                   phase="train", random_crop=True, crop_size=320).get_dataset()
    dataset_val = data.DeblurDataset(data_path["val"], data_split="val", num_steps=num_video_frames, batch_size=2,
                                     augment_images=False, repeat=False, shuffle=False, phase="val",
                                     step_size=10)
    dataset_val = dataset_val.get_dataset()
    model = DeblurModel(use_gan=FLAGS.use_gan, use_reconstruction=FLAGS.use_reconstruction, use_vgg=FLAGS.use_vgg,
                        hard_example_mining=FLAGS.hard_example_mining, num_steps_video=num_video_frames,
                        max_global_stride=FLAGS.global_stride,
                        disc_type="discriminator")

    output_path = os.path.join("logs", FLAGS.model_name)
    ckpt_name = "model_max_{}"

    checkpoint = model.checkpoint()
    checkpoint_manager_ssim = tf.train.CheckpointManager(checkpoint=checkpoint,
                                                         directory=output_path,
                                                         checkpoint_name=ckpt_name.format("ssim"),
                                                         max_to_keep=2)
    checkpoint_manager_psnr = tf.train.CheckpointManager(checkpoint=checkpoint,
                                                         directory=output_path,
                                                         checkpoint_name=ckpt_name.format("psnr"),
                                                         max_to_keep=2)
    model.load(output_path)

    if FLAGS.phase == "train":
        train(dataset_train_g, model, dataset_val, checkpoint_manager_ssim, checkpoint_manager_psnr, output_path)
    else:
        ssim, psnr = validate(model, dataset_val)
        print("SSIM: {:.2f}, PSNR: {:.2f}".format(ssim, psnr), flush=True)
    print("DONE")


def train(dataset_train_g, model, dataset_val, checkpoint_manager_ssim, checkpoint_manager_psnr, output_path):
    writer = tf.summary.create_file_writer(output_path)
    with writer.as_default():
        pbar = tqdm(dataset_train_g)
        max_ssim = 0
        max_psnr = 0
        for train_datapoint_g in pbar:
            step = model.step.numpy()
            images_blurred_g = train_datapoint_g["blurred_images"]
            images_sharp_g = train_datapoint_g["sharp_images"]
            loss_d, loss_g = model.train_step(images_blurred_g, images_sharp_g[:, 0, :, :, :])
            pbar.set_description("loss_g: {:.4f}, loss_d: {:.4f}".format(loss_g, loss_d))
            if step % 5000 == 1:
                # validate
                ssim, psnr = validate(model, dataset_val)
                tf.summary.scalar("metrics/ssim", data=ssim, step=step)
                tf.summary.scalar("metrics/psnr", data=psnr, step=step)
                print("SSIM: {:.2f}, PSNR: {:.2f}".format(ssim, psnr), flush=True)
                if ssim > max_ssim:
                    max_ssim = ssim
                    checkpoint_manager_ssim.save(checkpoint_number=step)
                if psnr > max_psnr:
                    max_psnr = psnr
                    checkpoint_manager_psnr.save(checkpoint_number=step)
