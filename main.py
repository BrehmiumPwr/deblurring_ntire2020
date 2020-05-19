import data
import os
from tqdm import tqdm
from model import DeblurModel
import tensorflow as tf
from skimage.measure import compare_ssim, compare_psnr
import numpy as np
from absl import app
from absl import flags
FLAGS = flags.FLAGS

regular_path = "datasets/REDS_MotionBlur"

flags.DEFINE_string('gpu', "3", "GPU to use. Empty string for CPU mode")
flags.DEFINE_string('model_name', "submission_stage2", "Name for the model")
flags.DEFINE_boolean('use_gan', False, "Use gan loss or not")
flags.DEFINE_boolean('use_reconstruction', True, "Use reconstruction loss or not")
flags.DEFINE_boolean('use_vgg', False, "Use vgg loss or not")

flags.DEFINE_integer('global_stride', 16, "internal downsampling factor")
flags.DEFINE_string('phase', "test", " train or val or test")
flags.DEFINE_string('data_path_train', regular_path, "where to find training data")
flags.DEFINE_string('data_path_val', regular_path, "where to find validation data")
flags.DEFINE_string('data_path_test', regular_path, "where to find test data")
flags.DEFINE_string('data_split', "test", " train or val or test or default(defaults to phase)")

flags.DEFINE_boolean('hard_example_mining', False, "use hard example mining?")

def main(argv):
    os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu
    if FLAGS.data_split == "default":
        FLAGS.data_split = FLAGS.phase
    if FLAGS.phase in ["train", "val"]:
        import train
        train.build_model(FLAGS)
    elif FLAGS.phase == "test":
        from inference_dual import Inference
        inference = Inference(FLAGS)
        inference.sharpen_dataset()


if __name__ == '__main__':
    app.run(main)
