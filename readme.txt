Please copy REDS Data to datasets/REDS_MotionBlur
such that the images are in datasets/REDS_MotionBlur/test_blur/test/test_blur/VIDEO_IDX/

for training:
add train_blur/train_sharp val_blur/val_sharp images in the same fashion

Saved models are located under logs/

Necessary packages can be found in requirements.txt

cmd-line for testing
python main.py --gpu=GPU_IDX --model_name=submission_stage2  --global_stride=16 --phase=test

Please cite the following
```
@InProceedings{BreScher_2020_Dual,
    author = {Brehm, Stephan and Scherer, Sebastian and Lienhart, Rainer},
    title = {High-Resolution Dual-Stage Multi-Level Feature Aggregation for Single Image and Video Deblurring},
    booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},
    month = {June},
    year = {2020}
}
```
