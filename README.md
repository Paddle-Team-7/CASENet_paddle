# CASENet: PyTorch Implementation
This is a paddle implementation for paper [CASENet: Deep Category-Aware Semantic Edge Detection](https://arxiv.org/abs/1705.09759).

This repository has been adapted to work with the Cityscapes Dataset.

## Input pre-processing
The author's preprocessing repository was used. Instructions for use can be found in the cityscapes-preprocess directory. This is used to generate binary files for each label in the dataset. 

For data loading into hdf5 file, an hdf5 file containing these binary files needs to be generated. For this conversion, run:
```
python utils/convert_bin_to_hdf5.py
```
after updating the directory paths (use absolute paths).

## Model
The model used is the ResNet-101 variant of CASENet. 

The model configuration (.prototxt) can be found: [here.](https://github.com/Chrisding/seal/blob/master/exper/sbd/config/deploy.prototxt)

The download links for pretrained weights for CASENet can be found: [here.](https://github.com/Chrisding/seal#usage)

The Pytorch pre-trained weights can be downloaded from: [Google Drive.](https://drive.google.com/open?id=1zxshISZtq0_S6zFB37F-FhE9wT1ZBrGK)

## Training
For training, run:
```
python main.py

Optional arguments:
    -h, --help              show help message and exit
    --checkpoint-folder     path to checkpoint dir (default: ./checkpoint)
    --multigpu              use multiple GPUs (default: False)
    -j, --workers           number of data loading workers (default: 16)
    --epochs                number of total epochs to run (default: 150)
    --start-epoch           manual epoch number (useful on restarts)
    --cls-num               The number of classes (default: 19 for Cityscapes)
    --lr-steps              iterations to decay learning rate by 10
    --acc-steps             steps for Gradient accumulation  (default: 1)
    -b, --batch-size        mini-batch size (default: 1)
    --lr                    lr (default: 1e-7)
    --momentum              momentum (default: 0.9)
    --weight-decay          weight decay (default: 1e-4)
    -p, --print-freq        print frequency (default: 10)
    --resume-model          path to latest checkpoint (default: None)
    --pretrained-model      path to pretrained checkpoint (default: None)
```

## Visualization
For visualizing feature maps, ground truths and predictions, run: 
```
python visualize_multilabel.py [--model MODEL] [--image_file IMAGE] [--image_dir IMAGE_DIR] [--output_dir OUTPUT_DIR]
```

For example, to visualize on the validation set of Cityscapes dataset:
```
python visualize_multilabel.py -m pretrained_models/model_casenet.pth.tar -f leftImg8bit/val/lindau/lindau_000045_000019_leftImg8bit.png -d cityscapes-preprocess/data_proc/ -o output/ 
```

## Testing
For testing a pretrained model on new images, run:
```
python get_results_for_benchmark.py [--model MODEL] [--image_file IMAGE] [--image_dir IMAGE_DIR] [--output_dir OUTPUT_DIR]
```

For example, 
```
python get_results_for_benchmark.py -m pretrained_models/model_casenet.pth.tar -f img1.png -d images/ -o output/
```

A class wise prediction map will be generated in the output directory specified.
