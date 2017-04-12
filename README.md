# Tensorflow implementation of "Perceptual Losses for Real-Time Style Transfer and Super-Resolution"
Fast artistic style transfer by using feed forward network.

<img src="https://github.com/cardinalblue/neural-style/blob/master/tf_version/sample_images/tubingen.jpg?raw=true" height="200px">

<img src="https://github.com/cardinalblue/neural-style/blob/master/tf_version/sample_images/Matisse.jpg?raw=true" height="200px">
<img src="https://github.com/cardinalblue/neural-style/blob/master/tf_version/sample_images/Matisse_output.jpg?raw=true" height="200px">

<img src="https://github.com/cardinalblue/neural-style/blob/master/tf_version/sample_images/Robert_Delaunay,_1906,_Portrait_de_Metzinger,_oil_on_canvas,_55_x_43_cm,_DSC08255.jpg?raw=true" height="200px">
<img src="https://github.com/cardinalblue/neural-style/blob/master/tf_version/sample_images/RobertD_output.jpg?raw=true" height="200px">

- input image size: 1024x768
- process time(CPU): 2.246 sec (Core i5-5257U)
- process time(GPU): 1.728 sec (GPU GRID K520)


## Requirement
- [Tensorflow 1.0](https://github.com/tensorflow/tensorflow)
- [Pillow](https://github.com/python-pillow/Pillow)
- [Numpy](https://github.com/numpy/numpy)
- [Scipy](https://github.com/scipy/scipy)


## Prerequisite
In this implementation, the VGG model part was based on [Tensorflow VGG16 and VGG19](https://github.com/machrisaa/tensorflow-vgg). Please add this as a submodule and follow the instructions there to have vgg16 model. Make sure the name of the module in your project matches the one in line 6 of`custom_vgg16.py`.

## Train
Need to train one image transformation network model per one style target.
According to the paper, the models are trained on the [Microsoft COCO dataset](http://mscoco.org/dataset/#download). 
Also, it will save the transformation model, including the trained weights, for later use (in C++) in ```graphs``` directory, while the checkpoint files would be saved in ```models``` directory. 

**NOTES!**
Due to the new version of tensorflow, the checkpoint files containing weights would look something like `<model_name>.data-xxxx-of-xxxxx`. For a morde detailed description, refer to the [official page](https://www.tensorflow.org/api_docs/python/tf/train/Saver).


```
python train.py -s <style_image_path> -d <training_dataset_path> -g 0
```

## Generate

**NOTES!**
1. Please provide the model **DIRECTORY** instead of model file. 
2. Make sure you put postfix for your output image. (E.g. *.jpg, *.png...etc.)

```
python generate.py <input_image_path> -m <model_directory> -o <output_image_path>
```

## Difference from paper
- Convolution kernel size 4 instead of 3.
- Training with batchsize(n>=2) causes unstable result.

## License
MIT

## Reference
- [Perceptual Losses for Real-Time Style Transfer and Super-Resolution](http://arxiv.org/abs/1603.08155)

Codes written in this repository based on following nice works, thanks to the author.

- [Chainer implementation of "Perceptual Losses for Real-Time Style Transfer and Super-Resolution"](https://github.com/yusuketomoto/chainer-fast-neuralstyle)
