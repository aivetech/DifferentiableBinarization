# DifferentiableBinarization
This is an implementation of [Real-time Scene Text Detection with Differentiable Binarization](https://arxiv.org/abs/1911.08947) in Keras and Tensorflow,
Most portions of code are borrowed from the official implementation [MhLiao/DB](https://github.com/MhLiao/DB).

## Train
`python train.py`
Here is a trained model on TotalText [baidu netdisk](https://pan.baidu.com/s/1SGKgI6pMuGvUb8RlHePQxA) code:jy6m

To train the model, use TotalText dataset as follows:
get the image lists and ground truths [here] (https://drive.google.com/drive/folders/12ozVTiBIqK8rUFWLUrlquNfoQxL2kAl7)
and get the images [here] (https://drive.google.com/file/d/1bC68CzsSVTusZVvOkk7imSZSbgD1MqK2)

Once extracted the dataset should be in `̀datasets/total_text` with the following structure
```
  datasets/total_text/train_images
  datasets/total_text/train_gts
  datasets/total_text/train_list.txt
  datasets/total_text/test_images
  datasets/total_text/test_gts
  datasets/total_text/test_list.txt
```


## Test
`python inference.py`

![image1](test/img192.jpg) 
![image2](test/img795.jpg)
![image3](test/img1095.jpg)
