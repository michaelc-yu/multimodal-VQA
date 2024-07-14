# Visual Question Answering (PyTorch)

Visual Question Answering model that answers questions about any image.

The architecture for the model is modeled after "Bottom-Up and Top-Down Attention for Image Captioning and Visual Question Answering" by Anderson et al. 
Architecture involves using a bottom-up FasterRCNN network for vision encoding and a top-down question embedding model, and integrated using attention mechanism for visual-semantic alignment.

## Dataset
The model is trained on images and question / answer pairs from the COCO dataset. https://cocodataset.org/#home

## Model
The model uses a combined bottom-up and top-down network approach inspired by the human visual system. The bottom-up network uses a FasterRCNN for vision encoding of the image, and the top-down network uses GRU for question embedding of the question. The attention mechanism is used for visual-semantic alignment to ensure the model attends to the important parts of the image.


## Performance
The model has been trained for 10 epochs on 1800 training images. It achieves a VQA accuracy score of ~0.45 with ~500 testing samples.


## Requirements
* numpy
* torch
* Pillow
* torchvision
* scikit-learn

Install using: ```pip install -r requirements.txt```


## Usage

```python vqa.py```

## Results
| Image          | Question                             |   Answer   |
|----------------|--------------------------------------|------------|
| ![COCO_train2014_000000581153](https://github.com/user-attachments/assets/60f77ed9-35e4-4bcd-8fd2-32d8117d3973) |  What is the cat doing?              |  eating    |
| ![COCO_train2014_000000580906](https://github.com/user-attachments/assets/c5870b7f-3c3d-4b80-9e37-03b67a9a163f) |  What color is the largest vehicle?  |  orange    |
| ![COCO_train2014_000000581884](https://github.com/user-attachments/assets/51f331fd-da4e-447e-8ce4-2c05660ea289) |  How many kites are there?           |  many      |
| ![COCO_train2014_000000580933](https://github.com/user-attachments/assets/7c500b5c-c8cb-45e2-9372-e011d53eeac8) |  What is in his right hand?          |  baseball  |

## References
Peter Anderson, Xiaodong He, Chris Buehler, Damien Teney, Mark Johnson, Stephen Gould, Lei Zhang. [Bottom-Up and Top-Down Attention for Image Captioning and Visual Question Answering](https://arxiv.org/pdf/1707.07998)

Damien Teney, Peter Anderson, Xiaodong He, Anton van den Hengel. [Tips and Tricks for Visual Question Answering: Learnings from the 2017 Challenge](https://arxiv.org/pdf/1708.02711)

