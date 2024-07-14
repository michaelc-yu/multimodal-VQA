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
| ![COCO_train2014_000000581153](https://github.com/user-attachments/assets/60f77ed9-35e4-4bcd-8fd2-32d8117d3973)
      |  What is the cat doing?              |  eating    |
| Image B        |  What color is the largest vehicle?  |  orange    |
| Image C        |  How many kites are there?           |  many      |
| Image D        |  What is in his right hand?          |  baseball  |

## References
Peter Anderson, Xiaodong He, Chris Buehler, Damien Teney, Mark Johnson, Stephen Gould, Lei Zhang. [Bottom-Up and Top-Down Attention for Image Captioning and Visual Question Answering](https://arxiv.org/pdf/1707.07998)

Tips and Tricks for Visual Question Answering: Learnings from the 2017 Challenge

