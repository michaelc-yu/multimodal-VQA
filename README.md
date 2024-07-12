Visual Question Answering (PyTorch)

Visual Question Answering model that answers questions about any image.

The architecture for the model is modeled after "Bottom-Up and Top-Down Attention for Image Captioning and Visual Question Answering" by Anderson et al. 
Architecture involves using a bottom-up FasterRCNN network for vision encoding and a top-down question embedding model, and integrated using attention mechanism for visual-semantic alignment.

Dataset
The model is trained on images and question / answer pairs from the COCO dataset. https://cocodataset.org/#home

Model


Performance


Requirements
numpy
torch
Pillow
torchvision
scikit-learn


Usage


Results


References


