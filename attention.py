
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torch.optim as optim
import torchvision.models as models


# Inputs: the region-based image features and the question hidden state

# Process:
# -compute attention scores for each image region based on the relevance to the question hidden state
# -generate a context vector as a weighted sum of the input features, where the weights are the attention scores

# Output: a context vector that highlights important parts of the image w.r.t question


class Attention(nn.Module):
    def __init__(self, feature_dim, hidden_dim, attention_dim):
        super(Attention, self).__init__()
        self.img_feature_proj_layer = nn.Linear(feature_dim, attention_dim)
        self.question_state_proj_layer = nn.Linear(hidden_dim, attention_dim)

        self.gated_tanh_layer = nn.Linear(attention_dim, attention_dim)
        self.gate_layer = nn.Linear(attention_dim, attention_dim)

        self.dense = nn.Linear(attention_dim, 1)

        self._initialize_weights()

    def _initialize_weights(self):
        # Initialize weights for linear layers
        init.xavier_uniform_(self.img_feature_proj_layer.weight.data)
        self.img_feature_proj_layer.bias.data.fill_(0)
        init.xavier_uniform_(self.question_state_proj_layer.weight.data)
        self.question_state_proj_layer.bias.data.fill_(0)
        init.xavier_uniform_(self.dense.weight.data)
        self.dense.bias.data.fill_(0)

    def forward(self, image_features, question_state):
        # print("In Attention forward method")
        # print(f"image_features: {image_features}")
        # print(f"question_state: {question_state}")
        # print(f"image_features shape: {image_features.shape}") # [8, 13, 4]
        # print(f"question_state shape: {question_state.shape}") # [8, 512]

        batch_size, num_boxes, feature_dim = image_features.size()
        batch_size, hidden_dim = question_state.size()

        # print(f"image_features shape: {image_features.shape}")  # [num_boxes, feature_dim]
        # print(f"question_state shape: {question_state.shape}")  # [batch_size, hidden_dim]

        # print(f"repeated image_features shape: {image_features.shape}")

        image_proj = self.img_feature_proj_layer(image_features)
        question_proj = self.question_state_proj_layer(question_state)

        # print(f"image_proj shape: {image_proj.shape}")
        # print(f"question_proj shape: {question_proj.shape}")

        # Combining projections
        question_proj = question_proj.unsqueeze(1).expand(-1, num_boxes, -1)  # [batch_size, num_boxes, attention_dim]
        # print(f"expanded question_proj shape: {question_proj.shape}")
        combined_representation = image_proj + question_proj
        gated_tanh = torch.tanh(self.gated_tanh_layer(combined_representation))
        gate = torch.sigmoid(self.gate_layer(combined_representation))
        gated_output = gated_tanh * gate

        attention_weights = F.softmax(self.dense(gated_output), dim=1)

        # print(f"attention weights shape: {attention_weights.shape}")

        weighted_image_features = (image_features * attention_weights).sum(dim=1)
        return weighted_image_features, attention_weights

