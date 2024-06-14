import os
import json
import collections

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from torchvision import transforms
import torchvision.models as models
from torchvision.models.detection import fasterrcnn_resnet50_fpn

import attention
import helpers



class VQAModel(nn.Module):
    def __init__(self, feature_dim, embed_dim, hidden_dim, vocab_size, answer_vocab_size, attention_dim, embedding_matrix):
        super(VQAModel, self).__init__()
        # embedding layer to convert words to vectors
        self.emb = nn.Embedding(vocab_size, embed_dim)
        self.emb.weight = nn.Parameter(embedding_matrix)
        self.emb.weight.requires_grad = False

        # process the sequence of word vectors
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)

        self.attn = attention.Attention(feature_dim, hidden_dim, attention_dim)

        self.dense1 = nn.Linear(hidden_dim + hidden_dim + feature_dim, hidden_dim)
        self.dense2 = nn.Linear(hidden_dim, answer_vocab_size)

    def forward(self, image_features, questions):
        # embed and process the question using LSTM
        embeddings = self.emb(questions)
        lstm_out, (question_state, _) = self.lstm(embeddings)
        question_state = question_state[-1]

        # compute attention scores for each image region based on the relevance to the question hidden state
        weighted_image_features, attention_weights = self.attn(image_features, question_state)

        # print(f"weighted image after applying attention: {weighted_image_features}")
        print(f"weighted image features shape: {weighted_image_features.shape}")
        print(f"hidden state shape: {question_state.shape}")
        # joint multimodal embedding of the question and the image
        combined = torch.cat((weighted_image_features, question_state), dim=1)
        print(f"combined shape {combined.shape}") # [batch_size, 1028]

        combined = self.dense1(combined)
        combined = F.relu(combined)
        output = self.dense2(combined)
        return output, attention_weights


class VQADataset(Dataset):
    def __init__(self, questions, annotations, image_dir, word_to_idx, answer_to_idx, transform=None):
        self.questions = questions
        self.annotations = annotations
        self.image_dir = image_dir
        self.word_to_idx = word_to_idx
        self.answer_to_idx = answer_to_idx
        self.transform = transform
        self.max_question_length = self.get_max_question_len()

    def __len__(self):
        return len(self.questions)

    def get_max_question_len(self):
        max_length = 0
        for question_data in self.questions:
            question = question_data['question']
            tokens = question.lower().strip().split()
            max_length = max(max_length, len(tokens))
        return max_length

    def preprocess_question(self, question):
        tokens = question.lower().strip().split()
        indices = [self.word_to_idx.get(token, self.word_to_idx["<UNK>"]) for token in tokens]
        if len(indices) < self.max_question_length:
            indices += [self.word_to_idx["<PAD>"]] * (self.max_question_length - len(indices))
        return indices[:self.max_question_length]

    def extract_image_id(self, filename):
        return int(filename.split('_')[-1].split('.')[0])

    def __getitem__(self, idx):
        question_data = self.questions[idx]
        question_text = question_data['question']
        question_indices = self.preprocess_question(question_text)
        
        annotation = next((ann for ann in self.annotations if ann['question_id'] == question_data['question_id']), None)
        answer_text = annotation['multiple_choice_answer']
        answer_idx = self.answer_to_idx.get(answer_text, self.answer_to_idx["<UNK>"])

        image_id = question_data['image_id']
        image_path = f"{self.image_dir}/COCO_train2014_{str(image_id).zfill(12)}.jpg"
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)

        return torch.tensor(question_indices, dtype=torch.long), torch.tensor(answer_idx, dtype=torch.long), image, image_id, question_text, answer_text



glove_file_path = "glove.6B.50d.txt"
glove_embeddings = helpers.load_glove_embeddings(glove_file_path)


with open('v2_OpenEnded_mscoco_train2014_questions.json', 'r') as f:
    questions_data = json.load(f)

with open('v2_mscoco_train2014_annotations.json', 'r') as f:
    annotations_data = json.load(f)

print(f"total of {len(questions_data['questions'])} questions")
print(f"total of {len(annotations_data['annotations'])} annotations")



image_dir = 'train2014'
image_files = os.listdir(image_dir)

def get_image_id(filename):
    prefix = "COCO_train2014_"
    suffix = ".jpg"
    image_id_str = filename[len(prefix):-len(suffix)]
    image_id = int(image_id_str)
    return image_id

image_ids = []

for filename in image_files:
    image_ids.append(get_image_id(filename))


def filter_by_image_ids(data, key, image_ids):
    return [item for item in data[key] if item['image_id'] in image_ids]

filtered_annotations = filter_by_image_ids(annotations_data, 'annotations', image_ids)
print(f"num filtered annotations: {len(filtered_annotations)}")

filtered_questions = filter_by_image_ids(questions_data, 'questions', image_ids)
print(f"num filtered questions: {len(filtered_questions)}")



vocab = helpers.create_vocab_list(filtered_questions)
word_to_idx = {word: idx for idx, word in enumerate(vocab)}
print(f"vocab: {vocab}")
print(f"length of vocab: {len(vocab)}")

answer_vocab = helpers.get_candidate_answers(filtered_annotations, threshold=1)
answer_to_idx = {answer : idx for idx, answer in enumerate(answer_vocab)}
print(f"answer_vocab: {answer_vocab}")
print(f"length of answer vocab: {len(answer_vocab)}")

# Each word is turned into a vector representation with
# a look-up table, whose entries are 300-dimensional vectors
# learned along other parameters during training.
# Those vectors are initialized with pretrained GloVe word embeddings
embedding_matrix = helpers.create_embedding_matrix(vocab, glove_embeddings, embedding_dimension=50)
print(f"embedding_matrix shape: {embedding_matrix.shape}")


# Define transformation for images
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


image_dir = 'train2014'

dataset = VQADataset(filtered_questions, filtered_annotations, image_dir, word_to_idx, answer_to_idx, transform=transform)

train_loader = DataLoader(dataset, batch_size=8, shuffle=False)


bottom_up_model = fasterrcnn_resnet50_fpn(weights='DEFAULT')
bottom_up_model.eval()

# feature_dim = 4 for 4 floats per bounding box
vqamodel = VQAModel(feature_dim=4, embed_dim=256, hidden_dim=512, vocab_size=len(vocab), answer_vocab_size=len(answer_vocab), attention_dim=128, embedding_matrix=embedding_matrix)
vqamodel.train()

print("starting to train")

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(vqamodel.parameters(), lr=0.001)

num_epochs = 5
for epoch in range(num_epochs):
    loss_accum = 0
    for questions, answers, images, image_id, question_text, answer in train_loader:
        print(f"image id: {image_id}")
        print(f"question text: {question_text}")
        print(f"answer text: {answer}")
        optimizer.zero_grad()
        image_features = bottom_up_model(images)
        image_features = image_features[0]['boxes']
        # print(f"image_features: {image_features}")

        output, _ = vqamodel(image_features, questions)
        # print(f"output: {output}")
        print(f"output shape: {output.shape}")

        predicted_answer_idx = torch.argmax(output, dim=1)
        print(f"predicted answer indices: {predicted_answer_idx}")

        predicted_answer_texts = [answer_vocab[idx.item()] for idx in predicted_answer_idx]
        print(f"Predicted answers: {predicted_answer_texts}")

        print(f"actual answers: {[answer_vocab[answer] for answer in answers]}")
        # print(f"actual answer shape: {answers.shape}")

        loss = criterion(output, answers)
        loss.backward()
        optimizer.step()
        loss_accum += loss.item()
        print("")
    print(f"epoch: {epoch+1}/{num_epochs} | loss: {loss_accum/len(train_loader)}")

