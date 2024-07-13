import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import json
import collections

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from torchvision import transforms
import torchvision.models as models
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from sklearn.utils import resample
import re

import attention
import helpers


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class VQAModel(nn.Module):
    def __init__(self, feature_dim, embed_dim, hidden_dim, vocab_size, answer_vocab_size, attention_dim, embedding_matrix):
        super(VQAModel, self).__init__()
        # embedding layer to convert words to vectors
        self.emb = nn.Embedding(vocab_size, embed_dim)
        self.emb.weight = nn.Parameter(embedding_matrix)
        self.emb.weight.requires_grad = False

        self.gru = nn.GRU(embed_dim, hidden_dim, num_layers=2, batch_first=True)

        self.attn = attention.Attention(feature_dim, hidden_dim, attention_dim)

        self.dense1 = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(p=0.5)
        self.dense2 = nn.Linear(hidden_dim, answer_vocab_size)

        self.tanh_gate_layer = nn.Linear(hidden_dim, hidden_dim)
        self.gate_layer = nn.Linear(hidden_dim, hidden_dim)

        self.proj_layer = nn.Linear(feature_dim, hidden_dim)

        self._initialize_weights()

    def _initialize_weights(self):
        # initialize weights for GRU
        for name, param in self.gru.named_parameters():
            if 'weight' in name:
                init.xavier_uniform_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)

        # initialize weights for FC layers
        init.xavier_uniform_(self.dense1.weight.data)
        self.dense1.bias.data.fill_(0)
        init.xavier_uniform_(self.dense2.weight.data)
        self.dense2.bias.data.fill_(0)

    def forward(self, image_features, questions):
        # print("in vqa forward")
        # print(f"image features: {image_features}") # image features is a tensor
        # print(f"questions: {questions}") # questions is a tensor of shape [8, 18]

        # embed and process the question using GRU
        embeddings = self.emb(questions)
        gru_out, question_state = self.gru(embeddings)
        question_state = question_state[-1]
        # print(f"question_state shape: {question_state.shape}") # [8, 512], [batch size x hidden dim]

        # compute attention scores for each image region based on the relevance to the question hidden state
        weighted_image_features, attention_weights = self.attn(image_features, question_state)

        # print(f"weighted image after applying attention: {weighted_image_features}")
        weighted_image_features = self.proj_layer(weighted_image_features)
        # print(f"weighted image features shape: {weighted_image_features.shape}") # [batch_size, 512]
        # print(f"question state shape: {question_state.shape}") # [batch_size, 512]
        # joint multimodal embedding of the question and the image should be done using an element-wise product
        combined = weighted_image_features * question_state
        # print(f"combined shape {combined.shape}") # [batch_size, 512]

        gated_tanh = torch.tanh(self.tanh_gate_layer(combined))
        gate = torch.sigmoid(self.gate_layer(combined))
        x = gated_tanh * gate
        # print(f"x shape: {x.shape}") # [batch_size, 512]

        x = F.relu(self.dense1(x))
        x = self.dropout(x)
        output = self.dense2(x)
        return output, attention_weights


class VQADataset(Dataset):
    def __init__(self, data, word_to_idx, answer_to_idx, image_dir, transform=None):
        self.data = data
        self.word_to_idx = word_to_idx
        self.answer_to_idx = answer_to_idx
        self.transform = transform
        self.max_question_length = self.get_max_question_len()
        self.image_dir = image_dir

    def __len__(self):
        return len(self.data)

    def get_max_question_len(self):
        max_length = 0
        for entry in self.data:
            question = entry['question']
            tokens = question.lower().strip().split()
            max_length = max(max_length, len(tokens))
        return max_length

    def tokenize(self, text):
        tokens = re.findall(r"\b\w+(?:'\w+)?\b|\?", text.lower())
        return tokens

    def preprocess_question(self, question):
        # print(f"in preprocess question, question: {question}")
        tokens = self.tokenize(question)
        # print(f"in preprocess question, tokens: {tokens}")
        indices = [self.word_to_idx.get(token, self.word_to_idx["<UNK>"]) for token in tokens]
        if len(indices) < self.max_question_length:
            indices += [self.word_to_idx["<PAD>"]] * (self.max_question_length - len(indices))
        return indices[:self.max_question_length]

    def __getitem__(self, idx):
        entry = self.data[idx]
        image_file = entry['image_file']
        question = entry['question']
        answer = entry['answer']

        image_path = os.path.join(self.image_dir, image_file)
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)

        question_indices = self.preprocess_question(question)
        answer_idx = self.answer_to_idx.get(answer.lower(), self.answer_to_idx["UNK"])

        return torch.tensor(question_indices, dtype=torch.long), torch.tensor(answer_idx, dtype=torch.long), image


glove_file_path = "glove.6B.50d.txt"
glove_embeddings = helpers.load_glove_embeddings(glove_file_path)


with open('v2_OpenEnded_mscoco_train2014_questions.json', 'r') as f:
    questions_data = json.load(f)

with open('v2_mscoco_train2014_annotations.json', 'r') as f:
    annotations_data = json.load(f)

print(f"total of {len(questions_data['questions'])} questions")
print(f"total of {len(annotations_data['annotations'])} annotations")



image_dir = 'train2014'
num_most_common_answers = 90
desired_answer_counts = 224

final_dataset, common_answers = helpers.extract_dataset_from_img_dir(image_dir, questions_data, annotations_data, num_most_common_answers, desired_answer_counts)

print(f"length of final dataset: {len(final_dataset)}")

vocab = helpers.create_vocab_list(final_dataset)
print(f"length of vocab: {len(vocab)}")
# print(f"vocab: {vocab}")

word_to_idx = {word: idx for idx, word in enumerate(vocab)}
idx_to_word = {idx: word for idx, word in enumerate(vocab)}
answer_to_idx = {answer: idx for idx, answer in enumerate(common_answers)}

# print(f"word to idx: {word_to_idx}")
# print(f"idx to word: {idx_to_word}")


embedding_matrix = helpers.create_embedding_matrix(vocab, glove_embeddings, embedding_dimension=50)
print(f"embedding_matrix shape: {embedding_matrix.shape}")


# Transform images to tensors
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])


batch_size = 32

dataset = VQADataset(final_dataset, word_to_idx, answer_to_idx, image_dir, transform=transform)
train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

print(f"Max question length in dataset: {dataset.get_max_question_len()}")

bottom_up_model = fasterrcnn_resnet50_fpn(weights='DEFAULT')
bottom_up_model.to(device)
bottom_up_model.eval()

# feature_dim = 4 for 4 floats per bounding box
vqamodel = VQAModel(feature_dim=4, embed_dim=50, hidden_dim=512, vocab_size=len(vocab), answer_vocab_size=len(common_answers), attention_dim=128, embedding_matrix=embedding_matrix)
vqamodel.to(device)
vqamodel.train()

print("starting to train")

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(vqamodel.parameters(), lr=0.001)
threshold = 0.7

num_epochs = 10
for epoch in range(num_epochs):
    loss_accum = 0
    for questions, answers, images in train_loader:
        # print(f"questions: {questions}")
        # print(f"answers: {answers}")
        # questions_list = questions.tolist()
        # print(f"questions list: {questions_list}")
        # questions_text = [idx_to_word[idx] for idx in questions_list[0]]
        # print(f"questions: {questions_text}")

        optimizer.zero_grad()

        images = images.to(device)
        questions = questions.to(device)
        answers = answers.to(device)

        # this is the bottom-up process that extracts the image features (bounding boxes) from raw images
        image_features = helpers.get_features_from_images(images, batch_size, bottom_up_model, threshold).to(device)

        # print(f"image_features: {image_features}")
        # print(f"len image_features: {len(image_features)}") # equal to batch size
        # print(f"image features shape: {image_features.shape}")
        # print(f"questions size: {questions.size()}") # [32, 18] -> [batch size, max q len]
        # print(f"questions shape: {questions.shape}")

        output, _ = vqamodel(image_features, questions)
        # print(f"output: {output}")
        # print(f"output shape: {output.shape}")

        predicted_answer_idx = torch.argmax(output, dim=1)
        # print(f"predicted answer indices: {predicted_answer_idx}")

        predicted_answer_texts = [common_answers[idx.item()] for idx in predicted_answer_idx]
        print(f"Predicted answers: {predicted_answer_texts}")

        print(f"Actual answers: {[common_answers[answer] for answer in answers]}")
        # print(f"actual answer shape: {answers.shape}")

        loss = criterion(output, answers)
        loss.backward()
        optimizer.step()
        loss_accum += loss.item()
        print("")
    print(f"epoch: {epoch+1}/{num_epochs} | loss: {loss_accum/len(train_loader)}")


# torch.save(vqamodel.state_dict(), 'vqamodel.pth')
# vqamodel.load_state_dict(torch.load('vqamodel.pth', map_location=torch.device('cpu')))

print("done training, starting eval")
vqamodel.eval()

max_q_len = dataset.get_max_question_len()
print(f"testing time, max q len: {max_q_len}")

test_dir = 'test-set'
num_most_common_answers = 90
desired_answer_counts = 224

test_dataset, _ = helpers.extract_dataset_from_img_dir(test_dir, questions_data, annotations_data, num_most_common_answers, desired_answer_counts)

# if any words in the question don't appear in the vocab, remove them
filtered_test_dataset = []
for x in test_dataset:
    question_words = helpers.tokenize(x['question'])
    print(f"question words: {question_words}")
    if all(word.lower() in vocab for word in question_words):
        filtered_test_dataset.append(x)

# if any answers don't appear in the training answer set, remove them
test_dataset = [x for x in filtered_test_dataset if x['answer'] in common_answers]
print(f"final test_dataset: {test_dataset}")
print(f"length final test_dataset: {len(test_dataset)}")

for x in test_dataset:
    question_words = helpers.tokenize(x['question'])
    for word in question_words:
        if word.lower() not in vocab:
            raise ValueError("word in question not in test dataset")
    answer = x['answer']
    if answer.lower() not in common_answers:
        raise ValueError("answer not in answer list")

batch_size = 1

# test_vocab = helpers.create_vocab_list(test_dataset)
word_to_idx = {word: idx for idx, word in enumerate(vocab)}

test_dataset = VQADataset(test_dataset, word_to_idx, answer_to_idx, test_dir, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

total_loss = 0.0
correct = 0
total = 0

with torch.no_grad():
    for questions, answers, images in test_loader:
        images = images.to(device)
        questions = questions.to(device)
        answers = answers.to(device)

        # questions_list = questions.tolist()
        # print(f"questions list: {questions_list}")
        # questions_text = [idx_to_word[idx] for idx in questions_list[0]]
        # print(f"questions: {questions_text}")

        # this is the bottom-up process that extracts the image features (bounding boxes) from raw images
        image_features = helpers.get_features_from_images(images, batch_size, bottom_up_model, threshold).to(device)

        output, _ = vqamodel(image_features, questions)

        predicted_answer_idx = torch.argmax(output, dim=1)
        predicted_answer_texts = [common_answers[idx.item()] for idx in predicted_answer_idx]
        print(f"Predicted answers: {predicted_answer_texts}")
        print(f"Actual answers: {[common_answers[answer] for answer in answers]}")

        loss = criterion(output, answers)
        total_loss += loss.item()
        total += answers.size(0)
        correct += (predicted_answer_idx == answers).sum().item()

average_loss = total_loss / len(test_loader)
accuracy = correct / total

print(f"average loss: {average_loss}")
print(f"accuracy: {accuracy}")


while True:
    img_file = input("Enter an image filename, or 'quit' to end: ")
    if img_file == "quit":
        break

    test_img = f"test-images/{img_file}"
    test_img = Image.open(test_img).convert('RGB')
    test_img = transform(test_img)
    test_img = test_img.unsqueeze(0).to(device)
    output = bottom_up_model(test_img)
    boxes = output[0]['boxes']
    labels = output[0]['labels']
    scores = output[0]['scores']

    filtered_boxes = [boxes[i].tolist() for i in range(len(labels)) if scores[i].item() > threshold]

    img_features = torch.tensor(filtered_boxes, dtype=torch.float32).unsqueeze(0).to(device)
    print("")

    while True:
        user_question = input("Enter a question, or 'quit' to end: ")
        if user_question == "quit":
            break
        print(user_question)

        q_tokens = user_question.lower().strip().split()
        q_indices = [word_to_idx.get(token, word_to_idx["<UNK>"]) for token in q_tokens]
        if len(q_indices) < max_q_len:
            q_indices += [word_to_idx["<PAD>"]] * (max_q_len - len(q_indices))

        # print(f"len of q indices: {len(q_indices)}")

        q_tensor = torch.tensor(q_indices, dtype=torch.long).unsqueeze(0).to(device)

        # print(f"q tensor shape: {q_tensor.shape}") # [1, 18] -> [batch_size, max q len]

        with torch.no_grad():
            output, _ = vqamodel(img_features, q_tensor)
            ans_idx = torch.argmax(output, dim=1)
            ans = common_answers[ans_idx]
            print(f"answer: {ans}")
    print("")
