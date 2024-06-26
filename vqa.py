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

        # process the sequence of word vectors
        # self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.gru = nn.GRU(embed_dim, hidden_dim, num_layers=2, batch_first=True) # encode each question as the hidden state q of a GRU

        self.attn = attention.Attention(feature_dim, hidden_dim, attention_dim)

        self.dense1 = nn.Linear(hidden_dim + hidden_dim + feature_dim, hidden_dim)
        self.dropout = nn.Dropout(p=0.5) # apply dropout
        self.dense2 = nn.Linear(hidden_dim, answer_vocab_size)

        self._initialize_weights()

    def _initialize_weights(self):
        # Initializing the weights for LSTM
        for name, param in self.gru.named_parameters():
            if 'weight' in name:
                init.xavier_uniform_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)

        # Initializing weights for dense layers
        init.xavier_uniform_(self.dense1.weight.data)
        self.dense1.bias.data.fill_(0)
        init.xavier_uniform_(self.dense2.weight.data)
        self.dense2.bias.data.fill_(0)


    def forward(self, image_features, questions):
        # print("in vqa forward")
        # print(f"image features: {image_features}") # image features is a tensor
        # print(f"questions: {questions}") # questions is a tensor of shape [8, 18]

        # embed and process the question using LSTM
        embeddings = self.emb(questions)
        gru_out, question_state = self.gru(embeddings)
        question_state = question_state[-1]
        # print(f"question_state shape: {question_state.shape}") # [8, 512], [batch size x hidden dim]

        # compute attention scores for each image region based on the relevance to the question hidden state
        weighted_image_features, attention_weights = self.attn(image_features, question_state)

        # print(f"weighted image after applying attention: {weighted_image_features}")
        # print(f"weighted image features shape: {weighted_image_features.shape}")
        # print(f"hidden state shape: {question_state.shape}")
        # joint multimodal embedding of the question and the image
        combined = torch.cat((weighted_image_features, question_state), dim=1)
        # print(f"combined shape {combined.shape}") # [batch_size, 1028]

        combined = self.dense1(combined)
        combined = F.relu(combined)
        combined = self.dropout(combined) # apply dropout
        output = self.dense2(combined)
        return output, attention_weights


class VQADataset(Dataset):
    def __init__(self, data, word_to_idx, answer_to_idx, image_dir, transform=None):
        self.data = data
        self.word_to_idx = word_to_idx
        self.answer_to_idx = answer_to_idx
        self.transform = transform
        self.max_question_length = self.get_max_question_len()

    def __len__(self):
        return len(self.data)

    def get_max_question_len(self):
        max_length = 0
        for entry in self.data:
            question = entry['question']
            tokens = question.lower().strip().split()
            max_length = max(max_length, len(tokens))
        return max_length

    def preprocess_question(self, question):
        tokens = question.lower().strip().split()
        indices = [self.word_to_idx.get(token, self.word_to_idx["<UNK>"]) for token in tokens]
        if len(indices) < self.max_question_length:
            indices += [self.word_to_idx["<PAD>"]] * (self.max_question_length - len(indices))
        return indices[:self.max_question_length]

    def __getitem__(self, idx):
        entry = self.data[idx]
        image_file = entry['image_file']
        question = entry['question']
        answer = entry['answer']

        image_path = os.path.join(image_dir, image_file)
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
image_files = os.listdir(image_dir)


data = collections.defaultdict(lambda: {'image_file': None, 'questions': [], 'answers': []})

def get_image_id(filename):
    prefix = "COCO_train2014_"
    suffix = ".jpg"
    image_id_str = filename[len(prefix):-len(suffix)]
    image_id = int(image_id_str)
    return image_id

# Populate the data structure with image file names
for filename in image_files:
    image_id = get_image_id(filename)
    data[image_id]['image_file'] = filename

# Populate the data structure with questions
for question in questions_data['questions']:
    image_id = question['image_id']
    if image_id in data:
        data[image_id]['questions'].append(question)

# Populate the data structure with answers
for annotation in annotations_data['annotations']:
    image_id = annotation['image_id']
    if image_id in data:
        data[image_id]['answers'].append(annotation['multiple_choice_answer'])

answer_counts = collections.Counter()
for entry in data.values():
    for answer in entry['answers']:
        answer_counts[answer] += 1

# print(f"Answer distribution: {answer_counts}")

common_answers = ['yes', 'no', '2', '1', 'white', '3', 'blue', '0', 'black', 'red', 'brown', 'green', 'yellow', '4', 'gray', 'nothing', 'frisbee', 'baseball', 'none', 'wood', 'silver', 'pizza', '5', '7', 'grass', 'tennis', 'orange', 'skiing', 'man', 'pink', 'left', 'snowboarding', 'cat', 'donut', 'kitchen', 'right', '6', 'dog', 'sandwich', 'umbrella', 'suitcase', 'surfing', 'giraffe', 'bird', '8', 'pepperoni', 'female', 'sunny', 'china airlines', 'fork', 'bus', 'cow', '10']

common_answers.append('UNK')

print(f"{len(common_answers)} common answers")

data_list = [value for key, value in data.items() if value['image_file'] is not None]


print("filtering questions/answers")

filtered_data_list = []
for datum in data_list:
    indices_to_remove = []
    for i, answer in enumerate(datum['answers']):
        if answer not in common_answers:
            indices_to_remove.append(i)

    for index in sorted(indices_to_remove, reverse=True):
        datum['questions'].pop(index)
        datum['answers'].pop(index)
    
    if datum['questions'] and datum['answers']:
        filtered_data_list.append(datum)


# print("new data list:")
# for i in range(len(filtered_data_list)):
#     print(filtered_data_list[i])
#     print("")

print(f"{len(filtered_data_list)} filtered data")


flattened_data = []
for entry in data_list:
    image_file = entry['image_file']
    for question, answer in zip(entry['questions'], entry['answers']):
        flattened_data.append({
            'image_file': image_file,
            'question': question['question'],
            'answer': answer
        })
print(f"Total number of samples: {len(flattened_data)}")

answer_counts = collections.Counter([entry['answer'].lower() for entry in flattened_data])

print("Original answer distribution:")
print(answer_counts)

max_count = max(answer_counts.values())

# Separate data by class
data_by_class = {answer: [] for answer in answer_counts.keys()}
for entry in flattened_data:
    data_by_class[entry['answer'].lower()].append(entry)

# Resample data to match the count of the most frequent class
resampled_data = []
for answer, data in data_by_class.items():
    if len(data) < max_count:
        resampled_data.extend(resample(data, replace=True, n_samples=max_count, random_state=42))
    else:
        resampled_data.extend(data)

# Check the new distribution
new_answer_counts = collections.Counter([entry['answer'].lower() for entry in resampled_data])
print("New answer distribution:")
print(new_answer_counts)




vocab = helpers.create_vocab_list(filtered_data_list)
word_to_idx = {word: idx for idx, word in enumerate(vocab)}
# print(f"vocab: {vocab}")
print(f"length of vocab: {len(vocab)}")

answer_to_idx = {answer: idx for idx, answer in enumerate(common_answers)}

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

def list_to_tensor(tensor_list, padding_value=0):
    max_num_boxes = max(tensor.size(0) for tensor in tensor_list)

    padded_tensors = []
    for tensor in tensor_list:
        num_boxes, feature_dim = tensor.shape
        padded_tensor = torch.full((max_num_boxes, feature_dim), padding_value, dtype=tensor.dtype, device=tensor.device)
        padded_tensor[:num_boxes, :] = tensor
        padded_tensors.append(padded_tensor)

    res = torch.stack(padded_tensors, dim=0)
    return res


image_dir = "train2014"
batch_size = 8

dataset = VQADataset(resampled_data, word_to_idx, answer_to_idx, image_dir, transform=transform)
train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

bottom_up_model = fasterrcnn_resnet50_fpn(weights='DEFAULT')
bottom_up_model.to(device) # Move bottom-up model to GPU
bottom_up_model.eval()

# feature_dim = 4 for 4 floats per bounding box
# vqamodel = VQAModel(feature_dim=4, embed_dim=256, hidden_dim=512, vocab_size=len(vocab), answer_vocab_size=len(common_answers), attention_dim=128, embedding_matrix=embedding_matrix)
vqamodel = VQAModel(feature_dim=4, embed_dim=50, hidden_dim=512, vocab_size=len(vocab), answer_vocab_size=len(common_answers), attention_dim=128, embedding_matrix=embedding_matrix)
vqamodel.to(device)  # Move VQAModel to GPU
vqamodel.train()

print("starting to train")

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(vqamodel.parameters(), lr=0.001)

num_epochs = 5
for epoch in range(num_epochs):
    loss_accum = 0
    for questions, answers, images in train_loader:
        # print(f"questions: {questions}")
        # print(f"answers: {answers}")
        optimizer.zero_grad()

        images = images.to(device)
        questions = questions.to(device)
        answers = answers.to(device)

        # image_features = bottom_up_model(images)[0]['boxes'].to(device)
        # image_features = [bottom_up_model(images)[i]['boxes'].to(device) for i in range(8)]
        all_image_features = []

        for i in range(batch_size):
            image = images[i].unsqueeze(0)
            with torch.no_grad():
                output = bottom_up_model(image)
            # print(f"output image feats: {output}")
            image_features = output[0]['boxes'].to(device)
            all_image_features.append(image_features)
        
        image_features = list_to_tensor(all_image_features)

        # print(f"image_features: {image_features}")
        # print(f"len image_features: {len(image_features)}") # equal to batch size
        # print(f"image features shape: {image_features.shape}")
        # print(f"questions size: {questions.size()}")

        output, _ = vqamodel(image_features, questions)
        # print(f"output: {output}")
        # print(f"output shape: {output.shape}")

        predicted_answer_idx = torch.argmax(output, dim=1)
        # print(f"predicted answer indices: {predicted_answer_idx}")

        predicted_answer_texts = [common_answers[idx.item()] for idx in predicted_answer_idx]
        print(f"Predicted answers: {predicted_answer_texts}")

        print(f"actual answers: {[common_answers[answer] for answer in answers]}")
        # print(f"actual answer shape: {answers.shape}")

        loss = criterion(output, answers)
        loss.backward()
        optimizer.step()
        loss_accum += loss.item()
        print("")
    print(f"epoch: {epoch+1}/{num_epochs} | loss: {loss_accum/len(train_loader)}")


torch.save(vqamodel.state_dict(), 'vqamodel.pth')

# vqamodel.load_state_dict(torch.load('vqamodel.pth'))
# vqamodel.eval()

# test_img = "test-images/COCO_train2014_000000086927.jpg"

# user_question = "Are there cars in this photo?"

# test_img = transform(test_img)
# img_features = bottom_up_model(test_img)
# img_features = img_features[0]['boxes']

# q_tokens = user_question.lower().strip().split()
# q_indices = [word_to_idx.get(token, word_to_idx["<UNK>"]) for token in q_tokens]
# q_tensor = torch.tensor(q_indices, dtype=torch.long)

# output, _ = vqamodel(img_features, q_tensor)
# ans_idx = torch.argmax(output, dim=1)
# ans = answer_vocab[ans_idx]
# print(f"answer: {ans}")

