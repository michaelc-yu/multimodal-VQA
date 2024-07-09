
import numpy as np
import torch
import collections



def load_glove_embeddings(glove_file_path):
    embedding_dict = {}
    with open(glove_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            vectors = np.asarray(values[1:], dtype='float32')
            embedding_dict[word] = vectors
    return embedding_dict

def create_embedding_matrix(vocab, glove_embeddings, embedding_dimension):
    embedding_matrix = np.zeros((len(vocab), embedding_dimension))
    for i, word in enumerate(vocab):
        vector = glove_embeddings.get(word)
        if vector is not None:
            embedding_matrix[i] = vector
        else:
            embedding_matrix[i] = np.zeros((embedding_dimension,))
    return torch.tensor(embedding_matrix, dtype=torch.float)

def create_vocab_list(all_data):
    # <UNK> for unseen words
    # <PAD> for padding questions to same length
    vocab = ["<UNK>", "<PAD>"]
    for item in all_data:
        question_txt = item['question']
        words = question_txt.split()
        for word in words:
            if word not in vocab:
                vocab.append(word)
    return vocab

def get_candidate_answers(annotations, threshold):
    answer_count = collections.defaultdict(int)
    for annotation in annotations:
        answer = annotation['multiple_choice_answer']
        answer_count[answer] += 1
    answer_vocab = [answer for answer, count in answer_count.items() if count >= threshold]
    answer_vocab.append("<UNK>")
    return answer_vocab


def list_to_tensor(list_of_lists, padding_value=0):
    tensor_list = []
    for boxes in list_of_lists:
        if len(boxes) == 0:
            tensor_list.append(torch.empty((0, 4), dtype=torch.float32))
        else:
            tensor_list.append(torch.tensor(boxes, dtype=torch.float32))
    max_num_boxes = max(tensor.size(0) for tensor in tensor_list)

    padded_tensors = []
    for tensor in tensor_list:
        # print(f"tensor: {tensor}")
        # print(f"tensor shape: {tensor.shape}")
        num_boxes, feature_dim = tensor.shape
        padded_tensor = torch.full((max_num_boxes, feature_dim), padding_value, dtype=tensor.dtype, device=tensor.device)
        padded_tensor[:num_boxes, :] = tensor
        padded_tensors.append(padded_tensor)

    res = torch.stack(padded_tensors, dim=0)
    return res

def get_features_from_images(images, batch_size, model, threshold):
    all_image_features = []

    for i in range(batch_size):
        image = images[i].unsqueeze(0)
        with torch.no_grad():
            output = model(image)
            boxes = output[0]['boxes']
            labels = output[0]['labels']
            scores = output[0]['scores']

            filtered_boxes = []
            for i in range(len(labels)):
                score = scores[i].item()
                if score > threshold:
                    filtered_boxes.append(boxes[i].tolist())

            all_image_features.append(filtered_boxes)

    image_features = list_to_tensor(all_image_features)
    return image_features

