
import numpy as np
import torch
import collections
import os
from sklearn.utils import resample
import re


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

def tokenize(text):
    tokens = re.findall(r"\b\w+(?:'\w+)?\b|[^\w\s]", text.lower())
    return tokens

def create_vocab_list(all_data):
    # <UNK> for unseen words
    # <PAD> for padding questions to same length
    vocab = ["<UNK>", "<PAD>"]
    for item in all_data:
        question_txt = item['question']
        # words = question_txt.split()
        words = tokenize(question_txt)
        # print(f"in create_vocab_list, words: {words}")
        for word in words:
            word = word.lower()
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

def get_image_id(filename):
    prefix = "COCO_train2014_"
    suffix = ".jpg"
    image_id_str = filename[len(prefix):-len(suffix)]
    if image_id_str == '':
        return None
    image_id = int(image_id_str)
    return image_id


def populate_dict_with_data(image_files, questions_data, annotations_data):
    data = collections.defaultdict(lambda: {'image_file': None, 'questions': [], 'answers': []})

    # Populate with image file names
    for filename in image_files:
        image_id = get_image_id(filename)
        if image_id == None:
            continue
        data[image_id]['image_file'] = filename

    # Populate with questions
    for question in questions_data['questions']:
        image_id = question['image_id']
        if image_id in data:
            data[image_id]['questions'].append(question)

    # Populate with answers
    for annotation in annotations_data['annotations']:
        image_id = annotation['image_id']
        if image_id in data:
            data[image_id]['answers'].append(annotation['multiple_choice_answer'])
    
    return data


def extract_dataset_from_img_dir(image_dir, questions_data, annotations_data, num_most_common_answers, desired_answer_counts):
    image_files = os.listdir(image_dir)

    data = populate_dict_with_data(image_files, questions_data, annotations_data)

    answer_counts = collections.Counter()
    for entry in data.values():
        for answer in entry['answers']:
            answer_counts[answer] += 1

    # print(f"Original answer distribution: {answer_counts}")

    common_answers = [item[0] for item in answer_counts.most_common(num_most_common_answers)]

    common_answers.append('UNK')

    print(f"{len(common_answers)} common answers")

    data_list = [value for key, value in data.items() if value['image_file'] is not None]

    # print("filtering questions/answers")
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

    print(f"{len(filtered_data_list)} filtered data")

    flattened_data = []
    for entry in filtered_data_list:
        image_file = entry['image_file']
        for question, answer in zip(entry['questions'], entry['answers']):
            flattened_data.append({
                'image_file': image_file,
                'question': question['question'],
                'answer': answer
            })
    # print(f"{len(flattened_data)} flattened data")

    answer_counts = collections.Counter([entry['answer'].lower() for entry in flattened_data])

    # print("Filtered answer distribution:")
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

    # print(f"{len(resampled_data)} resampled data")

    # Check the new distribution
    new_answer_counts = collections.Counter([entry['answer'].lower() for entry in resampled_data])
    # print("Resampled answer distribution:")
    # print(new_answer_counts)

    downsampled_data = []
    data_by_class_resampled = collections.defaultdict(list)

    for entry in resampled_data:
        data_by_class_resampled[entry['answer'].lower()].append(entry)

    for answer, data in data_by_class_resampled.items():
        if len(data) > desired_answer_counts:
            downsampled_data.extend(resample(data, replace=False, n_samples=desired_answer_counts, random_state=42))
        else:
            downsampled_data.extend(data)

    # Check the new distribution
    final_answer_counts = collections.Counter([entry['answer'].lower() for entry in downsampled_data])
    print("Final downsampled answer distribution:")
    print(final_answer_counts)

    final_dataset = downsampled_data
    return final_dataset, common_answers

