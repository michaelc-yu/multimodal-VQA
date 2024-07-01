
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
        questions = item['questions']
        for question in questions:
            question_txt = question['question']
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

