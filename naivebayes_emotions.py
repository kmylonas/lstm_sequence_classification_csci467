import json
import torch
from collections import Counter, defaultdict
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm
from sklearn.metrics import f1_score

# sadness (0), joy (1), love (2), anger (3), fear (4), surprise (5)

mapping = {
    0: 'sadness',
    1: 'joy',
    2: 'love',
    3: 'anger',
    4: 'fear',
    5: 'surprise'
}


def read_file(filename):
    data = []
    with open(filename) as file:
        for line in file:
            words = line.strip().split('\t')
            data.append(words)
    return data


def get_vocabulary(data):
    return list(set(word for words in data for word in words))


def get_label_counts(train_data):
    """Count the number of examples with each label in the dataset.

    We will use a Counter object from the python collections library.
    A Counter is essentially a dictionary with a "default value" of 0
    for any key that hasn't been inserted into the dictionary.

    Args:
        train_data: A list of (words, label) pairs, where words is a list of str
    Returns:
        A Counter object mapping each label to a count.
    """
    label_counts = Counter()
    # BEGIN_SOLUTION 4a
    for label in train_data:
        label_counts[label] += 1
    # END_SOLUTION 4a
    return label_counts


def get_word_counts(X_train, y_train):
    """Count occurrences of every word with every label in the dataset.

    We will create a separate Counter object for each label.
    To do this easily, we create a defaultdict(Counter),
    which is a dictionary that will create a new Counter object whenever
    we query it with a key that isn't in the dictionary.

    Args:
        train_data: A list of (words, label) pairs, where words is a list of str
    Returns:
        A Counter object where keys are tuples of (label, word), mapped to
        the number of times that word appears in an example with that label
    """
    word_counts = defaultdict(Counter)
    # BEGIN_SOLUTION 4a
    for words, label in zip(X_train, y_train):
        for word in words:
            word_counts[label][word] += 1
    # END_SOLUTION 4a
    return word_counts


def predict(words, label_counts, word_counts, vocabulary, total_words_per_label):
    """Return the most likely label given the label_counts and word_counts.

    Args:
        words: List of words for the current input.
        label_counts: Counts for each label in the training data
        word_counts: Counts for each (label, word) pair in the training data
        vocabulary: List of all words in the vocabulary
    Returns:
        The most likely label for the given input words.
    """
    labels = list(label_counts.keys())  # A list of all the labels
    # BEGIN_SOLUTION 4a
    l = 1

    best_prob = float('-inf')
    for label in labels:
        tau_w_k = []
        pi = label_counts[label] / sum(label_counts.values())

        for word in words:
            # t = (word_counts[label][word] + l) / (sum(word_counts[label].values()) + len(vocabulary)*l)
            t = (word_counts[label][word] + l) / \
                (total_words_per_label[label] + len(vocabulary)*l)
            tau_w_k.append(t)

        tau_w_k = np.array(tau_w_k)
        cur_prob = np.sum(np.log(tau_w_k)) + np.log(pi)

        if cur_prob > best_prob:
            pred = label
            best_prob = cur_prob

    return pred


def evaluate(label_counts, word_counts, vocabulary, dataset, name, total_words_per_label, class_weights, print_confusion_matrix=False):
    num_correct = 0
    confusion_counts = Counter()
    y_true = []
    y_pred = []
    for words, label in tqdm(dataset, desc=f'Evaluating on {name}'):
        pred_label = predict(
            words, label_counts, word_counts, vocabulary, total_words_per_label)
        confusion_counts[(label, pred_label)] += 1
        if pred_label == label:
            num_correct += 1
        y_pred.append(pred_label)
        y_true.append(label)
    accuracy = 100 * num_correct / len(dataset)
    f1_scores = f1_score(y_true, y_pred, average=None,
                         labels=[0, 1, 2, 3, 4, 5])
    print(f'{name} accuracy: {num_correct}/{len(dataset)} = {accuracy:.3f}%')
    print(''.join(["class\t\t"] + [value.rjust(12)
                                   for key, value in mapping.items()]))
    print(''.join([name] + [" F1-Score"] +
                  [str("{:.3f}".format(score)).rjust(13) for score in f1_scores]))
    print(
        f'Weighted Average F1: {"{:.3f}".format(np.dot(class_weights, np.array(f1_scores)))}\n')
    if print_confusion_matrix:
        print(''.join(['actual\\predicted'] +
                      [str(label).rjust(12) for label in range(6)]))
        for true_label in range(6):
            print(''.join([str(true_label).rjust(16)] + [
                str(confusion_counts[true_label, pred_label]).rjust(12)
                for pred_label in range(6)]))


def get_class_weights(y, a):
    c = Counter(y)

    denom = sum([c[w]**a for w in range(6)])
    # class_logits = [(c[w]/c.total())**a for w in range(6)]
    class_logits = [((c[w]**a)/denom) for w in range(6)]
    # class_weights = np.array([w/sum(class_logits) for w in class_logits])
    return class_logits


def main():
    X_train = read_file("./data/X_train.tsv")
    X_dev = read_file("./data/X_dev.tsv")
    X_test = read_file("./data/X_test.tsv")

    y_train = np.load("./data/y_train.npy")
    y_dev = np.load("./data/y_dev.npy")
    y_test = np.load("./data/y_test.npy")

    class_weights = get_class_weights(y_train, a=0.25)

    # The set of words present in the training data
    vocabulary = get_vocabulary(X_train)
    label_counts = get_label_counts(y_train)
    word_counts = get_word_counts(X_train, y_train)

    total_words_per_label = dict()
    for label in label_counts.keys():
        total_words_per_label[label] = sum(word_counts[label].values())

    print(f'Trained on: {len(y_train)} examples')
    for key, value in mapping.items():
        print(f'{key}: {value} ({label_counts[key]})')
    print("\n")
    evaluate(label_counts, word_counts, vocabulary, list(zip(X_train, y_train)),
             'train', total_words_per_label, class_weights=class_weights, print_confusion_matrix=False)
    evaluate(label_counts, word_counts, vocabulary, list(zip(X_dev, y_dev)),
             'dev', total_words_per_label, class_weights=class_weights, print_confusion_matrix=True)
    evaluate(label_counts, word_counts, vocabulary, list(zip(X_test, y_test)),
             'test', total_words_per_label, class_weights=class_weights, print_confusion_matrix=True)


if __name__ == '__main__':
    main()


# ####import json
# import torch
# from collections import Counter, defaultdict
# import numpy as np
# from torch.utils.data import Dataset, DataLoader, random_split
# from tqdm import tqdm
# from sklearn.metrics import f1_score

# # sadness (0), joy (1), love (2), anger (3), fear (4), surprise (5)

# mapping = {
#     0: 'sadness',
#     1: 'joy',
#     2: 'love',
#     3: 'anger',
#     4: 'fear',
#     5: 'surprise'
# }


# def read_file(filename):
#     data = []
#     with open(filename) as file:
#         for line in file:
#             words = line.strip().split('\t')
#             data.append(words)
#     return data


# def naive_bayes():
#     def get_vocabulary(data):
#         return list(set(word for words in data for word in words))

#     def get_label_counts(train_data):
#         """Count the number of examples with each label in the dataset.

#         We will use a Counter object from the python collections library.
#         A Counter is essentially a dictionary with a "default value" of 0
#         for any key that hasn't been inserted into the dictionary.

#         Args:
#             train_data: A list of (words, label) pairs, where words is a list of str
#         Returns:
#             A Counter object mapping each label to a count.
#         """
#         label_counts = Counter()
#         # BEGIN_SOLUTION 4a
#         for label in train_data:
#             label_counts[label] += 1
#         # END_SOLUTION 4a
#         return label_counts

#     def get_word_counts(X_train, y_train):
#         """Count occurrences of every word with every label in the dataset.

#         We will create a separate Counter object for each label.
#         To do this easily, we create a defaultdict(Counter),
#         which is a dictionary that will create a new Counter object whenever
#         we query it with a key that isn't in the dictionary.

#         Args:
#             train_data: A list of (words, label) pairs, where words is a list of str
#         Returns:
#             A Counter object where keys are tuples of (label, word), mapped to
#             the number of times that word appears in an example with that label
#         """
#         word_counts = defaultdict(Counter)
#         # BEGIN_SOLUTION 4a
#         for words, label in zip(X_train, y_train):
#             for word in words:
#                 word_counts[label][word] += 1
#         # END_SOLUTION 4a
#         return word_counts

#     def predict(words, label_counts, word_counts, vocabulary, total_words_per_label):
#         """Return the most likely label given the label_counts and word_counts.

#         Args:
#             words: List of words for the current input.
#             label_counts: Counts for each label in the training data
#             word_counts: Counts for each (label, word) pair in the training data
#             vocabulary: List of all words in the vocabulary
#         Returns:
#             The most likely label for the given input words.
#         """
#         labels = list(label_counts.keys())  # A list of all the labels
#         # BEGIN_SOLUTION 4a
#         l = 1

#         best_prob = float('-inf')
#         for label in labels:
#             tau_w_k = []
#             pi = label_counts[label] / sum(label_counts.values())

#             for word in words:
#                 # t = (word_counts[label][word] + l) / (sum(word_counts[label].values()) + len(vocabulary)*l)
#                 t = (word_counts[label][word] + l) / \
#                     (total_words_per_label[label] + len(vocabulary)*l)
#                 tau_w_k.append(t)

#             tau_w_k = np.array(tau_w_k)
#             cur_prob = np.sum(np.log(tau_w_k)) + np.log(pi)

#             if cur_prob > best_prob:
#                 pred = label
#                 best_prob = cur_prob

#         return pred

#     def evaluate(label_counts, word_counts, vocabulary, dataset, name, total_words_per_label, class_weights, print_confusion_matrix=False):
#         num_correct = 0
#         confusion_counts = Counter()
#         y_true = []
#         y_pred = []
#         for words, label in tqdm(dataset, desc=f'Evaluating on {name}'):
#             pred_label = predict(
#                 words, label_counts, word_counts, vocabulary, total_words_per_label)
#             confusion_counts[(label, pred_label)] += 1
#             if pred_label == label:
#                 num_correct += 1
#             y_pred.append(pred_label)
#             y_true.append(label)
#         accuracy = 100 * num_correct / len(dataset)
#         f1_scores = f1_score(y_true, y_pred, average=None,
#                              labels=[0, 1, 2, 3, 4, 5])
#         print(f'{name} accuracy: {num_correct}/{len(dataset)} = {accuracy:.3f}%')
#         print(''.join(["class\t\t"] + [value.rjust(12)
#               for key, value in mapping.items()]))
#         print(''.join([name] + [" F1-Score"] +
#               [str("{:.3f}".format(score)).rjust(13) for score in f1_scores]))
#         print(
#             f'Weighted Average F1: {"{:.3f}".format(np.dot(class_weights, np.array(f1_scores)))}\n')
#         if print_confusion_matrix:
#             print(''.join(['actual\\predicted'] +
#                   [str(label).rjust(12) for label in range(6)]))
#             for true_label in range(6):
#                 print(''.join([str(true_label).rjust(16)] + [
#                     str(confusion_counts[true_label, pred_label]).rjust(12)
#                     for pred_label in range(6)]))

#     def get_class_weights(y, a):
#         c = Counter(y)

#         denom = sum([c[w]**a for w in range(6)])
#         # class_logits = [(c[w]/c.total())**a for w in range(6)]
#         class_logits = [((c[w]**a)/denom) for w in range(6)]
#         # class_weights = np.array([w/sum(class_logits) for w in class_logits])
#         return class_logits

#     X_train = read_file("./data/X_train.tsv")
#     X_dev = read_file("./data/X_dev.tsv")
#     X_test = read_file("./data/X_test.tsv")

#     y_train = np.load("./data/y_train.npy")
#     y_dev = np.load("./data/y_dev.npy")
#     y_test = np.load("./data/y_test.npy")

#     class_weights = get_class_weights(y_train, a=0.25)

#     # The set of words present in the training data
#     vocabulary = get_vocabulary(X_train)
#     label_counts = get_label_counts(y_train)
#     word_counts = get_word_counts(X_train, y_train)

#     total_words_per_label = dict()
#     for label in label_counts.keys():
#         total_words_per_label[label] = sum(word_counts[label].values())

#     print(f'Trained on: {len(y_train)} examples')
#     for key, value in mapping.items():
#         print(f'{key}: {value} ({label_counts[key]})')
#     print("\n")
#     evaluate(label_counts, word_counts, vocabulary, list(zip(X_train, y_train)),
#              'train', total_words_per_label, class_weights=class_weights, print_confusion_matrix=False)
#     evaluate(label_counts, word_counts, vocabulary, list(zip(X_dev, y_dev)),
#              'dev', total_words_per_label, class_weights=class_weights, print_confusion_matrix=True)
#     evaluate(label_counts, word_counts, vocabulary, list(zip(X_test, y_test)),
#              'test', total_words_per_label, class_weights=class_weights, print_confusion_matrix=True)


# def main():
#     naive_bayes()


# if __name__ == '__main__':
#     main()
