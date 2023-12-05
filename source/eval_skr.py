from __future__ import print_function
from collections import Counter
import string
import re
import json
import spacy
import numpy as np
from scipy import stats
from statsmodels.stats.contingency_tables import mcnemar

nlp = spacy.load('en_core_web_md')


def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s.strip()))))


def f1_score(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def exact_match_score(prediction, ground_truth):
    return int(normalize_answer(prediction) == normalize_answer(ground_truth))


def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)


def get_answer_ref(prediction,answer):
    if 'answer is' in prediction:
        ans_index = prediction.index('answer is') + 9
        return prediction[ans_index:]
    else:
        nlp_text = nlp(prediction).sents
        sentences = [str(sent).strip() for sent in nlp_text]
        for sent in sentences:
            if answer.lower() in sent.lower():
                return answer
        return sentences[0].strip()


def evaluate_qa_ref(test_path, option):
    test_file = open(test_path, 'r')
    test_data = json.load(test_file)
    f1 = exact_match = total = 0
    for idx, test_case in enumerate(test_data):
        total += 1
        prediction = test_case[option]
        ground_truths = test_case['Gold answer']
        prediction = prediction.replace('\n', '').strip()
        prediction = get_answer_ref(prediction,ground_truths[0].strip())
        exact_match += metric_max_over_ground_truths(
            exact_match_score, prediction, ground_truths)
        f1 += metric_max_over_ground_truths(
            f1_score, prediction, ground_truths)
    exact_match = 100.0 * exact_match / total
    f1 = 100.0 * f1 / total
    print('exact match:', exact_match)
    print('f1:', f1)

option='chain_of_thought_gpt3'


cot=[]

test_path='../results/cot_dev.json'
test_file = open(test_path, 'r')
test_data = json.load(test_file)
f1 = exact_match = total = 0
for idx, test_case in enumerate(test_data):
    total += 1
    prediction = test_case[option]
    ground_truths = test_case['Gold answer']

    prediction = prediction.replace('\n', '').strip()
    prediction = get_answer_ref(prediction,ground_truths[0].strip())

    
    cot.append(metric_max_over_ground_truths(exact_match_score, prediction, ground_truths))
    exact_match += metric_max_over_ground_truths(
        exact_match_score, prediction, ground_truths)
    f1 += metric_max_over_ground_truths(
        f1_score, prediction, ground_truths)

print(test_path)
print(exact_match/total)
print(f1/total)
print('---'*10)

cot_ir=[]

test_path='../results/cot_dev_ir.json'
test_file = open(test_path, 'r')
test_data2 = json.load(test_file)
f1 = exact_match = total = 0
for idx, test_case in enumerate(test_data2):

    total += 1
    prediction = test_case[option]
    ground_truths = test_case['Gold answer']

    prediction = prediction.replace('\n', '').strip()
    prediction = get_answer_ref(prediction,ground_truths[0].strip())

    
    cot_ir.append(metric_max_over_ground_truths(exact_match_score, prediction, ground_truths))
    exact_match += metric_max_over_ground_truths(
        exact_match_score, prediction, ground_truths)
    f1 += metric_max_over_ground_truths(
        f1_score, prediction, ground_truths)

print(test_path)
print(exact_match/total)
print(f1/total)
print('---'*10)

for topk in [5,6,7,8,9,10]:

    ir_or_not=[]
    test_path='./dev_skr_knn.json' + '_knn_' + str(topk)
    print(test_path)
    test_file = open(test_path, 'r')
    test_data3 = json.load(test_file)
    upper=[]
    for idx, test_case in enumerate(test_data3):
        if test_case['cot_ir_or_not']=='Yes':
            ir_or_not.append(True)
        else:
            ir_or_not.append(False)

    assert len(cot)==len(cot_ir)==len(ir_or_not)
    assert len(test_data)==len(test_data2)==len(ir_or_not)

    c=0
    f1 = exact_match = total = 0
    for i in range(len(ir_or_not)):     
        total += 1
        ground_truths = test_data2[i]['Gold answer']
        if ir_or_not[i]:
            prediction = test_data2[i][option]
            prediction = prediction.replace('\n', '').strip()
            prediction = get_answer_ref(prediction,ground_truths[0].strip())
        else:
            prediction = test_data[i][option]
            prediction = prediction.replace('\n', '').strip()
            prediction = get_answer_ref(prediction,ground_truths[0].strip())

        exact_match += metric_max_over_ground_truths(
        exact_match_score, prediction, ground_truths)
        f1 += metric_max_over_ground_truths(
        f1_score, prediction, ground_truths)
    print(exact_match/total)
    print(f1/total)
    print('---'*10)

