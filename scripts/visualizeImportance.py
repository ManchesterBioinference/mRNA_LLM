import pandas as pd
import numpy as np
from tqdm import tqdm
import pickle
import argparse
import yaml

def get_real_score(attention_scores, kmer):
    counts = np.zeros([len(attention_scores) + kmer - 1])
    real_scores = np.zeros([len(attention_scores) + kmer - 1])

    for j in range(kmer):
        counts[j] += 1.0
        real_scores[j] += attention_scores[0]
    for i, score in enumerate(attention_scores[1:]):
        for j in range(kmer):
            counts[i+1 + j] += 1.0 # i+1 because we already added the first token
            real_scores[i+1 + j] += score
    real_scores = real_scores / counts
    return real_scores

def highlighter(i, weight, input_seq_list, max_abs_value=None):
    if max_abs_value is None:
        max_abs_value = max(abs(min(weight)), abs(max(weight)))
    curWeight = None
    if input_seq_list[0] == '[':
        if i < input_seq_list.find(']') + 1:
            curWeight = weight[0]
        else:
            curWeight = weight[i - input_seq_list.find(']')]
    else:
        curWeight = weight[i]
    
    if curWeight > max_abs_value:
        curWeight = max_abs_value
    if curWeight >= 0:
        red_intensity = int(255 * (1 - curWeight / max_abs_value))
        color = '#%02X%02X%02X' % (255, red_intensity, red_intensity)
    else:
        blue_intensity = int(255 * (1 - abs(curWeight) / max_abs_value))
        color = '#%02X%02X%02X' % (blue_intensity, blue_intensity, 255)
    
    input_seq_list[i] = input_seq_list[i].replace('T', 'U')
    if i == 0:
        input_seq_list[i] = '[CLS]'
    elif i == len(input_seq_list) - 1:
        input_seq_list[i] = 'END'
    word = '<span style="background-color:' + color + '; font-family: monospace;">' + input_seq_list[i] + '</span>'
    return word

def getOriginalSeq(input_seq_list, tokenized):
    if not tokenized:
        return input_seq_list
    text = ''
    for i in range(len(input_seq_list)):
        if input_seq_list[i][:2] == '[D':
            text += input_seq_list[i]
        else:
            text += input_seq_list[i][0]
            if i == len(input_seq_list) - 1:
                text += input_seq_list[i][1:]
    return text

def readData(path):
    tokensToScores = []
    with open(path, 'r') as f:
        count = 0
        for line in f:
            if count % 2 == 0:
                tokens = line.strip().split(',')
            else:
                scores = [float(x) for x in line.strip().split(',')]
                tokensToScores.append((tokens,scores))
            count += 1

    return tokensToScores

# integratedGradients = readData('output/importance/integratedGradient_list.csv')
# 
# text = ''
# for sample in tqdm(integratedGradients, desc='Plotting Colors'):
#     origSeq = getOriginalSeq(sample[0], True)
#     individual_nuc_scores = get_real_score(sample[1], 4)
#     text += ''.join([highlighter(k, individual_nuc_scores, origSeq.replace('T', 'U')) for k in range(len(origSeq))]) + '<br>'

parser = argparse.ArgumentParser()
parser.add_argument("--params", default='params.yaml', type=str, help="Path to the YAML file containing parameters.",)
parser.add_argument("--kmer", default=4, type=int, help="the kmer used for the model")
parser.add_argument("--SHAP", default="output/importance/shap.pkl", type=str, help="The path to the pickled SHAP data for each sample")
parser.add_argument("--scoresOnly", default="output/importance/CG3800/shapScores.npy", type=str, help="output path for SHAP scores numpy array")
parser.add_argument("--save_path", default="output/importance/CG3800/visualizeImportance.html", type=str, help="path to the output html file to visualize the SHAP importance scores")
parser.add_argument("--save_tokenized_path", default="output/importance/CG3800/visualizeImportance_tokenized.html", type=str, help="path to the output html file to visualize the SHAP importance scores")


args = parser.parse_known_args()[0]

# Read parameters from YAML file
if args.params:
    with open(args.params, 'r') as file:
        yaml_params = yaml.safe_load(file)
        for key, value in yaml_params['visualizeImportance'].items():
            parser.set_defaults(**{key: value})

args = parser.parse_args()

importance = pickle.load(open(args.SHAP, 'rb'))
for imp in importance:
    imp.scores *= len(imp.scores) # this makes the scores comparable between samples of different lengths
abs_values = [abs(min(sample.scores)) for sample in importance] + [abs(max(sample.scores)) for sample in importance]
# #remove outliers [array([0.22186287]), array([0.04263635]), array([0.05845571]), array([0.03809491])]
# Q1 = np.percentile(abs_values, 25)
# Q3 = np.percentile(abs_values, 75)
# IQR = Q3 - Q1
# outlier_threshold = Q3 + 1.5 * IQR
# abs_values = [x for x in abs_values if x <= outlier_threshold]
max_abs_value = np.mean(abs_values) + 4 * np.std(abs_values) # everything above this is set to the max value
#max_abs_value = max([abs(min(sample.scores)) for sample in importance] + [abs(max(sample.scores)) for sample in importance])
scores = []

text = ''
tokenizedText = ''
count = 0
for sample in tqdm(importance, desc='Plotting Colors'):
    text += '<span style = "font-family: monospace;">' +str(sample.id)+ ': ' + str(sample.actual) + " Pred:" + str(sample.prediction) + ' </span>'
    tokenizedText += '<span style = "font-family: monospace;">' +str(sample.id)+ ': ' + str(sample.actual) + " Pred:" + str(sample.prediction) + ' </span>'
    text += ''.join([highlighter(k, sample.scores, sample.tokens, max_abs_value) for k in range(len(sample.tokens))]) + '<br>'
    tokenizedText += '|'.join([highlighter(k, sample.scores, sample.tokens, max_abs_value) for k in range(len(sample.tokens))]) + '<br>'
    scores.append(sample.scores)
    count += 1

# Write text and tokenized text to html file
with open(args.save_path, 'w') as f:
    f.write(text)

with open(args.save_tokenized_path, 'w') as f:
    f.write(tokenizedText)