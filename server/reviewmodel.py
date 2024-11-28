from transformers import RobertaTokenizer, RobertaForSequenceClassification,BertForSequenceClassification,BertTokenizer
from torch.nn.functional import softmax
import torch
import re
import sys


reviewmodel_path=r"Users\ayush\OneDrive\Pictures\Documents\ML\deep learning\fraud-or-not\public\models\bert_reviews"


tokenizerreview = BertTokenizer.from_pretrained(reviewmodel_path)
modelreview = BertForSequenceClassification.from_pretrained(reviewmodel_path)

def predict(text):
    inputs = tokenizerreview(text, padding=True, truncation=True, max_length=256, return_tensors="pt")

    # Get model output (logits)
    outputs = modelreview(**inputs)

    probs = outputs['logits'].softmax(1)
    pred_label_idx = probs.argmax()

    # Now map the predicted class index to the actual class label

    # Since pred_label_idx is a tensor containing a single value (the predicted class index),
    # the .item() method is used to extract the value as a scalar

    pred_label = modelreview.config.id2label[pred_label_idx.item()]

    return probs, pred_label_idx, pred_label

def dataoutput(text):
    prob,idx,predresult=predict(text)
    if predresult=='OR':
        print("Review is real")
    else:
        print("Fake review detected!")

dataoutput(sys.argv[1])

