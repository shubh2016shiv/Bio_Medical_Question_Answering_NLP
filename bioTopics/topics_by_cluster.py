from glob import glob
from tqdm import tqdm
import json
from collections import OrderedDict
from bertopic import BERTopic
import warnings
import spacy
import os
from nltk.corpus import stopwords
import nltk
import streamlit as st

warnings.filterwarnings("ignore", category=DeprecationWarning)

nltk.download('stopwords')
# CONSTANT VARIABLES
TOPIC_MODEL_BY_CLUSTER__NAME = 'bioBERTopic_model'
TOPIC_MODEL_BY_CLUSTER__PATH = 'resources\\topicByClusterModel'


class TopicsByCluster:
    def __init__(self, bio_asq_path):
        self.bio_asq_path = bio_asq_path
        self.bio_docs = None

    def get_documents(self):
        documents = []
        if os.path.isdir(self.bio_asq_path):
            json_files = glob(self.bio_asq_path + "/*")
        else:
            assert False, "The path does not exist"
        for json_file in tqdm(json_files):
            with open(json_file) as file:
                asq_content = json.loads(file.read())
                documents.append(asq_content)

        self.bio_docs = []
        for doc in documents:
            for i in tqdm(range(len(doc['data'][0]['paragraphs']))):
                self.bio_docs.append(doc['data'][0]['paragraphs'][i]['context'])

        self.bio_docs = list(OrderedDict.fromkeys(self.bio_docs))

        return self.bio_docs

    def preprocess_doc(self,bio_docs):
        nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])
        stop_words = stopwords.words('english')
        text_out = []
        for text in tqdm(bio_docs):
            doc = nlp(text)
            new_text = []
            for token in doc:
                if (token.pos_ in ['NOUN', 'ADJ']) and (str(token) not in stop_words):
                    new_text.append(token.lemma_)
            final = " ".join(new_text)
            text_out.append(final)
        return text_out

    def trainBERTopicTransformerModel(self, topic_model_save_path, topic_model_save_name):

        topic_spacy_model = BERTopic(n_gram_range=(1, 3), language="english")
        print("Pre-Processing Medical Text..")
        # bio_docs = self.preprocess_doc(bio_docs)
        bio_docs = self.preprocess_doc(self.get_documents())
        print("Pre-Processing Done!")

        print("Training BERT Transformer Model to find topics...")
        topics, probs = topic_spacy_model.fit_transform(bio_docs)
        print("Training completed!")

        if not os.path.isdir(topic_model_save_path):
            os.mkdir(topic_model_save_path)
        model_save_path = topic_model_save_path + "/" + topic_model_save_name
        print("Saving the model to : {}".format(model_save_path))
        topic_spacy_model.save(model_save_path)
        print("Model saved!")

        return topic_spacy_model
