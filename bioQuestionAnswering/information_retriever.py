import json
from collections import OrderedDict
from glob import glob
from tqdm import tqdm
import os
from sentence_transformers import SentenceTransformer, util
import time
import torch


class InformationRetriever:
    def __init__(self, bio_asq_path):
        self.bio_asq_path = bio_asq_path
        self.bio_docs = None
        self.sentence_encoding_model = None
        self.encoded_corpus = None
        self.encoded_query = None

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


    def encode_docs_using_qa_transformer(self, encoded_corpus_save_path: str, encoded_corpus_save_name: str):
        self.bio_docs = self.get_documents()
        self.sentence_encoding_model = SentenceTransformer('multi-qa-mpnet-base-cos-v1')
        start_time = time.time()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.encoded_corpus = self.sentence_encoding_model.encode(self.bio_docs, convert_to_tensor=True, \
                                                                  batch_size=32, device=device)
        print("Corpus Encoding completed!")
        print("---  %f minutes ---" % ((time.time() - start_time) / 60))

        if not os.path.isdir(encoded_corpus_save_path):
            os.mkdir(encoded_corpus_save_path)
        encoded_corpus_save_full_path = encoded_corpus_save_path + "/" + encoded_corpus_save_name
        print("Saving Encoded Corpus on path: " + encoded_corpus_save_full_path)
        torch.save(self.encoded_corpus, encoded_corpus_save_full_path)

        return self.encoded_corpus

    def retrieve_docs_based_on_query(self, query, encoded_corpus):
        if self.encoded_corpus is None:
            self.encoded_corpus = encoded_corpus
        if self.bio_docs is None:
            self.bio_docs = self.get_documents()

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.sentence_encoding_model = SentenceTransformer('multi-qa-mpnet-base-cos-v1')
        self.encoded_query = self.sentence_encoding_model.encode(query, convert_to_tensor=True, device=device)

        cos_scores = util.pytorch_cos_sim(self.encoded_query, self.encoded_corpus)[0]
        top_results = torch.topk(cos_scores, k=10)
        print(top_results)
        relevant_docs = []
        for score, idx in zip(top_results[0], top_results[1]):
            relevant_docs.append(self.bio_docs[idx])

        return relevant_docs
