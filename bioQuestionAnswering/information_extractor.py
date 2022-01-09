from transformers import pipeline
import streamlit as st


class InformationExtractor:
    def __init__(self, query, context):
        self.query = query
        self.context = context

    @st.cache(allow_output_mutation=True)
    def get_qa_pipeline(self):
        qa_biobert_model = 'dmis-lab/biobert-base-cased-v1.1-squad'
        qa_pipeline = pipeline(tokenizer=qa_biobert_model, model=qa_biobert_model, task='question-answering')
        return qa_pipeline

    def search_and_predict_answers(self):
        qa_pipeline = self.get_qa_pipeline()
        possible_answers = []
        for context in self.context:
            possible_answers.append(qa_pipeline(question=self.query, context=context))

        return possible_answers
