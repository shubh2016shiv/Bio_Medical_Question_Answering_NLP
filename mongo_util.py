import pymongo
import streamlit as st


class SetUpMongo:
    def __init__(self, db_name, connection_string):
        self.db_name = db_name
        try:
            self.client = pymongo.MongoClient(connection_string)
            st.sidebar.success("MongoDB Cloud connection established")
        except (Exception,) as e:
            st.error("Error in connecting MongoDB Cloud")

    def get_collection(self):
        database = self.client[self.db_name]
        collection = database.factoid
        return collection

    def get_filtered_bioasq_docs(self,filter_query_string,collection=None):
        if collection is None:
            collection = self.get_collection()

        documents = []
        questions = []
        for found_doc in collection.find(filter_query_string):
            documents.append(found_doc['context'])
            questions.append(found_doc['qas'][0]['question'])

        return tuple((documents,questions))

