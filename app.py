import streamlit as st
from bertopic import BERTopic
import st_initlialize
from bioTopics.topics_by_cluster import TopicsByCluster
import gc
import os
import streamlit.components.v1 as components
from streamlit_tags import st_tags
from collections import OrderedDict
from bioTopics.topics_by_diseases_or_genetics import TopicsByDiseasesOrGenetics
import gdown
from bioQuestionAnswering.information_retriever import InformationRetriever
import torch
import joblib
from sentence_transformers import util


st.set_page_config(layout="wide")
st.title("Project - Topic Modelling and Question Answering on Bio-Medical Text")

mongo_connection, config = st_initlialize.initialize_streamlit_and_mongodb()
collection = mongo_connection.get_collection()


# Functions
@st.experimental_singleton(suppress_st_warning=True)
def get_cached_topic_cluster_model():
    with st.spinner("Loading BERT Topic Model..."):
        model_path_ = config['topic_cluster']['model_path'] + "/" + config['topic_cluster']['model_name']
        if os.path.isdir(config['topic_cluster']['model_path']) and os.path.exists(model_path_):
            topic_model_ = BERTopic.load(model_path_)
            return topic_model_
        else:
            st.info("BERT Topic model is not available")
            return None

@st.experimental_singleton(suppress_st_warning=True)
def get_cached_disease_genetic_entities(entity_type):
    topics = TopicsByDiseasesOrGenetics(ner_path=config['NER']['disease_genetics_NER_path'])
    if entity_type == 'disease':
        return topics.getDiseases()
    elif entity_type == 'genetics':
        return topics.getGenetics()        
        
@st.experimental_singleton(suppress_st_warning=True)
def get_cached_qa_encoding_model():
    model = joblib.load(config['qa_encoded_corpus']['path']
                        + "/" +
                        config['qa_encoded_corpus']['model_name'])

    encoded_corpus = joblib.load(config['qa_encoded_corpus']['path']
                                + "/" +
                                config['qa_encoded_corpus']['encoded_corpus_name'])

    return model,encoded_corpus      

@st.experimental_singleton(suppress_st_warning=True)
def get_bio_docs():
    docs = joblib.load(config['qa_encoded_corpus']['path']
                       + "/" +
                       config['qa_encoded_corpus']['bio_docs_name'])
    return docs
        
        
def get_keywords_and_filter_query(_topic_model, _bio_topics):
    words = [word[0] for word in _topic_model.get_topic(int(_bio_topics.split(" ")[1]))]

    keywords = st_tags(
        label='# Words:',
        text='Press enter to add more',
        value=words,
        maxtags=15)

    st.write('<style>div.row-widget.stRadio > div{flex-direction:row;justify-content: center;} </style>',
             unsafe_allow_html=True)

    st.write('<style>div.st-bf{flex-direction:column;} div.st-ag{font-weight:bold;padding-left:2px;}</style>',
             unsafe_allow_html=True)

    and_or_filter = st.radio("Filter Documents by 'OR' or 'AND' keywords in MongoDB", ("OR", "AND"))

    doc_filter_query = []
    for topic in keywords:
        doc_filter_query.append({"context": {'$regex': topic}})

    if and_or_filter == 'OR':
        doc_filter_query = {'$or': doc_filter_query}
    elif and_or_filter == 'AND':
        doc_filter_query = {'$and': doc_filter_query}

    filter_query_expander = st.expander("Expand to show Filter Query")
    with filter_query_expander:
        st.write(doc_filter_query)
        
    return doc_filter_query
        
 
def get_docs_and_ques(query):
    with st.spinner("Filtering Documents in Mongo Cloud Database and fetching results"):
        _documents, _questions = mongo_connection.get_filtered_bioasq_docs(query)
    doc_expander = st.expander("Expand it to get filtered documents based on query from Mongo Database")
    query_expander = st.expander("Expand it to get filtered question based on query from Mongo Database", expanded=True)
    _documents, _questions = list(OrderedDict.fromkeys(_documents)),list(OrderedDict.fromkeys(_questions))
    with doc_expander:
        st.write(_documents)
    with query_expander:
        st.write(_questions)


st.sidebar.header("Navigation")
navigation_options = st.sidebar.radio("Options", options=["Show Project Architecture and details",
                                                          "Search Bio-Topics & Questions",
                                                          "Search Answers based on Questions"])

if navigation_options == "Show Project Architecture and details":
    st.info("💁Mark the below checkbox step by step to trigger the Engine")
    model_path = config['topic_cluster']['model_path'] + "/" + config['topic_cluster']['model_name']
    
    # Model Setup Sequence Checkbox
    
    topic_bert_checkbox = st.checkbox(label="Initialize models")
    disease_genetics_ner_checkbox = st.checkbox(label="Download Extracted Disease and Genetic Entities")
    qa_encoding_sentence_transformer_checkbox = st.checkbox(label="Initialize Sentence Transformer QA Encoding Model")
    
    if topic_bert_checkbox and not os.path.exists(model_path):
        with st.spinner("Please wait. Creating BERT Topic Model.."):
            topic_by_cluster = TopicsByCluster(bio_asq_path=config['bioASQ_path']['path'])
            topic_model = topic_by_cluster.trainBERTopicTransformerModel(
                topic_model_save_path=config['topic_cluster']['model_path'],
                topic_model_save_name=config['topic_cluster']['model_name'])

            cluster_viz = topic_model.visualize_topics()
            cluster_viz.write_html(config['topic_cluster']['model_path'] + "/" + config['topic_cluster']['cluster_viz_name'])
            del topic_model, cluster_viz
            gc.collect()
            
    elif disease_genetics_ner_checkbox and not (os.path.exists(config['NER']['disease_genetics_NER_path'])):
        with st.spinner("Please wait. Downloading Extracted Diseases and Genes related NER.."):
             if not os.path.isdir(config['NER']['disease_genetics_NER_path']):
                 os.mkdir(config['NER']['disease_genetics_NER_path'])
                 gdown.download_file_from_google_drive(config['NER']['disease_NER_share_id'],
                                                          config['NER']['disease_genetics_NER_path']
                                                          + "/" + 'DiseasesNER.txt')
                 gdown.download_file_from_google_drive(config['NER']['genetics_NER_share_id'],
                                                          config['NER']['disease_genetics_NER_path']
                                                          + "/" + 'geneticsNER.txt')
                
    elif qa_encoding_sentence_transformer_checkbox and not (os.path.exists(config['qa_encoded_corpus']['path'])):
        with st.spinner("Please Wait. Setting up Encoding Sentence Transformer model for QA.. "):
            if not os.path.isdir(config['qa_encoded_corpus']['path']):
                os.mkdir(config['qa_encoded_corpus']['path'])
                gdown.download_file_from_google_drive(config['qa_encoded_corpus']['model_share_id'],
                                                      config['qa_encoded_corpus']['path']
                                                      + "/" +
                                                      config['qa_encoded_corpus']['model_name'])

                gdown.download_file_from_google_drive(config['qa_encoded_corpus']['encoded_corpus_share_id'],
                                                      config['qa_encoded_corpus']['path']
                                                      + "/" +
                                                      config['qa_encoded_corpus']['encoded_corpus_name'])
                
                gdown.download_file_from_google_drive(config['qa_encoded_corpus']['bio_docs_share_id'],
                                                      config['qa_encoded_corpus']['path']
                                                      + "/" +
                                                      config['qa_encoded_corpus']['bio_docs_name'])
                
    elif (os.path.exists(model_path)) and (os.path.exists(config['NER']['disease_genetics_NER_path'])) and (config['qa_encoded_corpus']['path']):
        st.success("Models are ready and Engine is now hot!!")
        
elif navigation_options == "Search Bio-Topics & Questions":
    topic_model = get_cached_topic_cluster_model()
    topic_selection = st.sidebar.selectbox(
        "Select the method of determining Topics",
        ["Bio Clusters", "Diseases", "Genetics"]
    )

    if topic_selection == 'Bio Clusters':
        HtmlFile = open(config['topic_cluster']['model_path'] + "/" + config['topic_cluster']['cluster_viz_name']
                        , 'r', encoding='utf-8')
        source_code = HtmlFile.read()

        components.html(source_code, height=650)
        bio_topics = st.sidebar.selectbox("Select the Bio Cluster Topic Number",
                                          ["Topic {}".format(i) for i in range(0, len(topic_model.topics) - 1)])
        
        filter_query = get_keywords_and_filter_query(topic_model, bio_topics)
        get_docs_and_ques(filter_query)
        
    elif topic_selection == 'Diseases':
        disease_entities = get_cached_disease_genetic_entities(entity_type='disease')
        disease_option = st.selectbox("Diseases / Health Issues / Syndromes", sorted(disease_entities))
        filter_query = {"context": {'$regex': disease_option}}
        get_docs_and_ques(filter_query)

    elif topic_selection == 'Genetics':
        genetics_entities = get_cached_disease_genetic_entities(entity_type='genetics')
        genetics_option = st.selectbox("Genes / Proteins / Antibodies", sorted(genetics_entities))
        filter_query = {"context": {'$regex': genetics_option}}
        get_docs_and_ques(filter_query)
        
elif navigation_options == "Search Answers based on Questions":        
    with st.spinner("Please Wait. Setting up Information Retriever.. "):
        model,encoded_corpus = get_cached_qa_encoding_model()
        print(encoded_corpus)
        bio_docs = get_bio_docs()
    query = st.text_input(label="Type the question here")
    if query != "":
        query_embedding = model.encode(query, convert_to_tensor=True)
        st.write(query_embedding)
        cos_scores = util.pytorch_cos_sim(query_embedding, encoded_corpus)[0]
        top_results = torch.topk(cos_scores, k=10)
        st.write(top_results)
        relevant_context = []
        for score, idx in zip(top_results[0], top_results[1]):
            relevant_context.append(bio_docs[idx])
            # print(bio_docs[idx], "(Score: {:.4f})".format(score))

    st.write(relevant_context)
