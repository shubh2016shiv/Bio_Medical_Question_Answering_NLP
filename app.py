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
from bioQuestionAnswering.information_extractor import InformationExtractor
import torch
import joblib
from sentence_transformers import util
import shutil


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
        
        
def get_keywords_and_filter_query(_topics, _bio_topics):
    
    ## uncomment below line to train BERTopic from Scratch
#     words = [word[0] for word in _topic.get_topic(int(_bio_topics.split(" ")[1]))]
    words = [word[0] for word in _topics[(int(_bio_topics.split(" ")[1]))]]

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
navigation_options = st.sidebar.selectbox("Options", options=["Show Project Details and Architecture",
                                                          "Search Bio-Topics & Questions",
                                                          "Search Answers based on Questions"])

if navigation_options == "Show Project Details and Architecture":
    st.header("Instructions to proceed")
    st.info("💁Mark the below checkbox step by step to trigger the Engine")
    model_path = config['topic_cluster']['model_path'] + "/" + config['topic_cluster']['model_name']
    
    # Model Setup Sequence Checkbox
    topic_bert_checkbox = st.checkbox(label="Setup the Bio-Topic Cluster created using BERT Topic Model")
    disease_genetics_ner_checkbox = st.checkbox(label="Setup Disease and Genetic Entities Extracted using NER Pipeline of Hugging Face's Transformer")
    qa_encoding_sentence_transformer_checkbox = st.checkbox(label="Setup Sentence Transformer QA Encoding Model for Information Retrieval")
###   Uncomment below to train BERTopic from Scratch  ##  
#     if topic_bert_checkbox and not os.path.exists(model_path):
#         with st.spinner("Please wait. Creating BERT Topic Model.."):
#             topic_by_cluster = TopicsByCluster(bio_asq_path=config['bioASQ_path']['path'])
#             topic_model = topic_by_cluster.trainBERTopicTransformerModel(
#                 topic_model_save_path=config['topic_cluster']['model_path'],
#                 topic_model_save_name=config['topic_cluster']['model_name'])

#             cluster_viz = topic_model.visualize_topics()
#             cluster_viz.write_html(config['topic_cluster']['model_path'] + "/" + config['topic_cluster']['cluster_viz_name'])
#             del topic_model, cluster_viz
#             gc.collect()

## Comment Below 'if' block to traing BERTopic from scratch 
    if topic_bert_checkbox and not os.path.exists(config['topic_cluster']['model_path']):
        with st.spinner("Please wait. Downloading Topic Clusters from Pre-Trained BERT Topic Model.."):
            if not os.path.isdir(config['topic_cluster']['model_path']):
                os.mkdir(config['topic_cluster']['model_path'])
                gdown.download_file_from_google_drive(config['topic_cluster']['cluster_topics_share_id'],
                                                  config['topic_cluster']['model_path']
                                                  + "/" + config['topic_cluster']['cluster_topics_name'])
                gdown.download_file_from_google_drive(config['topic_cluster']['cluster_viz_share_id'],
                                                  config['topic_cluster']['model_path']
                                                  + "/" + config['topic_cluster']['cluster_viz_name']) 
            
        st.info("💁Now, Topic Clusters can be Explored from Navigation > Search Bio-Topics & Questions > Bio Clusters")
            
    elif disease_genetics_ner_checkbox and not (os.path.exists(config['NER']['disease_genetics_NER_path'])):
        with st.spinner("Please wait. Downloading Extracted Diseases and Genes related Entities.."):
             if not os.path.isdir(config['NER']['disease_genetics_NER_path']):
                 os.mkdir(config['NER']['disease_genetics_NER_path'])
                 gdown.download_file_from_google_drive(config['NER']['disease_NER_share_id'],
                                                          config['NER']['disease_genetics_NER_path']
                                                          + "/" + 'DiseasesNER.txt')
                 gdown.download_file_from_google_drive(config['NER']['genetics_NER_share_id'],
                                                          config['NER']['disease_genetics_NER_path']
                                                          + "/" + 'geneticsNER.txt')
        st.info("💁Now, Topics based on Disease and Genetic Entities can be Explored from Navigation > Search Bio-Topics & Questions > Diseases / Genetics")
                
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
        st.info("💁Now, Question and Answering on Bio-Medical data can be done from Navigation > Search Answers based on Questions")    
                
    elif (os.path.exists(model_path)) and (os.path.exists(config['NER']['disease_genetics_NER_path'])) and (config['qa_encoded_corpus']['path']):
        st.success("Pre-Trained Models and Pipelines are ready. Engine is now hot.\
        \n >> Topics based on Clusters, Diseases or Genetics and Questions \
                related to them can be searched or explored from Navigation option: 'Search Bio-Topics and Questions' in Slidebar.\
                \n>> After searching topic and its related question, copy the question and navigate to Navigation option: 'Search Answers based on Questions' in Slidebar to predict the answers to Bio-Medical Question")

    st.header("Downloads - Data and Jupyter Notebooks")
    download_options = st.selectbox("Options", options=["None",
                                                        "Bio-Medical Question Answers (BioASQ) Data",
                                                        "Jupyter Notebook: Topic Modelling and Disease/Genetic Entities Extraction",
                                                        "Jupyter Notebook: Question Answering using Information Retrieval and Extraction"])
    if download_options == "Bio-Medical Question Answers (BioASQ) Data":
        shutil.make_archive('BioASQ_data', 'zip', config['bioASQ_path']['path'])
        download_file_name = './BioASQ_data.zip'
    if download_options == "Jupyter Notebook: Topic Modelling and Disease/Genetic Entities Extraction":
        download_file_name = './BioASQ_Topic_Modelling_and_NER.ipynb'
    if download_options == "Jupyter Notebook: Question Answering using Information Retrieval and Extraction":
        download_file_name = './BioASQ_Question_Answering_Project.ipynb'
    if download_options != "None":
        with open(download_file_name, "rb") as file:
            st.download_button(label="Download", data=file,file_name=download_file_name.replace("./",""))
        
elif navigation_options == "Search Bio-Topics & Questions":
    # uncomment below line to train BERTopic from Scratch
#     topic_model = get_cached_topic_cluster_model()
    topic_selection = st.sidebar.selectbox(
        "Select the method of determining Topics",
        ["Bio Clusters", "Diseases", "Genetics"]
    )

    if topic_selection == 'Bio Clusters':
        HtmlFile = open(config['topic_cluster']['model_path'] + "/" + config['topic_cluster']['cluster_viz_name']
                        , 'r', encoding='utf-8')
        source_code = HtmlFile.read()

        components.html(source_code, height=650)
        
        # comment below line to train BERTopic from Scratch
        topics = joblib.load(config['topic_cluster']['model_path'] + "/" + config['topic_cluster']['cluster_topics_name'])
        
        # uncomment below line to train BERTopic from Scratch
#         bio_topics = st.sidebar.selectbox("Select the Bio Cluster Topic Number",
#                                           ["Topic {}".format(i) for i in range(0, len(topic_model.topics) - 1)])

        # comment below line to train BERTopic from Scratch
        bio_topics = st.sidebar.selectbox("Select the Bio Cluster Topic Number",
                    ["Topic {}".format(i) for i in range(0, len(topics) - 1)])
        
        # uncomment below line to train BERTopic from Scratch
#         filter_query = get_keywords_and_filter_query(topic_model, bio_topics)
        
        # comment below line to train BERTopic from Scratch
        filter_query = get_keywords_and_filter_query(topics, bio_topics)
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
       
        cos_scores = util.pytorch_cos_sim(query_embedding, encoded_corpus)[0]
        top_results = torch.topk(cos_scores, k=10)
        st.write(top_results)
        relevant_context = []
        for score, idx in zip(top_results[0], top_results[1]):
            relevant_context.append(bio_docs[idx])
            # print(bio_docs[idx], "(Score: {:.4f})".format(score))

        document_expander = st.expander("Click here to show Top 10 Documents relevant to the question based on cosine similarity")
        with document_expander:
            st.write(relevant_context)

        with st.spinner("💁Extracting 10 possible answers using Information Extractor QA pipeline"):
            extractor = InformationExtractor(query, relevant_context)
            answers = {}
            st.success("✓ Below are predicted possible answers.")
            answers_expander = st.expander("Click here to show Predicted answers")
            with answers_expander:
                predicted_10_answers = extractor.search_and_predict_answers()
                for prediction in predicted_10_answers:
                    answers[prediction['answer']] = prediction['score']

                st.write(answers)
                
            fig = go.Figure(go.Bar(
                    y=list(answers.keys()),
                    x=list(answers.values()),
                    orientation='h'))

            st.plotly_chart(fig)
