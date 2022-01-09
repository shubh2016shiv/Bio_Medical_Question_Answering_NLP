import streamlit as st
from bertopic import BERTopic
import st_initlialize
from bioTopics.topics_by_cluster import TopicsByCluster
import gc
import os

st.set_page_config(layout="wide")
st.title("Project - Topic Modelling and Question Answering on Bio-Medical Text")

mongo_connection, config = st_initlialize.initialize_streamlit_and_mongodb()
collection = mongo_connection.get_collection()

st.sidebar.header("Navigation")
navigation_options = st.sidebar.radio("Options", options=["Show Project Architecture and details",
                                                          "Search Bio-Topics & Questions",
                                                          "Search Answers based on Questions"])

if navigation_options == "Show Project Architecture and details":
    model_path = config['topic_cluster']['model_path'] + "/" + config['topic_cluster']['model_name']
    if st.button(label="Initialize models") and not os.path.exists(model_path):
        with st.spinner("Please wait. Creating BERT Topic Model.."):
            topic_by_cluster = TopicsByCluster(bio_asq_path='./BioASQ_data')
            topic_model = topic_by_cluster.trainBERTopicTransformerModel(
                topic_model_save_path=config['topic_cluster']['model_path'],
                topic_model_save_name=config['topic_cluster']['model_name'])

            cluster_viz = topic_model.visualize_topics()
            cluster_viz.write_html(config['topic_cluster']['model_path'] + "/" + config['topic_cluster']['cluster_viz_name'])
            del topic_model, cluster_viz
            gc.collect()
    elif os.path.exists(model_path):
        st.info("Model already exists")
elif navigation_options == "Search Bio-Topics & Questions":
    with st.spinner("Loading BERT Topic Model..."):
        model_path = config['topic_cluster']['model_path'] + "/" + config['topic_cluster']['model_name']
        if os.path.isdir(config['topic_cluster']['model_path']) and os.path.exists(model_path):
            topic_model = BERTopic.load(model_path)
        else:
            st.info("BERT Topic model is not available")
