import streamlit as st
from bertopic import BERTopic
import st_initlialize
from bioTopics.topics_by_cluster import TopicsByCluster
import gc
import os
import streamlit.components.v1 as components
from streamlit_tags import st_tags

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
        
        

st.sidebar.header("Navigation")
navigation_options = st.sidebar.radio("Options", options=["Show Project Architecture and details",
                                                          "Search Bio-Topics & Questions",
                                                          "Search Answers based on Questions"])

if navigation_options == "Show Project Architecture and details":
    st.info("💁Press the below button to trigger the Engine")
    model_path = config['topic_cluster']['model_path'] + "/" + config['topic_cluster']['model_name']
    if st.button(label="Initialize models") and not os.path.exists(model_path):
        with st.spinner("Please wait. Creating BERT Topic Model.."):
            topic_by_cluster = TopicsByCluster(bio_asq_path=config['bioASQ_path']['path'])
            topic_model = topic_by_cluster.trainBERTopicTransformerModel(
                topic_model_save_path=config['topic_cluster']['model_path'],
                topic_model_save_name=config['topic_cluster']['model_name'])

            cluster_viz = topic_model.visualize_topics()
            cluster_viz.write_html(config['topic_cluster']['model_path'] + "/" + config['topic_cluster']['cluster_viz_name'])
            del topic_model, cluster_viz
            gc.collect()
    elif os.path.exists(model_path):
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
        
        get_keywords_and_filter_query(topic_model, bio_topics)
