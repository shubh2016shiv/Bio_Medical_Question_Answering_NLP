
# Project Title

Bio-Medical and Topic Modelling Question Answering using NLP Transformers

## Dataset Description
BioASQ is a large-scale biomedical semantic indexing and question answering (Bio-QA) dataset that contains a collection of factoid questions and their answers related to biomedical literature. It is a benchmark for evaluating the performance of biomedical QA systems, and it is used in the annual BioASQ challenge. The dataset is in JSON format and has the following structure:

    title: The title of the dataset.
  
    paragraphs: An array of objects, each object representing a paragraph of biomedical literature and its corresponding question-answer pairs.
    |
    -> context: A string of text representing the paragraph of biomedical literature.
    | 
    -> qas: An array of question-answer objects, each containing the following fields:
        |
        -> question: A string of text representing a factoid question.
        |
        -> id: A unique identifier for the question-answer pair.

*Example*
```notepad
{
	"version": "BioASQ6b",
	"data": [{
		"title": "BioASQ6b",
		"paragraphs": [{
			"context": "The antibody aducanumab reduces A\u03b2 plaques in Alzheimer's disease. Alzheimer's disease (AD) is characterized by deposition of amyloid-\u03b2 (A\u03b2) plaques and neurofibrillary tangles in the brain, accompanied by synaptic dysfunction and neurodegeneration. Antibody-based immunotherapy against A\u03b2 to trigger its clearance or mitigate its neurotoxicity has so far been unsuccessful. Here we report the generation of aducanumab, a human monoclonal antibody that selectively targets aggregated A\u03b2. In a transgenic mouse model of AD, aducanumab is shown to enter the brain, bind parenchymal A\u03b2, and reduce soluble and insoluble A\u03b2 in a dose-dependent manner. In patients with prodromal or mild AD, one year of monthly intravenous infusions of aducanumab reduces brain A\u03b2 in a dose- and time-dependent manner. This is accompanied by a slowing of clinical decline measured by Clinical Dementia Rating-Sum of Boxes and Mini Mental State Examination scores. The main safety and tolerability findings are amyloid-related imaging abnormalities. These results justify further development of aducanumab for the treatment of AD. Should the slowing of clinical decline be confirmed in ongoing phase 3 clinical trials, it would provide compelling support for the amyloid hypothesis.",
			"qas": [{
				"question": "What disease is the drug aducanumab targeting?",
				"id": "58a95c711978bbde22000001_000"
			}]
		}]
	}]
}
```
      
The dataset contains a large collection of factoid questions and their answers, related to biomedical topics. It is a useful benchmark to evaluate the performance of biomedical QA systems. The dataset is provided in JSON format and can be used to train and evaluate machine learning models for answering factoid questions in the biomedical field.

## 🛠 Skills
Pytorch, MongoDB, Python, Streamlit 1.4.0, Spacy 3.2.0, HuggingFace


## 🔗 Links
**Working Application Demonstration**

[![Open Application in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://shubh2016shiv-bio-medical-qa-new-app-tguwht.streamlit.app/)

**My LinkedIn Profile**

[![linkedin](https://img.shields.io/badge/linkedin-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/shubham-singh-060525108/)

## Bio-Medical Topics and Their Clusters

BERTopic is used for training and creating cluster on Bio-Medical Corpus after extracting out only NOUN and ADJECTIVES using Spacy NLP library. The Cluster are as follows: 

![](https://github.com/shubh2016shiv/Bio_Medical_QA_New/blob/main/Image%20Resources/Bio-Medical%20Topic%20Cluster.gif)

## Extracting Disease Entities using NER HuggingFace Pipeline
[Code Section for getting Diseases Entities using HuggingFace](https://colab.research.google.com/drive/1BOOWj70x5YgNaCwKNqdBoHb5VpfFYnJb?authuser=1#scrollTo=mkryooi2JzPo)

Disease Entity recognition and extraction using transformer called [Bio-Former](https://huggingface.co/bioformers/bioformer-cased-v1.0-ncbi-disease) with HuggingFace NER pipeline
A transformer called [Bio-Former](https://huggingface.co/bioformers/bioformer-cased-v1.0-ncbi-disease) is used with NER HuggingFace Pipeline to extract out the Diseases
![](https://github.com/shubh2016shiv/Bio_Medical_QA_New/blob/main/Image%20Resources/Disease%20NER%20using%20HuggingFace.png)

## Extracting Genetic Entities using NER HuggingFace Pipeline
[Code Section for getting Genetic Entities using HuggingFace](https://colab.research.google.com/drive/1BOOWj70x5YgNaCwKNqdBoHb5VpfFYnJb?authuser=1#scrollTo=5Us7Y12YJ3TF)

Genetic Entity recognition and extraction is done using BioBERT Disease NER ([biobert_genetic_ner](https://huggingface.co/alvaroalon2/biobert_genetic_ner))
![](https://github.com/shubh2016shiv/Bio_Medical_QA_New/blob/main/Image%20Resources/Genetic%20NER%20using%20HuggingFace.png)

## Question-Answering using Information Retrieval and Extraction
[The Colab Notebook for Question-Answering on Bio-Medical Corpus](https://colab.research.google.com/drive/13rTpvjzE6qgArvrbc2wCri70U0wki7I4?usp=sharing)

Select 'Search Answers based on Question' in Navigation and Give the Bio-Medical related question in text area.
![](https://github.com/shubh2016shiv/Bio_Medical_QA_New/blob/main/Image%20Resources/Asking%20Bio-Medical%20Question.png)

Press ENTER key to trigger the information retriever to retrive top 10 most matching documents related to the question in decreasing order of cosine similarity between question and documents 
![](https://github.com/shubh2016shiv/Bio_Medical_QA_New/blob/main/Image%20Resources/Information%20Retrieval.png)

After Retrieving the best matching documents, the information extraction is automatically initiated to search for the answer for question in the retrived documents.
For this purpose, Question-Answering pipeline using HuggingFace transformer - [BioBERT Transformer](https://huggingface.co/dmis-lab/biobert-large-cased-v1.1) is used. 
![](https://github.com/shubh2016shiv/Bio_Medical_QA_New/blob/main/Image%20Resources/Information%20Extraction.png)
