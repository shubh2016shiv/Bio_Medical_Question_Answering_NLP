
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

BERTopic is used for training and creating cluster on Bio-Medical Corpus after retaining only NOUN and ADJECTIVES using Spacy NLP library. The Cluster are as follows: 
![](https://github.com/shubh2016shiv/Bio_Medical_QA_New/blob/main/Image%20Resources/Bio-Medical%20Topic%20Cluster.gif)

## Extracting Disease Entities using NER HuggingFace Pipeline
[Code Section for getting Diseases Entities using HuggingFace](https://colab.research.google.com/drive/1BOOWj70x5YgNaCwKNqdBoHb5VpfFYnJb?authuser=1#scrollTo=mkryooi2JzPo)

A transformer called [Bio-Former](https://huggingface.co/bioformers/bioformer-cased-v1.0-ncbi-disease) is used with NER HuggingFace Pipeline to extract out the Diseases
![](https://github.com/shubh2016shiv/Bio_Medical_QA_New/blob/main/Image%20Resources/Disease%20NER.png)
