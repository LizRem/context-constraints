# Context constraints
This repository contains the code to understand context length constraints in clinical langauge models using structured EHR data. We evaluate 4 LLaMA-3 models with varying context lengths (512, 1024, 2048, 4096 tokens) and assess performance on three three tasks: retinopathy, neuropathy and nephropathy within 10 years of T2D diagnosis. 

## The challenge 
There is limited research into clinical foundation models with longer context lengths. As highlighted recently by [Wornow et al.,](https://arxiv.org/abs/2412.16178) the majority of research on clinical foundation models has been limited to models with shorter context lengths (256-512 tokens) or restricted to a single context window. In this study, we compare 4 models with varying context lengths and evaluate two training paradigms 1) pretrain + fine tune, with 2) pretrain + linear probing as well as comparing to a count-based XGBoost model. 

## Data availability 
These models are trained on data from the Clinical Practice Research Datalink (CPRD), data is available to researchers on [application](https://www.cprd.com/). Given that data is not publically available, this repository contains the code but does not release the model or tokenizer.  

## Data preparation 
Each patient is represented by a sentence of clinical codes, as this research uses CPRD, each patient is represented by strings of BNF, SNOMED, ICD10 and OPCS codes which are ordered chronolgoically. Demographic tokens (sex, ethnicity and IMD quartile) which are not associated with a specific clinical encounter are inserted at the beginning of all patient sequences, as demostrated in the figure below. Additionally a death token is inserted at the end of the patient sequence when recorded.
![Data input format](images/data-prep.png)

## Model training
The following scripts are included:
XX: This script trains a domain specific tokenizer. 

XX: This script optimises hyperparametes (learning rate, batch size and warm-up ratio) then conducts a full pretraining of the clinical language model for 7 epoches.

XX: This script conducts and evaluates a full fine-tuning of the pretrained model

XX: This script conducts and evaluates linear probing (only single classification layer is trained).

XX: This script trains and evaluates a count-based XGBoost baseline model using varying fixed length representations.

## Evaluation
The following notebooks are included:
XX: This notebook is used to analyse the SHAP values of the model to identify clinical events most strongly associated with a positive prediciton of each microvascular complication. 

XX: This notebook creates Uniform Manifold Approximation and Projection (UMAP) embeddings to help visualise embeddings within a 2-dimensional space.

XX: This notebook evaluates models over subgroup performance, stratifying across ethnicity, sex and IMD. 

XX: This notebook analyses self attention weights from across all layers and heads to assess attention distribution at different context lengths and patient sequence lengths. 
