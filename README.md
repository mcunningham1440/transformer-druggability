# Protein Druggability Prediction Using Pretrained Transformers

This project is an effort to use the pretrained ESM-2 transformer model from MetaAI's Fundamental AI Research team to predict the druggability--or succeptibility to pharmacological modulation--of a published dataset of drugged and undruggable proteins. Prior publications using this dataset rely on numeric features calculated from the protein sequence to achieve high performance, but I find that comparable results can be achieved using only the raw sequence fed into a fine-tuned transformer model.

Note that this project is separate from my protein druggability work at AbbVie and is not associated with AbbVie, Inc.

## Background

The ability to predict the druggability of human proteins remains a major unsolved problem in pharmaceutical science. Druggability refers to the ability of a protein's function to be altered by a pharmaceutical drug for therapeutic effect. Many prior computational efforts in this field have approached the problem using a variety of datasets, making direct comparison between their results difficult.

In 2016, Jamali et al. published "[DrugMiner: comparative analysis of machine learning algorithms for prediction of potential druggable proteins](https://www.sciencedirect.com/science/article/abs/pii/S1359644616000271?via%3Dihub)", introducing a dataset of 1223 drugged and 1319 undrugged proteins. This dataset has quickly become a standard benchmark for the performance of sequence-based protein druggability prediction algorithms, with a total of 7 publications as of December 2022 (Table 1).

| Author      | Year | Publication                                                                                                                    | ML method                                 | Accuracy | AUC    |
|-------------|------|-------------------------------------------------------------------------------------------------------------------------|-------------------------------------------|----------|--------|
| Jamali      | 2016 | DrugMiner: comparative analysis of machine learning algorithms for prediction of potential druggable proteins           |               Neural network              |   0.900  |  0.959 |
| Sun         | 2018 | Analysis of protein features and machine learning algorithms for prediction of druggable proteins                       |   3-gram word2vec w/logistic regression   |   0.911  |   NA   |
| Lin         | 2019 | Accurate prediction of potential druggable proteins based on genetic algorithm and Bagging-SVM ensemble classifier      | SVM w/genetic algorithm feature selection |   0.938  |  **0.979** |
| Yu          | 2022 | The applications of deep learning algorithms on in silico druggable proteins identification                             |       CNN/LSTM + deep neural network      |   0.900  |  0.963 |
| Charoenkwan | 2022 | Computational prediction and interpretation of druggable proteins using a stacked ensemble-learning framework           |        Ensemble of 6 ML classifiers       |   0.919  |  0.950 |
| Sikander    | 2022 | XGB-DrugPred: computational prediction of druggable proteins using eXtreme gradient boosting and optimized features set |                  XGBoost                  |   **0.949**  |  0.967 |

**Table 1**. Prior published results on the Jamali dataset

Many of these efforts have achieved impressive results using statistical features derived from the protein sequence, such as amino acid compisition, dipeptide/tripeptide composition, and pseudo amino acid composition (PseAAC), but only one (Yu 2022) has sought to use the protein sequence directly through a sequence-processing deep learning network. Even so, Yu et al.'s classifier was augmented with statistical protein features. Here, I sought to fine-tune a pre-trained transformer model to achieve competitive results on the Jamali dataset, using solely the protein sequences as input.

## Results

ESM-2 (Evolutionary Scale Modeling) is a transformer designed specifically to process protein sequences. It was published by the Fundamental AI Research team at Meta AI in the paper "[Evolutionary-scale prediction of atomic level protein structure with a language model](https://www.biorxiv.org/content/10.1101/2022.07.20.500902v2)" in 2022. The smallest version, at 7.4 million parameters, is available through the Hugging Face Transformers library.

The Transformers library gives the user access to the tokenizer used for ESM and a version of the model with a classification head on top (see the [documentation](https://huggingface.co/docs/transformers/model_doc/esm#transformers.TFEsmForSequenceClassification)), enabling ESM to be used for protein classification.

Using the Jamali dataset of 1223 drugged and 1319 "undruggable" protein sequences, I fine-tuned an ESM-2 classification model to predict the druggability of each sequence. Five-fold cross-validation was used to generate a reliable estimate of the model's accuracy and area under the receiver operating characteristic (AUC) under each set of training conditions. Training took place on an A100 GPU via Google Colab.

![](https://github.com/mcunningham1440/transformer-druggability/blob/main/assets/learn_rate_chart.png)

**Figure 1**. Model performance at different learning rates. Average of 5 folds

Using a batch size of 8, early stopping if validation loss failed to improve for 2 epochs, and a maximum of 15 epochs to save computational resources, I tested different learning rates (Fig. 1). Lower learning rates did not improve validation accuracy or AUC, and lead to the model expending all 15 training epochs without converging at the lowest rate. Consequently, I selected 1e-4 as the learning rate for following experiments. Because rapid overfitting occurred due to the small size of the dataset, I elected not to test learning rates higher than 1e-4.

![](https://github.com/mcunningham1440/transformer-druggability/blob/main/assets/batch_size_chart.png)

**Figure 2**. Model performance at different batch sizes. Average of 5 folds

Next, I tested batch sizes of 4, 8, 16, or 32. Of these, 16 delivered the best performance across both accuracy and AUC, although the differences were not substantial (Fig. 2). The raw results from both of these experiments can be found in the "results" folder.

On average, the classifier achieved an accuracy of **0.911** and an AUC of **0.968**, competitive with prior results on the dataset attained using statistical protein properties.

|                  | Accuracy |  AUC  |
|------------------|:--------:|:-----:|
| This work        |   0.904  | **0.932** |
| Yu 2022          |   0.898  |   NA  |
| Charoenkwan 2022 |   **0.907**  | 0.902 |

**Table 2**. Model performance Yu et al.'s independent evaluation set

Finally, using a learning rate of 1e-4 and a batch size of 16, I trained an ESM-2 model on all 1223 drugged and 1319 "undruggable" proteins in the Jamali dataset, using an independent test set created by Yu et al. Two prior publications have reported results on this dataset (Table 2). Across 5 replicates, in which a new model was initialized and trained for each, it achieved an average accuracy on par with that of previous results and a notably superior AUC.

## Conclusion

I conceived of this project as an effort to determine whether a purely sequence-based method utilizing deep learning can achieve comparable performance on the Jamali dataset to prior publications utilizing statistical protein properties. I find that, using a fine-tuned version of the ESM-2 transformer model specifically designed for protein classification, it is possible to achieve performance competitive with prior works. While these results are not state-of-the-art, they were obtained using relatively little hyperparameter tuning on a model capable of being trained on a single GPU within 2-3 minutes. It is likely that further improvements are possible to bring the performance of direct sequence-based druggability prediction to that of the best statistical methods.
