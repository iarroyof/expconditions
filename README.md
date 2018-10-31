# Introduction
Experimental conditions are pieces of information that express details on how specific experiments were performed in most of the biomedical scientific papers. The importance of such an information is crucial for the verification of the results reported in the papers. Despite the importance of experimental conditions, they sometimes are not curated in biological databases. 

Sometimes research cannot be reproduced correctly due to the lack of details about the conditions leading the results reported. 
In this work we propose a method for experimental conditions identification we called the Sliding Classifier. 
This method bases on the ideas of Harris on cooccurrence and structure of language. In the context of experimental conditions 
mentioned in a text, these ideas state that the different instances of a given experimental condition have similar contexts 
independently of the specificity of a sentence mentioning it. This mainly due to the patterns in the use of language structure.

# Sliding Classifier
The Sliding Classifier is a learning machine trained for extraction of experimental conditions from scientific literature in 
the biomedical domain. This method thus takes text samples in a form of paragraphs. A pre-processing step breaks each input 
sample into sliding windows. The center of each sliding window is a target word, which is considered as a label for classification.
Thus, all sliding windows become the new samples (sliding samples) excluding the target word (the label). The aim is to train a 
classifier for discriminating the sliding samples according to their labels, which are either experimental condition tags or any 
other word. See [here](https://github.com/iarroyof/expconditions/blob/master/esquema-gcs.xsd) the complete list of tags.

# Results
The results showed that the proposed approach for experimental conditions extraction performed well. As there are not approaches 
to this problem, the performance of our approach was assessed by using evaluation measures such as Precision and Recall. These 
measures were taken by each class, which provides a multiview standpoint of how well actually the classes were detected by the 
classifier in test data while avoiding class imbalace biases. See this [table](https://github.com/iarroyof/expconditions/blob/master/results.csv) summarizing the results for each experimental 
condition (the [labels](https://github.com/iarroyof/expconditions/blob/master/esquema-gcs.xsd)).
