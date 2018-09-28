<h1> Automated cyber security assessment system with 
<p>Natural Language Processing(NLP) and Convolutional Neural Network(CNN)</p></h1>

<h2>Motivation</h2>

This system might replace day-to-day low level of cognitive tasks: Review the assessment sheet.

It could help to automatically figure out Vunerability names per each category

<h2>Data</h2>

Training dataset: 33,000 number of vulnerability names and categories from trackers


<h2>Methodology</h2>

<h3>1. Data preprocessing.</h3>
<p>- It is an unstructured dataset(name), which is not organized in the model.</p> 
<p>- Natural Language Processing is the application that drives the meaning from text inputs.</p>

- Stemming
<p>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;*The processing of reducing infected(or derived) words to their word root(Have, has, having --> hav).</p>

- Tokenizing
<p>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;*Breaks up the texts into words.(ex-'The weather is nice'-->['the','weather','is','nice'])</p>
                                                                                                                                                   
- Numbering 
<p>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;*Convert their word roots into numbers in order for the model to recognize the texts.</p>

- Bag of words 
<p>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;*Create a unique list of word numbers.(ex-['the','weather','is','nice']-->[1,2,3,7])</p>

- padding 
<p>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;*Add extra cells surrounding the input to fix the size of lists.</p>
                                               

<h3>2. Model</h3>

- Convolutional Neural Network(CNN)

- batch issue
<p>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;*I set up batch_size which is equal to 100.</p> 

- Dropout
<p>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;*It helps to prevent CNN models from overfitting.(rate=.5)</p> 

                                                                                         
<h2>Accuracy Rate</h2>                                                                                        
- 91% ~ 93% * test size: 3,300 (10% of total data)
 
                                                                                        
<h2>Reference</h2>
<p>- https://github.com/hunkim/DeepLearningZeroToAll</p> 
<p>- https://github.com/golbin/TensorFlow-Tutorials</p> 
<p>- https://github.com/tflearn/tflearn/tree/master/examples</p> 
