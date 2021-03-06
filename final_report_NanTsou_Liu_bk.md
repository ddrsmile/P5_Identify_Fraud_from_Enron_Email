# Porject 5: Identify Fraud from Enron Email
---
**Udacity Data Analyst Nonedegree**

By Nan-Tsou Liu @ 2016-07-11

## Introduction
<p>
<a href=https://en.wikipedia.org/wiki/Enron>Enron Corporation</a> was an American energy, commodities, and services company based in Houston, Texas. Before its bankruptcy in 2001, there were about 20,000 employees and was one of the world's major electricity, natural gas, communications and pulp and paper companies.
</p>
<p>
<a href=https://en.wikipedia.org/wiki/Enron_Corpus>The Enron Corpus</a> is a large database of over 600,000 emails generated by 158 employees of the Enron Corporation and acquired by the Federal Energy Regulatory Commission during its investigation after the company's collapse. 
</p>

## Short Question

>Summarize for us the goal of this project and how machine learning is useful in trying to accomplish it. As part of your answer, give some background on the dataset and how it can be used to answer the project question. Were there any outliers in the data when you got it, and how did you handle those? <br/>
[relevant rubric items: “data exploration”, “outlier investigation”]


###Goal of the Project:

<p>
In this project, the prediction model is built by using the python module called scikit-learn to identify the person of interest (POI). The dataset contains 146 records with 1 labele (POI), 14 financial features and 6 email feature. In the data of labels, there are 18 records have been marked as POI. The following skills are applied to carried out the project:
<ul>
<li>feature selection and scaling</li>
<li>algorithm selection and tuning</li>
<li>validation and classic mistakes</li>
<li>evaluation metrics and interpretation of algorithm's performance</li>
</ul>
</p>

###Outliers
<p>
By observing the data and pdf file, enron61702insiderpay.pdf, the obvious outliers are <strong>TOTAL</strong> and <strong>THE TRAVEL AGENCY IN THE PARK</strong> which are simply removed by <code>pop()</code> method of dictionary in Python. Besides, the record of <strong>LOCKHART EUGENE E</strong> is empty, thus it was also removed.
</p>
<p>
I found out that the records of <strong>BELFER ROBERT</strong> and <strong>BHATNAGAR SANJAY</strong> are not consistent with the data in pdf file by accident. Therefore, I fixed them before removing outliers and replacing NaN with 0. I did not check every records so that I am not sure whether there are others not consistent or not.
</p>

---

>What features did you end up using in your POI identifier, and what selection process did you use to pick them? Did you have to do any scaling? Why or why not? As part of the assignment, you should attempt to engineer your own feature that does not come ready-made in the dataset -- explain what feature you tried to make, and the rationale behind it. (You do not necessarily have to use it in the final analysis, only engineer and test it.) In your feature selection step, if you used an algorithm like a decision tree, please also give the feature importances of the features that you use, and if you used an automated feature selection function like SelectKBest, please report the feature scores and reasons for your choice of parameter values.

<p>
At the beginning, I used all the original features except <strong>mail_address</strong> in the dataset. Besides, I added <strong>financial relatived features</strong> like the ratios of each payment feature or stock feature to the total amount of financial and <strong>message related feature</strong>.
</p> 
<p>
The reasons I added these features are that first, I assumed that POI has somehow great relationship with the financial status. I calculated the ratio of each financial features to <strong>total_financial</strong> (summation of <strong>total_payments</strong> and <strong>total_stock_value</strong>) because I thought that the person of POI might had large percentage of restricted_stock or salary of the total financial status. It also means that <strong>the composition of the financial status</strong> might be the good features for model training.
</p>
<p>
On the other hand, the messages of each person should be a strong feature to identify POI. Therefore, I culaculated the ratio of <strong>poi_relative_message</strong> (summation of <strong>from_poi_to_this_person</strong>, <strong>from_this_person_to_poi</strong> and shared <strong>receipt_with_poi</strong>) to <strong>total_messages</strong> (summation of <strong>from_message</strong> and <strong>to_message</strong>).
</p>

###Added Features

<table>
<thead>
<tr>
<td><strong>Feature</strong></td>
<td><strong>Description</strong></td>
</tr>
</thead>
<tbody>
<tr>
<td><strong>total_financial</strong></td>
<td>total_payments + total_stock_value</td>
</tr>
<tr>
<td><strong>{each feature}_ratio</strong></td>
<td>{each feature} / total_financial</td>
</tr>
<tr>
<td><strong>poi_ratio_messages</strong></td>
<td>poi_related_messages / total_messages</td>
</tr>
<tr>
<td><strong>poi_related_messages</strong></td>
<td>from_poi_to_this_person + from_this_person_to_poi + shared_receipt_with_poi (NOT USED)</td>
</tr>
<tr>
<td><strong>total_messages</strong></td>
<td>from_messages + to_messages (NOT USED)</td>
</tr>
</tbody>
</table>

### Engineering Data

<p>
The end-up using features were selected during GridSearchCV pipeline search with following steps:
<ol>
<li>scale all features to be between 0 and 1 with MinMaxScaler</li>
<li>dimension reduction with SelectKBest and Principal Components Analysis</li>
<li>tune parameters of each models</li>
</ol>
Features scaling was carried out since PCA and various models such as Logistic Regression perform optimally with scaled features. Feature scaling is also necessary since they were on very different scales, ranging from hundreds of e-mails to millions of dollars.
</p>
<p>
<strong>SelectKBest</strong> and <strong>Principal Components Analysis (PCA)</strong> dimension reduction were run during each of the cross-validation loops during the grid search. The K-best features were selected using <strong>Anova F-value classification</strong> scoring function. The K-best features were then used in reducing dimension with PCA. Finally,the N principal components were fed into a classification algorithm.
</p>

---

>What algorithm did you end up using? What other one(s) did you try? How did model performance differ between algorithms?

<p>
The ended up using algorithm was determined by the results of <strong>GridSearchCV</strong> respected <strong>recall score</strong> and <strong>precision score</strong> with k-best feature selection, PCA reduction and parameters tuning. <strong>Support Vector Classifier (SVC)</strong> showed the best results, and therefore it was the ended up using algorithm in this project. Besides SVC, I also tried <strong>Logistic Regression (LogReg)</strong>, <strong>Linear Support Vector Machine (LSVC)</strong>, <strong>Decision Tree (DTree)</strong> and <strong>K-Means Classifier (KMeans)</strong>.
</p>
<p>
Actually, LogReg and LSVC also showed the competive results compared with that of SVC. And the results of both algorithms were similar. The recall scores are about <strong>0.72</strong> and <strong>0.68</strong> of LogReg and LSVC respectively. And the precision scores are about <strong>0.3</strong> of both algorithms. According to the obversation by manual parameter tuning, recall score and precision score were changed against with each other. And the parameter <strong>C</strong> affected the results mostly. And I found out an interesting phenomenon that the value ended with 5 like <code>[0.05, 0.5, 0.15]</code> could keep recall score at good value and promote precision score well. And, precision score kelp <strong>around 0.3</strong>.
</p>

<p>
The results of DTree were fair although the results matched the requirements of the assignment. 
</p>

<p>
On the other hand, the precision score of KMeans kept at the value <strong>about 0.15</strong>, which is far from the requirement of the assignment. No matter how I tuned the parameters, although precision score was about <strong>0.5</strong>. In my opinion, KMeans might not suitable for this prediction after I added the new features.
</p>

---

>What does it mean to tune the parameters of an algorithm, and what can happen if you don’t do this well?  How did you tune the parameters of your particular algorithm? (Some algorithms do not have parameters that you need to tune -- if this is the case for the one you picked, identify and briefly explain how you would have done it for the model that was not your final choice or a different model that does utilize parameter tuning, e.g. a decision tree classifier). 

<p>
Tuning the parameters of an algorithm is a process to promote the performance of the model. Depended on the structure and nature of the dataset, tuning the parameters would cost lots of time when doing model training. In this project, <strong>Grid Search</strong> was used to tune the parameters of the algorithms with 1000 randomized stratified cross-validation stratified splits. The parameters of each algorithms with highest average score were choosen for the models.
</p>

### Final Results of each Algorithm
<table>
<thead>
<tr>
<td><strong>Parameter</strong></td>
<td><strong>Logistic Regression</strong></td>
<td><strong>Linear Support Vector Classifier</strong></td>
<td><strong>Support Vector Classifier</strong></td>
<td><strong>Decision Tree</strong></td>
</tr>
</thead>
<tbody>
<tr>
<td><strong>C</strong></td>
<td>0.5</td>
<td>0.15</td>
<td>0.01</td>
<td>-</td>
</tr>
<tr>
<td><strong>class_wight</strong></td>
<td>auto</td>
<td>auto</td>
<td>auto</td>
<td>balanced</td>
</tr>
<tr>
<td><strong>tol</strong></td>
<td>1e-64</td>
<td>1e-32</td>
<td>1e-8</td>
<td>-</td>
</tr>
<tr>
<td><strong>n_components of PCA</strong></td>
<td>0.5</td>
<td>0.5</td>
<td>0.5</td>
<td>0.5</td>
</tr>
<tr>
<td><strong>whiten of PCA</strong></td>
<td>True</td>
<td>False</td>
<td>False</td>
<td>False</td>
</tr>
<tr>
<td><strong>selection of SelectKBest</strong></td>
<td>17</td>
<td>15</td>
<td>15</td>
<td>10</td>
</tr>
<tr>
<td><strong>gamma</strong></td>
<td>-</td>
<td>-</td>
<td>0.0</td>
<td>-</td>
</tr>
<tr>
<td><strong>kernel</strong></td>
<td>-</td>
<td>linier</td>
<td>rbf</td>
<td>-</td>
</tr>
<tr>
<td><strong>criterion</strong></td>
<td>-</td>
<td>-</td>
<td>-</td>
<td>entropy</td>
</tr>
<tr>
<td><strong>splitter</strong></td>
<td>-</td>
<td>-</td>
<td>-</td>
<td>best</td>
</tr>
<tr>
<td><strong>max_depth</strong></td>
<td>-</td>
<td>-</td>
<td>-</td>
<td>2</td>
</tr>
<tr>
<td><strong>min_sample_leaf</strong></td>
<td>-</td>
<td>-</td>
<td>-</td>
<td>20</td>
</tr>
<tr>
<td><strong>min_sample_split</strong></td>
<td>-</td>
<td>-</td>
<td>-</td>
<td>2</td>
</tr>
</tbody>
</table>

---

>What is validation, and what’s a classic mistake you can make if you do it wrong? How did you validate your analysis?

<p>
Validation is the process to ensure that the results produced built model with other unknown data is reliable. In my case, A classic mistake is over-fitting, which makes the model too particular so that the results produced with the other data are poor and unreliable. One of major purpose of validation is to avoid over-fitting.
</p>

### Over-Fitting on SVC with Parameter C

<p>
I found out an interesting results when I manually tuned parameters of SVC. As the table shown below, I obtained recall score which is <strong>1.0000</strong> when I set 0.05 to parameter <strong>C</strong>. At the beginning, I did not noticed that it was the result caused by over-fitting. But I noticed that it was caused by parameter C. So, I did the simple investigate on the internet. The simple description of C is that the value of the regularization constraint, which tells the SVM optimization <strong>how much you want to avoid misclassifying</strong> each training dataset. And a very small value of C will cause the optimizer to look for a larger-margin to separate hyperplane, which means that there might be many points which are hyperplane misclassifies.</p>
<table>
<thead>
<tr>
<td><strong>CASE</strong></td>
<td><strong>C</strong></td>
<td><strong>recall score</strong></td>
<td><strong>precision score</strong></td>
</tr>
</thead>
<tbody>
<tr>
<td>1</td>
<td>0.01</td>
<td>0.7335</td>
<td>0.3317</td>
</tr>
<td>2</td>
<td>0.05</td>
<td>1.0000</td>
<td>0.1333</td>
</tr>
</tbody>
</table>


### Cross-Validation
<p>
Cross-Validation was applied on the validation of model. It is a process that randomly split the data into training and testing dataset. And then it trains the model with the training data and validates with the testing data.
In this project, the whole dateset was splitted with 1000 randomized <strong>stratified cross-validation splits</strong>. And then the parameters with the best performance over 1000 splits were selected.
</p>

### Parameter Tuning
<p>
As I mentioned above, <strong>GridSearchCV</strong> with over 1000 stratified shuffled cross-validation 90%-training/ 10%-testing splits was used to tune the parameters in this project. Besides, K-best selection and PCA reduction processes were embraced into the parameter tuning loop. Compared with outside selection, the selection in the loop might promote the consistence of parameter tuning and give a less biased estimate of performance on any new unseen data that this model might be used for.
</p>
<p>
As the results of selected the final model, 15 features were selected. PCA reduction gave 2 principal components. These parameters were used in the final <strong>Support Vector Machine</strong> classification model.
</p>
<p>
One should be notified is that these features might change slightly each time since the k-best selection was carried out inside of the pipeline. Below are the final 15 features chosen when the entire dataset was fit to the final chosen model pipeline:
</p>

###K-Best Feature (Top 15)
<table>
<thead>
<tr>
<td><strong>feature</strong></td>
<td><strong>score↑</strong></td>
</tr>
</thead>
<tbody>
<tr>
<td><strong>total_stock_value</strong></td>
<td>22.5105490902</td>
</tr>
<tr>
<td><strong>exercised_stock_options</strong></td>
<td>22.3489754073</td>
</tr>
<tr>
<td><strong>bonus</strong></td>
<td>20.7922520472</td>
</tr>
<tr>
<td><strong>salary</strong></td>
<td>18.2896840434</td>
</tr>
<tr>
<td><strong>deferred_income</strong></td>
<td>11.4248914854</td>
</tr>
<tr>
<td><strong>poi_ratio_messages</strong></td>
<td>10.0194150056</td>
</tr>
<tr>
<td><strong>long_term_incentive</strong></td>
<td>9.92218601319</td>
</tr>
<tr>
<td><strong>total_payments</strong></td>
<td>9.28387361843</td>
</tr>
<tr>
<td><strong>restricted_stock</strong></td>
<td>8.83185274222</td>
</tr>
<tr>
<td><strong>shared_receipt_with_poi</strong></td>
<td>8.58942073168</td>
</tr>
<tr>
<td><strong>loan_advances</strong></td>
<td>7.18405565829</td>
</tr>
<tr>
<td><strong>bonus_ratio</strong></td>
<td>6.58326879249</td>
</tr>
<tr>
<td><strong>expenses</strong></td>
<td>5.41890018941</td>
</tr>
<tr>
<td><strong>from_poi_to_this_person</strong></td>
<td>5.24344971337</td>
</tr>
<tr>
<td><strong>loan_advances_ratio</strong></td>
<td>4.56000450411</td>
</tr>
</tbody>
</table>

---

>Give at least 2 evaluation metrics and your average performance for each of them.  Explain an interpretation of your metrics that says something human-understandable about your algorithm’s performance.

### Final Evaluation Matric Results of each Algorithm

<p>
<strong>Recall</strong> and <strong>Precision</strong> were used as the primary evaluattion metrics. The definition are shown below:
</p>
<p>
<code>recall = True_Positive / (True_Positive + False_Negative)</code>
</p>
<p>
According to the difinition of recall, it is reliable that the target is not what we are interested if it is marked negative with high recall score. Because high recall score also means low false negative.
</p>
<p>
<code>precision = True_Positive / (True_Positive + False_Positive)</code>
</p>
<p>
On the other hand, the target marked positive with high precision score can be thought it is what we are interested with high confidence. Because high precision could be thought as that false positive is low.
</p>

<p>
In this porject, lots of records in the dataset are unknown and we have much more negative labels than positive ones. Thus, in my opinion, it is a little bit not practical to build a model which can identify the target is POI, which means the model with high precision score. On the other hand, it should be more practical to build the model which can identify the target is not POI, which means the model with high recall score. As the result, it is not surprised that recall socres are <strong>much greater</strong> than precision scores of all the results of all the algorithms in this project.
</p>

<table>
<thead>
<tr>
<td><strong>Model</strong></td>
<td><strong>Recall Score by GridSearch</strong></td>
<td><strong>Recall</strong></td>
<td><strong>Precision</strong></td>
<td><strong>KBest Features</strong></td>
<td><strong>PCA Components</strong></td>
</tr>
</thead>
<tbody>
<tr>
<td><strong>Logistic Regression</strong></td>
<td>0.718</td>
<td>0.732</td>
<td>0.295</td>
<td>17</td>
<td>2</td>
</tr>
<tr>
<td><strong>Linear SVC</strong></td>
<td>0.685</td>
<td>0.683</td>
<td>0.292</td>
<td>15</td>
<td>2</td>
</tr>
<tr>
<td><strong>SVC</strong></td>
<td>0.716</td>
<td>0.7335</td>
<td>0.3317</td>
<td>15</td>
<td>2</td>
</tr>
<tr>
<td><strong>Decision Tree</strong></td>
<td>0.611</td>
<td>0.593</td>
<td>0.389</td>
<td>10</td>
<td>2</td>
</tbody>
</table>

---

## Conclusion
<p>
During this project, I understand what machine learning more. In order to finished this project. I did many study on the skills and knowledge on the internet, especially for the usage of scitkit-learn module. Besides, I also noticed that the structure and nature of the feature dataset do affect the result deeply. To work on create useful feature dataset with original dataset should be a big knowledge and needed much experience. Of course, with this course and the project, I did learn lots of skills and knowledge about not only machine learning but also python. I was glad that I have learn how to do basic machine learning with python and then where and how to find the resource to train myself.
</P>

## Script

<table>
<thead>
<tr>
<td><strong>File</strong></td>
<td><strong>Description</strong></td>
</tr>
</thead>
<tbody>
<tr>
<td>poi_id.py</td>
<td>main script to train POI classification model</td>
</tr>
<tr>
<td>poi_data.py</td>
<td>fix non-consistent data and reform data structure</td>
</tr>
<tr>
<td>poi_add_feature.py</td>
<td>add new feature to dataset with original dataset</td>
</tr>
<tr>
<td>poi_pipeline.py</td>
<td>build pipeline for GridSearch of each algorithm</td>
</tr>
<tr>
<td>poi_validate.py</td>
<td>validate the model whoes parameters are tuned by GridSearch</td>
</tr>
<tr>
<td>tester.py</td>
<td>a tools provided by Udacity to test trained model</td>
</tr>
<tr>
<td>tools/feature_format.py</td>
<td>a tools provided by Udacity to reform dataset for machine learning</td>
</tr>
</tbody>
</table>

## Reference

<table>
<tbody>
<tr>
<td>Udacity</td>
<td>https://www.udacity.com/</td>
</tr>
<tr>
<td>Enron Corporation</td>
<td>https://en.wikipedia.org/wiki/Enron</td>
</tr>
<tr>
<td>The Enron Corpus</td>
<td>https://en.wikipedia.org/wiki/Enron_Corpus</td>
</tr>
<tr>
<td>Support Vector Classifier</td>
<td>http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html</td>
</tr>
<tr>
<td>Linear Support Vector Classifier</td>
<td>http://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html</td>
</tr>
<tr>
<td>Logistic Regression</td>
<td>http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html</td>
</tr>
<tr>
<td>Decision Tree</td>
<td>http://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html</td>
</tr>
</tr>
<tr>
<td>KMeans</td>
<td>http://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html</td>
</tr>
<tr>
<td>GridSearchCV</td>
<td>http://scikit-learn.org/stable/modules/generated/sklearn.grid_search.GridSearchCV.html</td>
</tr>
<tr>
<td>Pipeline</td>
<td>http://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html</td>
</tr>
<tr>
<td>StratifiedShuffleSplit</td>
<td>http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html</td>
</tr>
<tr>
<td>StratifiedShuffleSplit</td>
<td>http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html</td>
</tr>
<tr>
<td>LogReg vs LSVC 1</td>
<td>https://www.quora.com/What-is-the-difference-between-Linear-SVMs-and-Logistic-Regression</td>
</tr>
<tr>
<td>LogReg vs LSVC 2</td>
<td>http://stats.stackexchange.com/questions/95340/comparing-svm-and-logistic-regression</td>
</tr>
<tr>
<td>What is the influence of C in SVMs with linear kernel?</td>
<td>http://stats.stackexchange.com/questions/31066/what-is-the-influence-of-c-in-svms-with-linear-kernel</td>
</tr>
<tr>
<td>Introduction to Machine Learning with Python and Scikit-Learn</td>
<td>http://kukuruku.co/hub/python/introduction-to-machine-learning-with-python-andscikit-learn</td>
</tr>
<tr>
<td>Using Pipeline and GridSearchCV for More Compact and Comprehensive Code</td>
<td>https://civisanalytics.com/blog/data-science/2016/01/06/workflows-python-using-pipeline-gridsearchcv-for-compact-code/</td>
</tr>
</tbody>
</table>