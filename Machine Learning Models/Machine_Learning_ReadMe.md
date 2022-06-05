# Machine Learning model seletion, Optimisation and Tuning

## Model selection

We are trying to predict the style of beer (e.g. Pale ale, Pilsner) based on the beer reviewes, the state it was brewed in and the availability. During our analysis we tried 3 different models to test our hypotheses.

SciKitLearn is the ML library we'll be using to create a classifier. Our final project will involve creating a supervised machine learning model that uses classification to predict the discrete beer style of a beer with the following inputs below:

- Smell

- Look

- Feel

- Taste

- State

- Abv

- distribution (sold in breweries, bars, eatery, stores or beer-to-go stores)

- availability (year-round, rotating)



### Training and Testing Datasets
The train_test_split function was used to train and test the model. Initially we trained 75% of the data and tested the model with the remaining 25% (default). We also tried training the model with 67% of the data, and testing it with the remaining 33%.

## Logistic Regression
The first model we decided to implement was a logistic regression model. Since the model is straightforward, we decided to see if this simple model would be a good predictor for our data.

#### Preliminary Feature Engineering
The preliminary feature engineering involved:
- Encoding features such as 'state', and 'style'
- Multiplied highly correlated variables together such as 'taste' and 'feel', and 'taste' and 'smell'

The model had an accuracy of around 11-12%. Using logistic regression did not provide a strong model with any of the combinations because it was not able to capture the variation in the data well.

#### Benefits
Below are the benefits of the model:
- Logistic Regression is simple to understand
- It requires less training relative to other classifier models
- It performs well for simple datasets as well as when the data set is linearly separable
- It doesn’t make any assumptions about the distributions of classes in feature space
- A Logistic Regression model is less likely to be over-fitted
- They are easier to implement, interpret, and very efficient to train

#### Limitations
Below are the limitations of the model:
- Sometimes a lot of Feature Engineering is required
- If the independent features are correlated with each other it may affect the performance of the classifier
- It is quite sensitive to noise and overfitting
- Logistic Regression should not be used if the number of observations is lesser than the number of features
- By using Logistic Regression, non-linear problems can’t be solved because it has a linear decision surface
- By using Logistic Regression, it is tough to obtain complex relationships

## Random Forest
Random Forest machine learning model is used to train and test our database, to predict beer type based on various features. Random forest classifiers are ensemble learning model that combines multiple smaller models into a more robust and accurate model. Random forest models work well with large tabular dataset with robusteness and scalability and both output and feature selection are easy to interpret. Addictionaly, they handle outliers and nonlinear data. Considering the substential amount of tabular data we have, we choose random forest model for our predictions.

### Preliminary data preprocessing
- Distribution of nemerical columns 
  Overall distrubtion and outliers are checked for nemerical columns including beer abv, look, smell, taste, feel and overall and score for reviews. Most of the values are left skewed except abv. Outliers are checked but due to the extreme small portion of the outliers compared to the large sample size, we didn't exclude any outliers.
- Stratification
  There are 40 output classes for target column (Beer style) with over 4 million rows with some classes larger than other (Pale Ale: 1 million vs Ohters: 11k). To prevent undersampling of groups, we take a sample size of 11100 from each output values instead of stratifying the target column.
- Encoding labels
  Two categorical variables (Availability and State of the beer) are transformed into nemerical data by `LabelEncoder`.

### Preliminary feature selection
Nine features are selected for the training input data: state beer, availablility, abv, look, smell, taste, feel, overall and score. Name, id, country and city of the beer and brewery are removed. Style of the beer is the target set.

### Spliting training and testing sets
Stratified and encoded dataset is split into training and testing sets by 67-33 split.

### Results
#### Fit the model
- Training and Testing features data are scaled. We create the randome forest instance and fit the model with our training sets and make predictions using testing dsets. `n_estimator` is set to 128. 

#### Evaluate the model
- Confusion matrix, classification report with accuracy score, precision, recall and f1 scores are shown to evaluate the model.
- Precision, recall and f1-score for each class of beer are relatively similar around 50% to 70%. Considering precision and recall are similar in value, there is not much trade off between precision and recall, we will use accuracy to compare how well each model works. 
- Predictive accuracy is 60%. The top three relative important features are abv, state of the beer, score.
 
#### Optimize the model
Model is optimized by, adjusting input data and increasing n_estimator. 
- By correlation analysis, taste seems to be highly correlated with smell or look, we tried removing taste, or combine taste with look and smell. Removal of the columns reduced accuracy to 30%. 
- We trie removing "Other" from output in case the variability within the bucketed class will cause confusion in the model. However, the accurcy stayed the same.
- Increasing `n_estimator` from 128 to 500 doesn't significantly increase accuracy and also cause extremely slow processing speed and freezing of computer due to large amound of data processed. We decide to keep `n_estimator` at 128.
The highest predictive accuracy from Random Forest model is 60%.

#### Benefits
- Are robust against overfitting
- Can handle thousands of input variable without variable deletion
- Are robust to outliers and nonlinear data
- Run efficiently on large dataset with less code and faster performance
- Can be used to rank the importance of input variable in a natural way

#### Limitations
- Large number of trees can make algorithm too slow for training and predictions (require more computational power and resources)
- May miss variability within dataset

## XGBoost model
###Background
XGBoost is a decision-tree-based ensemble Machine Learning algorithm that uses a gradient boosting framework. The image below depicts the evolution of XGBoost model.

![Evolution of XGBoost model](https://github.com/jaykansara2019/Group-A_UofT_Data-Bootcamp_Final-Project/blob/d8a259decc511850780c74e89b27bee67608a0c8/Images/Data/Evolution%20of%20XGBoost%20model.jpeg)

The mechanism utilised for optimisation of the Gradient boosting model is illustrated below:

![](https://github.com/jaykansara2019/Group-A_UofT_Data-Bootcamp_Final-Project/blob/d8a259decc511850780c74e89b27bee67608a0c8/Images/Data/XGBoost%20mechanism.png)
[Reference](https://towardsdatascience.com/https-medium-com-vishalmorde-xgboost-algorithm-long-she-may-rein-edd9f99be63d#:~:text=What%20is%20XGBoost%3F,all%20other%20algorithms%20or%20frameworks.)

### Preliminary feature engineering and selection:
After the initial exploration and data transformation process, the string values in the 'Stratified_df' file were encoded using the LabelEncoder. Features selected in the fifth optimisation attempt are alcohol content (abv), availability, state, taste, smell, and overall score. The feature that we are trying to predict is a type of beer (e.g. Lager or Pale Ale). The data were scaled using the StandardSclaer to ensure that the outliers will not affect the model. The number of outliers was less than 5% in the data fed into the Machine Learning model. 

### Model Train-Test split and Preliminary result
The training sample was used at a default (75%) volume and 67% volume. The accuracy score obtained with 67% test data volume is 65% whereas the accuracy score with the default test volume (75%) is 64%.

### Benefits:
- Highly Flexible
- Uses the power of parallel processing
- Faster than Gradient Boosting
- Supports regularization
- Designed to handle missing data with its in-build features.
- Users can run cross-validation after each iteration.
- Works well in small to medium dataset10

### Limitations:
XGBoost does not perform quite well on sparse and unstructured data. The Gradient Boosting in general is very sensitive to outliers since every classifier is forced to fix the errors in the predecessor learners.

### Model Selection
Based on the accuracy of all three models, we have decided to go with XGBoost. Further optimisation attempts will be performed by modifying the feature selection process and data stratification.


## Optimisation and tuning attempts

The first 6 optimisation attempts were focused on feature selection based on the importance and correlation. To avoid bias, we removed the 'Overall' score feature due to its high correlation with the feature 'taste'. The final features (X) are:
- taste
- distribution (sold in breweries, bars, eatery, stores or beer-to-go stores)
- availability (year-round, rotating)
- abv (alcohol content)
- state_beer

The optimisation 7, we compared the training and testing accuracy and Optimisation 8 was performed to tune the number of trees (n_estimator) using GridSerachCV. The result of n_estimator v/s loss ratio is illustrated below:

![](https://github.com/jaykansara2019/Group-A_UofT_Data-Bootcamp_Final-Project/blob/3036a8dd5bb8834d67e87794f2ec0047e73a7d71/Images/Machine%20Learning/n_estimators.png)

In Optimisation 9 and 10, using Optuna we tried with the larger model to learn the best range of hyperparameters (e.g. alpha, lambda etc).

Based on Optuna tuning attempts we determined the best values for each hyperparameter, the importance and the number of trials required to achieve the optimal accuracy.

![](https://github.com/jaykansara2019/Group-A_UofT_Data-Bootcamp_Final-Project/blob/6a5ee33d9f9cc3f572c0f721456a6e6b5c489eaa/Images/Machine%20Learning/Hypeparameter%20importance.png)

![](https://github.com/jaykansara2019/Group-A_UofT_Data-Bootcamp_Final-Project/blob/6a5ee33d9f9cc3f572c0f721456a6e6b5c489eaa/Images/Machine%20Learning/Hyperparameter%20objective%20values.png)

![](https://github.com/jaykansara2019/Group-A_UofT_Data-Bootcamp_Final-Project/blob/6a5ee33d9f9cc3f572c0f721456a6e6b5c489eaa/Images/Machine%20Learning/Optimisation%20history.png)

In the final attempt, we are trying to use the shorter version of the model by trying to incorporate the hyperparameter outputs that we obtained from Optimisation 8 and 10. The testing accuracy achieved during the final attempt is 76.30%. The feature importance for the final model is depicted below:

![](https://github.com/jaykansara2019/Group-A_UofT_Data-Bootcamp_Final-Project/blob/a324552b5e0adbb44013da7d289fd79516fcb85a/Images/Machine%20Learning/Feature%20importance%20ranking%20(1).png)




