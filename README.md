# Group-A_UofT_Data-Bootcamp_Final-Project

  # <ins>**Beer Reviews Analysis**</ins>

  The topic we have selected for our final project is creating a machine learning algorithm that will determine the beer stlye based on select inputs. The reason for selecting this topic is because we are able to work with 3 mass datasets as well as many members in our group being self-proclaimed beer connoisseur.

  ### <ins>**Description of Source Data**</ins>

  There are three source files in the CSV format that we obtained from Kaggle:
  - [beers.csv](https://www.kaggle.com/datasets/ehallmar/beers-breweries-and-beer-reviews)
  - [breweries.csv](https://www.kaggle.com/datasets/ehallmar/beers-breweries-and-beer-reviews)
  - [reviewes.csv](https://www.kaggle.com/datasets/ehallmar/beers-breweries-and-beer-reviews)

  #### <ins>***Beer.csv***</ins>

  The beer.csv file has 358,873 rows. The column header, data types and null values are depicted below:


  ![](https://github.com/jaykansara2019/Group-A_UofT_Data-Bootcamp_Final-Project/blob/cb87fd41d4f2b2ced89b18a7cc774a9b0baa775a/Images/df_beers.png)

  ![](https://github.com/jaykansara2019/Group-A_UofT_Data-Bootcamp_Final-Project/blob/cb87fd41d4f2b2ced89b18a7cc774a9b0baa775a/Images/df_beers%20(null%20values).png)

  ![](https://github.com/jaykansara2019/Group-A_UofT_Data-Bootcamp_Final-Project/blob/cb87fd41d4f2b2ced89b18a7cc774a9b0baa775a/Images/df_beers%20(data%20types).png)


  #### <ins>***Breweries.csv***</ins>

  The breweries.csv file has 50,347 rows. The column header, data types and null values are depicted below:

  ![](https://github.com/jaykansara2019/Group-A_UofT_Data-Bootcamp_Final-Project/blob/cb87fd41d4f2b2ced89b18a7cc774a9b0baa775a/Images/df_breweries.png)

  ![](https://github.com/jaykansara2019/Group-A_UofT_Data-Bootcamp_Final-Project/blob/cb87fd41d4f2b2ced89b18a7cc774a9b0baa775a/Images/df_breweries%20(null%20values).png)

  ![](https://github.com/jaykansara2019/Group-A_UofT_Data-Bootcamp_Final-Project/blob/cb87fd41d4f2b2ced89b18a7cc774a9b0baa775a/Images/df_breweries%20(data%20types).png)

  #### <ins>***reviews.csv***</ins>

  The reviews.csv file has 9,073,128 rows. The column header, data types and null values are depicted below:

  ![](https://github.com/jaykansara2019/Group-A_UofT_Data-Bootcamp_Final-Project/blob/cb87fd41d4f2b2ced89b18a7cc774a9b0baa775a/Images/df_reviews.png)

  ![](https://github.com/jaykansara2019/Group-A_UofT_Data-Bootcamp_Final-Project/blob/cb87fd41d4f2b2ced89b18a7cc774a9b0baa775a/Images/df_reviews%20(null%20values).png)

  ![](https://github.com/jaykansara2019/Group-A_UofT_Data-Bootcamp_Final-Project/blob/cb87fd41d4f2b2ced89b18a7cc774a9b0baa775a/Images/df_reviews%20(data%20types).png)

  ### <ins>**Questions we hope to answer with the data**</ins>

  - Are the input variables strong indicators of the beer style?
  - Do beers brewed in specific states have higher reviews?
  - Which States have the highest numbers of breweries?
  - Which beer style is has the highest reviews?


## Machine Learning Model

SciKitLearn is the ML library we'll be using to create a classifier. Our final project will involve creating a supervised machine learning model that uses classification to predict the discrete beer style of a beer with the following inputs below:

•	Aroma

•	Appearance

•	Texture

•	Taste

•	State

•	Bewery


A few issues with the dataset and ways to overcome them are listed below:

1.	There are 104 unique beer styles. This needs to be cut down by binning similar beer styles and grouping beer styles with counts below 500 to an “Other” category.

2.	The number of reviews for the various beer styles vary significantly, with some styles receiving 300 reviews and others receiving over 10,000. Some techniques that we can use to resolve this class imbalance is oversampling, undersampling, and SMOTEENN.

3.	The 3 tables need to be merged.

Below are the steps we are following:

1.	USE DATA -> Use the data in the brewery, beers, and reviews csv.

2.	CLEAN & SCRUB DATA -> Merge the datasets in Python or SQL, filter for specifically breweries in the USA, and get rid of unnecessary columns and bin similar beer styles together, drop null values.

3.	RUN DATA THROUGH ALOGORITHM -> Create 3 different classification algorithms for each class using class imbalance technique (oversampling, undersampling, and SMOTEENN).

4.	a) EVALUATE -> Determine if the model has acceptable accuracy, precision, recall, and f1    score.
    b) REPEAT -> Repeat steps 2-4 if model is not strong enough.

5.	SAVE MODEL AND RESULTS -> Save the model and present to class

## Database
An ERD of our database is shown below:

![ERD-Beer Reviews](https://github.com/jaykansara2019/Group-A_UofT_Data-Bootcamp_Final-Project/blob/336a78fef4e5ad9639944e100fdeeb0b88ab5b28/Images/Database/ERD-Beer%20Reviews.png)

## Machine Learning Model: Random Forest
Random Forest machine learning model is used to train and test our database, to predict beer type based on various features.
### Preliminary data preprocessing
- Distribution of nemerical columns 
  Overall distrubtion and outliers are checked for nemerical columns including beer abv, look, smell, taste, feel and overall and score for reviews. Most of the values are left skewed except abv. Outliers are checked but due to the extreme small portion of the outliers compared to the large sample size, we didn't exclude any outliers.
- Stratification
  There are 40 output classes for target column (Beer style) with over 4 million rows with some classes larger than other (Pale Ale: 1 million vs Ohters: 11k). To prevent undersampling of groups, we take a sample size of 11100 from each output values instead of stratifying the target column.
- Encoding labels
  Two categorical variables (Availability and State of the beer) are transformed into nemerical data by LabelEncoder.
### Preliminary feature selection
Nine features are selected for the training input data: state beer, availablility, abv, look, smell, taste, feel, overall and score. Name, id, country and city of the beer and brewery are removed. Style of the beer is the target set.
### Spliting training and testing sets
Stratified and encoded dataset is split into training and testing sets by 67-33 split.
### Results
#### Fit the model
- Training and Testing features data are scaled. We create the randome forest instance and fit the model with our training sets and make predictions using testing dsets. n_estimator is set to 128. 
#### Evaluate the model
- Confusion matrix, classification report with accuracy score, precision, recall and f1 scores are shown to evaluate the model.
- Precision, recall and f1-score for each class of beer are relatively similar around 50% to 70%. Considering precision and recall are similar in value, there is not much trade off between precision and recall, we will use accuracy to compare how well each model works. 
- Predictive accuracy is 60%. The top three relative important features are abv, state of the beer, score. 
#### Optimize the model
Model is optimized by, adjusting input data and increasing n_estimator. 
- By correlation analysis, taste seems to be highly correlated with smell or look, we tried removing taste, or combine taste with look and smell. Removal of the columns reduced accuracy to 30%. 
- We trie removing "Other" from output in case the variability within the bucketed class will cause confusion in the model. However, the accurcy stayed the same.
- Increasing n_estimator from 128 to 500 doesn't significantly increase accuracy and also cause extremely slow processing speed and freezing of computer due to large amound of data processed. We decide to keep n_estimator at 128.
The highest predictive accuracy from Random Forest model is 60%.
### Model Choice
Random forest classifiers are ensemble learning model that combines multiple smaller models into a more robust and accurate model. Random forest models work well with large tabular dataset with robusteness and scalability and both output and feature selection are easy to interpret. Addictionaly, they handle outliers and nonlinear data. Considering the substential amount of tabular data we have, we choose random forest model for our predictions.
#### Benefits
- Are robust against overfitting
- Can handle thousands of input variable without variable deletion
- Are robust to outliers and nonlinear data
- Run efficiently on large dataset with less code and faster performance
- Can be used to rank the importance of input variable in a natural way
#### Limitations
- Large number of trees can make algorithm too slow for training and predictions (require more computational power and resources)
- May miss variability within dataset



