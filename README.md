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

•	Smell

•	Look

•	Feel

•	Taste

•	State

•	Abv

•	Availability

A few issues with the dataset and ways to overcome them are listed below:

1.	There are 104 unique beer styles. This needs to be cut down by binning similar beer styles and grouping beer styles with counts below 500 to an “Other” category.

2.	The number of reviews for the various beer styles vary significantly, with some styles receiving 300 reviews and others receiving over 10,000. Some techniques that we can use to resolve this class imbalance is oversampling, undersampling, and SMOTEENN.

3.	The 3 tables need to be merged.

Below are the steps we are following:

1.	USE DATA -> Use the data in the brewery, beers, and reviews csv.

2.	CLEAN & SCRUB DATA -> Merge the datasets in Python or SQL, filter for specifically breweries in the USA, and get rid of unnecessary columns and bin similar beer styles together, drop null values.

3.	RUN DATA THROUGH ALOGORITHM -> Create 3 different classification algorithms:
    a) Logistic Regression
    b) Random Forest
    c) XG Boost


4.	a) EVALUATE -> Determine if the model has acceptable accuracy, precision, recall, and f1    score.
    b) REPEAT -> Repeat steps 2-4 if model is not strong enough.

5.	SAVE MODEL AND RESULTS -> Save the model and present to class

### Preliminary Data Preprocessing
The preliminary processing of the data involved:
- Merging the breweries dataframe with the beers dataframe, and finally merging this to the reviews dataframe
- Filter the dataframe for beers in the USA
- Dropping null values
- Bin similiar beer styles together

### Training and Testing Datasets
The train_test_split function was used to train and test the model. Initially we trained 75% of the data and tested the model with the remaining 25% (default). We also tried training the model with 67% of the data, and testing it with the remaining 33%.

### Logistic Regression
The first model we decided to implement was a logistic regression model. Since the model is straightforward, we decided to see if this simple model would be a good predictor for our data.

#### Preliminary Feature Engineering
The preliminary feature engineering involved:
- Encoding features such as 'state', and 'style'
- Multiplied highly correlated variables together such as 'taste' and 'feel', and 'taste' and 'smell'

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

#### Feature Combinations
Combinations of various features were used to increase the models accuracy as well as reducing the number of categories the model had to predict (removing "Other"). Below is a list of the various feature combinations and alterations to the beer style.

- Combination 1:
  a) state_encoded
  b) abv
  c) look
  d) smell
  e) taste
  f) feel

- Combination 2:
  a) state_encoded
  b) abv
  c) look
  d) taste x feel
  e) taste x smell

- Combination 3:
  a) state_encoded
  b) abv
  c) taste x feel

- Combination 4:
  a) state_encoded
  b) abv
  c) look
  d) taste x feel
  e) taste x smell
  f) Dropped the beer styles classified as "Other"

Each model had an accuracy of around 11-12%. Using logistic regression did not provide a strong model with any of the combinations because it was not able to capture the variation in the data well.
  
  



## Database
As the project progressed, based on the emerging needs, we made few changes to field names in all 3 tables. Removed fields/column names which has less relevant information and yet posed a challenge with storage of data. Also, dropped all rows with null values to get accurate results from our ML models. Explanation of the updates in detail below for each table and an updated ERD.

![ERD-Beer Reviews](https://github.com/jaykansara2019/Group-A_UofT_Data-Bootcamp_Final-Project/blob/99e63ef6f4295c0b1994ce370187fb1e54f61c30/Images/Database/ERD_updated.png)

**Breweries Table:**
Changed *id*field name to *brewery_id* to specify this id indicates brewery. This update would help the users of the DB to easily distinguish between other id fields and help simplify the parent-child relationship. *name*ield was changed to *brewery_name* for same reason. All other fields city, state, country, notes, types kept similar as before. *brewery_id* is the primary key in this table. Total number of rows are 50,347, same as source breweries.csv file. 

![breweries table](https://github.com/jaykansara2019/Group-A_UofT_Data-Bootcamp_Final-Project/blob/2dcc5ba8ec7ae23dc84f46984bb31af1091c3a3c/Images/Database/breweries_table.png)

**Beers Table:**
Both *id* and *name* fields were changed to *beer_id* and *beer_name* respectively to describe the field better and to describe the primary-foreign key relation easily with other tables. We also decided to remove notes field as most were null or without relevant comments. *beer_id* is the primary key in this table. *brewery_id* is the foreign key in beers table in relation to *brewery_id* in breweries table. Total 358,873 rows imported from beers.csv, same as source file.

![beers table](https://github.com/jaykansara2019/Group-A_UofT_Data-Bootcamp_Final-Project/blob/067e09be6c23273ce73eb8e2813d2cb55601df84/Images/Database/beers_table.png)

Reviews Table:
All fields in reviews table kept same as previous except for *text* field. We found this field is relatively less relevant for our analysis, hence removed this field completely. *beer_id* is a foreign key in reviews table in relation to beers table. There is no primary key in this table. A total of 9,073,128 rows were imported from the source, reviews.csv file. All rows were imported successfully.

![reviews table](https://github.com/jaykansara2019/Group-A_UofT_Data-Bootcamp_Final-Project/blob/067e09be6c23273ce73eb8e2813d2cb55601df84/Images/Database/reviews_table.png)




## Machine Learning Model: Random Forest
Random Forest machine learning model is used to train and test our database, to predict beer type based on various features.
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



