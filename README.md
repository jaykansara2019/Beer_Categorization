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
An ERD of our database is shown below:

![ERD-Beer Reviews](https://github.com/jaykansara2019/Group-A_UofT_Data-Bootcamp_Final-Project/blob/336a78fef4e5ad9639944e100fdeeb0b88ab5b28/Images/Database/ERD-Beer%20Reviews.png)




