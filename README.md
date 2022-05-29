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

## XGBoost model
XGBoost is a decision-tree-based ensemble Machine Learning algorithm that uses a gradient boosting framework. The image below depicts the evolution of XGBoost model.

![Evolution of XGBoost model](https://github.com/jaykansara2019/Group-A_UofT_Data-Bootcamp_Final-Project/blob/d8a259decc511850780c74e89b27bee67608a0c8/Images/Data/Evolution%20of%20XGBoost%20model.jpeg)

## Benefits of XGBoost model:
- Highly Flexible
- Uses the power of parallel processing
- Faster than Gradient Boosting
- Supports regularization
- Designed to handle missing data with its in-build features.
- Users can run cross-validation after each iteration.
- Works well in small to medium dataset

The mechanism utilised for optimisation of the Gradient boosting model is illustrated below:

![](https://github.com/jaykansara2019/Group-A_UofT_Data-Bootcamp_Final-Project/blob/d8a259decc511850780c74e89b27bee67608a0c8/Images/Data/XGBoost%20mechanism.png)
[Reference](https://towardsdatascience.com/https-medium-com-vishalmorde-xgboost-algorithm-long-she-may-rein-edd9f99be63d#:~:text=What%20is%20XGBoost%3F,all%20other%20algorithms%20or%20frameworks.)

## Data preparation:
After the initial exploration and data transformation process, the string values in the 'Stratified_df' file were encoded using the LabelEncoder. Features selected in the fifth optimisation attempt are alcohol content (abv), availability, state, taste, smell, and overall score). The feature that we are trying to predict is a type of beer (e.g. Lager or Pale Ale). The data were scaled using the StandardSclaer to ensure that the outliers will not affect the model. The number of outliers was less than 5% in the data fed into the Machine Learning model. 

## Model Train-Test split and Preliminary result
The training sample was used at a default (75%) volume and 67% volume. The accuracy score obtained with 67% test data volume is 65% whereas the accuracy score with the default test volume (75%) is 64%.
Further optimisation attempts will be performed by modifying the feature selection process and data stratification.

## Limitation of XGBoost model:
XGBoost does not perform quite well on sparse and unstructured data. The Gradient Boosting in general is very sensitive to outliers since every classifier is forced to fix the errors in the predecessor learners.

