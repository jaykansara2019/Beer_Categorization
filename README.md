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





