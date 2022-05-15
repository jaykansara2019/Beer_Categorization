# Group-A_UofT_Data-Bootcamp_Final-Project

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

