# **Group-A_UofT_Data-Bootcamp_Final-Project**

# **Beer Reviews Analysis**

The topic selected for our final project out of four options is the Beer Analysis. The reason for selecting this topic is predominantly the mass of data. The volume would help us to give ample data to train our machine learning model.

### **Description of source data**

There are three source files in the CSV format:
- beers.csv
- breweries.csv
- reviewes.csv

### ***Beer.csv***

The beer.csv file has 358,873 rows. The column header, data types and null values are depicted below:


![](https://github.com/jaykansara2019/Group-A_UofT_Data-Bootcamp_Final-Project/blob/cb87fd41d4f2b2ced89b18a7cc774a9b0baa775a/Images/df_beers.png)

![](https://github.com/jaykansara2019/Group-A_UofT_Data-Bootcamp_Final-Project/blob/cb87fd41d4f2b2ced89b18a7cc774a9b0baa775a/Images/df_beers%20(null%20values).png)

![](https://github.com/jaykansara2019/Group-A_UofT_Data-Bootcamp_Final-Project/blob/cb87fd41d4f2b2ced89b18a7cc774a9b0baa775a/Images/df_beers%20(data%20types).png)


### ***Breweries.csv***

The breweries.csv file has 50,347 rows. The column header, data types and null values are depicted below:

![](https://github.com/jaykansara2019/Group-A_UofT_Data-Bootcamp_Final-Project/blob/cb87fd41d4f2b2ced89b18a7cc774a9b0baa775a/Images/df_breweries.png)

![](https://github.com/jaykansara2019/Group-A_UofT_Data-Bootcamp_Final-Project/blob/cb87fd41d4f2b2ced89b18a7cc774a9b0baa775a/Images/df_breweries%20(null%20values).png)

![](https://github.com/jaykansara2019/Group-A_UofT_Data-Bootcamp_Final-Project/blob/cb87fd41d4f2b2ced89b18a7cc774a9b0baa775a/Images/df_breweries%20(data%20types).png)

### ***reviews.csv***

![](https://github.com/jaykansara2019/Group-A_UofT_Data-Bootcamp_Final-Project/blob/cb87fd41d4f2b2ced89b18a7cc774a9b0baa775a/Images/df_reviews.png)

![](https://github.com/jaykansara2019/Group-A_UofT_Data-Bootcamp_Final-Project/blob/cb87fd41d4f2b2ced89b18a7cc774a9b0baa775a/Images/df_reviews%20(null%20values).png)

![](https://github.com/jaykansara2019/Group-A_UofT_Data-Bootcamp_Final-Project/blob/cb87fd41d4f2b2ced89b18a7cc774a9b0baa775a/Images/df_reviews%20(data%20types).png)


The reviews.csv file has 9,073,128 rows. The column header, data types and null values are depicted below:


The data frame will be cleaned by dropping null values and bucketing the type of beer (e.g. Indian Pale Ale, and American Pale Ale will be categorised as 'Ale'). The country values will be filtered to only keep the United States data. With the help of either SQL or Python joins, all three data frames will be merged to create a unified data frame.

### **Questions we hope to answer with the data**

- Is there any pattern in the review type (look, smell, taste and feel) and type of beer. In other words, can we predict a type of beer based on the user reviews?
- States with the highest numbers of breweries.
- Highest overall ranked beers
- Breweries with the top-ranked beers
- heat map of the breweries.




