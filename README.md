# Multiclass-Product-Classification-ML

## Introduction
Every e-commerce website has a catalog consisting of thousands of products that belong to hundreds of browse nodes (browse node or product group represent a collection of product of similar category). These nodes helps a customer to navigate through the website and classify products to product type groups.

I created this ML model as part of my Amazon ML challenge at HackerEarth from 30 July 2021 - 1 August 2021.

As a part of this model we will use product metadata to classify products into browse node. Product metadata consists of product title, description, bullet points and brand.

The objective of this project is to apply classification learning model on Amazon products dataset with 15000 products, and therefore to obtain a predictive model for identifying future products categories.

## Dataset Description
- Input Features - TITLE, DESCRIPTION, BULLET_POINTS, BRAND
- Target Column - BROWSE_NODE_ID
- Overall Dataset size used for this model - 15,000 (available at - https://drive.google.com/file/d/17yk2SOn8bbwtSCEl45lDexrLX5XMDs9C/view?usp=sharing)
- Original Dataset size from hackathon - 29,03,024
- Dataset from hackathon is available at - https://s3-ap-southeast-1.amazonaws.com/he-public-data/dataset52a7b21.zip

![Initial Dataset](https://github.com/jshreyansh87/Multiclass-Product-Classification-ML/blob/main/Product%20Classification%20Images/Initial%20dataset.JPG)

## Aim of Project
The project is aimed to classify products according to available metadata (title, description, bullet points and brand) to a given node ID, by building ML algorithm that classifies products according to their data.

Main Challenge in this Project includes:
- Dealing with a large amount of messy data
- Cleaning the dataset
- Converting string data to a numerical vector and applying a suitable ML model

## Analyzing the Model
1. First step is to check out the given training dataset and for that I am reading the dataset into a pandas dataframe. Now, plotting a graph to check count of each browse node ID and checking the heatmap for null values.


![count vs node id](https://github.com/jshreyansh87/Multiclass-Product-Classification-ML/blob/main/Product%20Classification%20Images/count%20vs%20node%20id.JPG) ![Heatmap null values](https://github.com/jshreyansh87/Multiclass-Product-Classification-ML/blob/main/Product%20Classification%20Images/Heatmap%20null%20values.JPG)

2. Next step is to clean this dataset. For this task I have considered the following:
   - Dropping duplicate rows
   - Dropping rows without title
   - Deleting special chars from title, description, bullet points
   - Convert title, description, bullet points, brand to lower case
   - Replace null values in description, bullet points, brands with ''
   - Removing rows that do not contain description, bullet points, brands, all these values at the same time.
   - Joining title, description, bullet points and brands in a single string.

![cleaned dataset](https://github.com/jshreyansh87/Multiclass-Product-Classification-ML/blob/main/Product%20Classification%20Images/cleaned%20dataset.JPG)

3. Now the string created after cleaning dataset, needs to be processed to create a string that can be used to create vectors. For this process I have considered two approaches- lemmatizing and stemming. In my case I have used stemming because even though it's not as accurate as lemmatizing but it runs comparatively much faster than the other approach, and it was also giving me a better result. Steps to process the data include:
   - Tokenize the data (Creating token of each word in string)
   - Removing stop words (words that do not contribute to classify a data like 'is', 'the', 'and', 'or', etc.)
   - Apply Stemming or lemmatizing

4. Using TF-IDF vectorizer to create vectors for the stemmed string.

5. splitting the model into 80% training and 20% testing data.

6. Deploying LinearSVC model from sklearn.svm library to train the model and calculating the accuracy of the model. I got an **accuracy of 55.52%** (This accuracy increases as the training data size is increased, using google colab I was not able to run the model on a larger dataset as it was consuming the whole allocated RAM after which colab interrupts the execution). I have also plotted actual vs predicted browse node ID.

![actual vs pred](https://github.com/jshreyansh87/Multiclass-Product-Classification-ML/blob/main/Product%20Classification%20Images/actual%20vs%20predicted.JPG) ![count vs pred](https://github.com/jshreyansh87/Multiclass-Product-Classification-ML/blob/main/Product%20Classification%20Images/count%20vs%20predicted%20node.JPG)
