# Gurjar_Upasani_Final_Project_IA651D1_U24

# Overview

In this project, the objective is to predict whether the person has Diabetes or not based on various features such as Pregnancies, Insulin Level, Age, BMI. The data set that has used in this project has taken from the kaggle . "This dataset is originally from the National Institute of Diabetes and Digestive and Kidney Diseases. The objective of the dataset is to diagnostically predict whether or not a patient has diabetes, based on certain diagnostic measurements included in the dataset. Used classification machine learning approaches.

# Motivation

The motivation was to experiment with end to end machine learning model " Diabetes is an increasingly growing health issue due to our inactive lifestyle. If it is detected in time then through proper medical treatment, adverse effects can be prevented. To help in early detection, technology can be used very reliably and efficiently. Using machine learning we have built a predictive model that can predict whether the patient is diabetes positive or not.". This is also sort of fun to work on a project like this which could be beneficial for the society.

# Data Summary & Column Description


**Pregnancies** =  'the condition or period of being pregnant.'

**Glucose**    = 'type of sugar in the blood and is the major source of energy for the body's cells.'

**BloodPressure** = 'the force of blood pushing against the walls of your arteries as your heart pumps blood through your body'

**SkinThickness** = 'Thick skin is thicker due to it containing an extra layer in the epidermis, called the stratum lucidum'

**Insulin** = 'a hormone produced in the pancreas by the islets of Langerhans, which regulates the amount of glucose in the blood. The lack of insulin causes a form of diabetes.'

**BMI**  = 'Body mass index (BMI) is a ratio of a person's weight in kilograms to the square of their height in meters.'

**DiabetesPedigreeFunction** = 'The 'DiabetesPedigreeFunction' is a function that scores the probability of diabetes based on family history, with a realistic range of 0.08 to 2.42'

**Age**. = 'The length of time that a person has lived'

**Outcome** = Diabetes, Yes or No




# Exploratory data analysis(EDA)

**BloodPressure Variable**

First we checked the distribution of data. data is normally distributed.

![download (2)](https://github.com/user-attachments/assets/12f68e00-9a6b-4862-bfed-85a7be59f91e)

**Insulin Variable**

Insulin variable distribution was right skewed upon checking found that there were a lot of 0 values. To make data normally disturbed used median method and replaced 0 with median of non zero values.

![download (3)](https://github.com/user-attachments/assets/137e652c-0419-4730-a02b-6de6ab861fb7)


**Median** 

Replaced all rows with NA or 0 with Median. We also tried Mean but distribution was right skewed so found median more useful approach for our use case.


**Output variable**

Explored the output variable and seems data is balanced where 65% values are o(no diabetes)  and 35% values are 1(yes diabetes).

![download (4)](https://github.com/user-attachments/assets/e404df7e-1840-4cef-9b82-bc95cdd0c930)



**Correlation Matrix**

![download (5)](https://github.com/user-attachments/assets/ddc4bf0e-fa8a-4bfa-959c-7ddb19508858)



**Pair scatter plot**

![download (7)](https://github.com/user-attachments/assets/68d96b0f-c764-4763-a8b9-cb96977088aa)



**PCA**

![download](https://github.com/user-attachments/assets/3385b924-67bc-4dd8-9fc3-0d2926467668)


![download (1)](https://github.com/user-attachments/assets/e5085443-2df2-4e4d-b3f3-28c884be4f5d)


# Machine learning Mdel 

**Decision Tree**

![download (8)](https://github.com/user-attachments/assets/93e968b1-922d-440d-8868-8037f46fcf71)

![download (9)](https://github.com/user-attachments/assets/9e4e1cc3-d284-4f51-87e5-61c5d34dbbc9)

<img width="456" alt="Screenshot 2024-08-02 at 10 20 57 AM" src="https://github.com/user-attachments/assets/91575854-f134-4950-8a47-e03e0d613536">

<img width="795" alt="Screenshot 2024-08-02 at 10 21 09 AM" src="https://github.com/user-attachments/assets/5964e809-b728-4c01-853c-958929e7e77b">




**Random Forest**

![download (10)](https://github.com/user-attachments/assets/ad27f656-1094-4bff-9da4-a40b1e50031a)

<img width="403" alt="Screenshot 2024-08-02 at 10 24 17 AM" src="https://github.com/user-attachments/assets/9b5a6b95-861e-4708-b5e4-e9a3e262f1c5">

<img width="765" alt="Screenshot 2024-08-02 at 10 24 27 AM" src="https://github.com/user-attachments/assets/c2d0e931-1d10-4691-8344-24215718e66f">


**Feature Importance in Random Forest**

![download (11)](https://github.com/user-attachments/assets/2bab6930-0728-45ae-8e92-65e7c6dd9de7)

<img width="811" alt="Screenshot 2024-08-02 at 10 26 10 AM" src="https://github.com/user-attachments/assets/73ae7ebf-b32e-4daf-ad86-16d9a620ca80">



# Learning Objective
The following points were the objective of the project .

#Data gathering

#Descriptive Analysis

#Data Visualizations

#Data Preprocessing

#Data Modelling

#Model Evaluation

#Model Deployment

#Technical Aspect

#Training a machine learning model using scikit-learn.

#A user has to put details like Number of Pregnancies, Insulin Level, Age, BMI etc .
Once it get all the fields information , the prediction is displayed.


