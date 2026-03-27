# -House-Price-Predictor
# Name-Varun Vaish
# Reg no - 25BAI11368
# Project Title
House Price Predictor: An End-to-End Machine Learning Pipeline

# Overview of the Project

This project is a Python-based Machine Learning application designed to predict real estate prices based on specific property features. It demonstrates the complete Data Science pipeline, including data generation, preprocessing (cleaning and encoding), model training, and evaluation.

The system uses a Linear Regression algorithm to understand relationships between variables like square footage, number of bedrooms, age, and location to output a predicted price. It includes an interactive console interface allowing users to input their own house details for real-time price estimation.

# Features

Synthetic Data Generation: Creates a realistic dataset programmatically, eliminating the need for external CSV files for demonstration purposes.

## Automated Data Cleaning: 
Handles missing data points using statistical imputation strategies (replacing missing values with the mean).

## Categorical Encoding: 
Converts text-based location data (Rural, Suburb, City) into numerical format using One-Hot Encoding.

## Model Performance Metrics: 
Automatically calculates and displays the Mean Squared Error (MSE) and R² Score to evaluate accuracy.

## Interactive Prediction: 
A user-friendly command-line interface that accepts user inputs and returns a predicted price.

## Data Visualization: 
Generates a scatter plot comparing Actual vs. Predicted prices to visually assess model performance.

# Technologies/Tools Used

## Python: 
The core programming language used for the logic.

## Pandas: 
Used for DataFrames, data manipulation, and One-Hot Encoding.

## NumPy: 
Used for numerical operations and array handling.

## Scikit-Learn (sklearn):

1. Splitting data (train_test_split)

2. Imputing missing values (SimpleImputer)

3. The Machine Learning Algorithm (LinearRegression)

4. Metrics (mean_squared_error, r2_score)

5. Matplotlib: Used to plot the "Actual vs. Predicted" graph.

# Steps to Install & Run the Project

## Prerequisites: 
Ensure you have Python installed on your machine.

## Installation: 
Open your terminal or command prompt and install the required libraries using pip:
pip install pandas numpy matplotlib scikit-learn

## Running the Project:

1. Save the code in a file named house_price_predictor.py.

2. Navigate to the folder where the file is saved.

3. Run the script using the command:
python house_price_predictor.py

# Instructions for Testing
Once the script is running, follow these steps to test the functionality:

## Review Training Log: 
The console will first display the raw data, the cleaned data, and the model coefficients. Verify that the "Model trained successfully!" message appears.

## Interactive Input: 
When the prompt --- Live Prediction (Interactive) --- appears:

1. Square Footage: Enter a number (e.g., 2500).

2. Bedrooms: Enter a number (e.g., 4).

3. Age: Enter the age of the house in years (e.g., 10).

4. Location: Enter one of the valid locations: City, Suburb, or Rural.

## View Result: The program will calculate and print the Estimated Price.

## Graph: A window will pop up showing the visualization. Close the window to terminate the program.

## Screenshots
<img width="610" height="685" alt="Screenshot 2025-11-23 at 8 06 48 PM" src="https://github.com/user-attachments/assets/60a04a18-5492-438b-b0ac-49bfd527cc47" />
<img width="753" height="488" alt="Screenshot 2025-11-23 at 8 07 03 PM" src="https://github.com/user-attachments/assets/f189f935-44fc-47da-b1aa-1adae246e3c9" />
