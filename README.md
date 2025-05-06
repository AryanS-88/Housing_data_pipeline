# Housing Data Pipeline

## Overview

This project focuses on building a data pipeline and predictive model for housing data analysis. The primary objective is to preprocess housing data, perform exploratory data analysis (EDA), and develop a regression model to predict housing prices.

## Dataset

The dataset used in this project is `housing.csv`, which contains various features related to housing in California. Each row represents a district, and the columns include attributes such as:

- `longitude` and `latitude`: Geographical coordinates
- `housing_median_age`: Median age of houses in the district
- `total_rooms` and `total_bedrooms`: Total number of rooms and bedrooms
- `population`: Total population in the district
- `households`: Total number of households
- `median_income`: Median income of households
- `median_house_value`: Median house value (target variable)
- `ocean_proximity`: Proximity to the ocean

## Project Structure

```
 Housing_data_pipeline/
├── Housing.ipynb
├── housing.csv
├── california.png
└── README.md
```

- `Housing.ipynb`: Jupyter Notebook containing the entire data pipeline and modeling process.
- `housing.csv`: Dataset used for analysis and modeling.
- `california.png`: Visualization image showing housing data distribution.
- `README.md`: Project documentation.

## Steps Taken

### 1. Data Loading

- Imported necessary libraries such as `pandas`, `numpy`, `matplotlib`, and `seaborn`.
- Loaded the dataset using `pandas.read_csv()`.

### 2. Exploratory Data Analysis (EDA)

- Displayed the first few rows of the dataset using `head()`.
- Checked for missing values and data types using `info()` and `describe()`.
- Visualized data distributions and relationships using histograms, scatter plots, and correlation matrices.
- Plotted geographical data to observe housing distribution across California.

### 3. Data Preprocessing

- Handled missing values in the `total_bedrooms` column by imputing with the median.
- Converted categorical variable `ocean_proximity` into numerical values using one-hot encoding.
- Created new features such as `rooms_per_household`, `bedrooms_per_room`, and `population_per_household` to enhance model performance.
- Scaled numerical features using `StandardScaler` to normalize the data.

### 4. Model Training

- Split the dataset into training and testing sets using `train_test_split()`.
- Trained a `LinearRegression` model on the training data.
- Evaluated model performance using metrics like Mean Squared Error (MSE) and Root Mean Squared Error (RMSE).
- Visualized actual vs. predicted values to assess model accuracy.

### 5. Model Evaluation

- Analyzed residuals to check for patterns that might indicate model shortcomings.
- Compared performance with other regression models like Decision Tree and Random Forest (if implemented).
- Fine-tuned model parameters to improve accuracy.

## Requirements

- Python 3.x
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- Jupyter Notebook

## Usage

1. Clone the repository:

   ```bash
   git clone https://github.com/AryanS-88/Housing_data_pipeline.git
   ```

2. Navigate to the project directory:

   ```bash
   cd Housing_data_pipeline
   ```

3. Install the required packages:

   ```bash
   pip install -r requirements.txt
   ```

4. Open the Jupyter Notebook:

   ```bash
   jupyter notebook Housing.ipynb
   ```

5. Run the cells sequentially to execute the data pipeline and model training.

## Results

- The Linear Regression model achieved an RMSE of approximately X (replace with actual value).
- Feature engineering significantly improved model performance.
- Visualizations provided insights into the data distribution and model predictions.


