# Predict-Clicked-Ads-Customer-Classification-by-using-Machine-Learning

![image](https://github.com/riyouuyt/Predict-Clicked-Ads-Customer-Classification-by-using-Machine-Learning/assets/122600889/4e7663c9-1223-48cf-9beb-aa17ecf8d775)


## Project Overview 🌟 
A company in Indonesia wants to evaluate the effectiveness of the advertisements they broadcast. This is crucial for the company to assess the success of their marketing efforts in attracting customers to view the advertisements. By analyzing historical advertising data and identifying insights and patterns, this project aims to assist the company in defining their marketing targets. The primary objective of this project is to create a machine learning classification model that can determine the right customer targets.

## Goals 🎯
* Create Effective Model: Build a machine learning model to predict ad clicks accurately.
* Enhance Marketing Strategy: Use data insights to improve the company's marketing campaigns.
* Optimize Targeting: Help the company target the right audience for better ad performance.

## Objectives 📋 
* Model Development: Develop a robust ad click prediction model.
* Insight Generation: Identify actionable insights from historical ad data.
* Data-Driven Decisions: Promote data-driven marketing decisions within the company.

## Data Preparation for Clicked Ads Prediction

### Python Libraries for The Project:

- **Pandas**: is used for data manipulation, analysis, and cleaning. It provides data structures and functions for working with structured data.
  
![image](https://github.com/riyouuyt/Investigate-Hotel-Business-using-Data-Visualization/assets/122600889/5303d66a-c02d-4384-9587-157f41f508e9)

- **NumPy**: is a fundamental library for scientific computing in Python. It's used for numerical operations, including handling arrays and matrices.
  
![image](https://github.com/riyouuyt/Investigate-Hotel-Business-using-Data-Visualization/assets/122600889/cc10b08d-0ea2-4d93-819b-8d1b3a114ea5)

- **Seaborn (sns)**: is a data visualization library based on Matplotlib. It's used to create attractive and informative statistical graphics.
  
![image](https://github.com/riyouuyt/Investigate-Hotel-Business-using-Data-Visualization/assets/122600889/aaf9fb56-b8af-4e28-aaaf-50bc5d838e20)

- **Matplotlib**: is a widely used data visualization library. It's employed to create a wide range of static, animated, and interactive plots and figures.

![image](https://github.com/riyouuyt/Investigate-Hotel-Business-using-Data-Visualization/assets/122600889/77de4ab7-5558-4307-a429-4ab239557797)

- **scikit-learn** 🧠: Utilized for machine learning tasks.

![image](https://github.com/riyouuyt/Predict-Clicked-Ads-Customer-Classification-by-using-Machine-Learning/assets/122600889/3ac3314f-74d7-403a-a47a-585d7d517cc6)


### Dataset Overview:

- **Total Rows**: 1000
- **Total Columns**: 11

### Columns Explanation:

- `Daily Time Spent on Site`: Average time spent by users on the website.
- `Age`: Age of users.
- `Area Income`: Income level of users' areas.
- `Daily Internet Usage`: Daily internet usage by users.
- `Male`: Gender of users (binary: "Male" or "Female").
- `Timestamp`: Timestamps or dates of user data.
- `Clicked on Ad`: Indicator of whether a user clicked on an ad (binary: "Yes" or "No").
- `City`: City where users are located.
- `Province`: Province or region of users.
- `Category`: Categorical variable or category.

## Exploratory Data Analysis

### Numerical Features

![image](https://github.com/riyouuyt/Predict-Clicked-Ads-Customer-Classification-by-using-Machine-Learning/assets/122600889/18f61f6d-cc05-4498-ae7f-0c2ae17c0c48)

📈 Key Insights based on histogram:

* **Daily Time Spent on Site:**
  * Nearly symmetrical distribution.
  * Slight negative kurtosis indicates a flatter distribution.

* **Age:**
  * Slightly positively skewed distribution.
  * Negative kurtosis suggests a relatively flat peak.

* **Area Income:**
  * Moderately negatively skewed distribution.
  * Kurtosis close to zero indicates a normal-like distribution.

* **Daily Internet Usage:**
  * Nearly symmetrical distribution.
  * Negative kurtosis indicates a flat peak.


### Categorical Features:

* **Province Distribution:**

![image](https://github.com/riyouuyt/Predict-Clicked-Ads-Customer-Classification-by-using-Machine-Learning/assets/122600889/38ccba5a-099f-4bac-85b0-951677e7e6c2)
 
The top 5 provinces with the highest representation among users are:
  * Daerah Khusus Ibukota Jakarta (253 users)
  * Jawa Barat (210 users)
  * Jawa Timur (90 users)
  * Banten (76 users)
  * Jawa Tengah (53 users)
Jakarta's capital region, Daerah Khusus Ibukota Jakarta, has the highest user count.

* **City Distribution:**

![image](https://github.com/riyouuyt/Predict-Clicked-Ads-Customer-Classification-by-using-Machine-Learning/assets/122600889/1ac72a2e-6356-46a0-9258-4dc4ec0ee64e)

The top 5 cities with the highest representation among users are:
  * Surabaya (64 users)
  * Bandung (64 users)
  * Jakarta Timur (63 users)
  * Jakarta Selatan (61 users)
  * Jakarta Barat (57 users)
Surabaya and Bandung are the most prominent cities among users.

* **Ads Categories 📢:**

![image](https://github.com/riyouuyt/Predict-Clicked-Ads-Customer-Classification-by-using-Machine-Learning/assets/122600889/70944a4a-0fee-443d-a041-41483b197684)

We examined the distribution of users among different ad categories. The top ad categories include `Otomotif`, `House`, `Health`, `Fashion`, `Food`, `Furniture`, `Travel`, `Electronic`, `Finance`, and `Bank`. Understanding these categories helps us tailor our marketing strategies to target specific interests and preferences of our users.

* **Gender Distribution 👫:**

![image](https://github.com/riyouuyt/Predict-Clicked-Ads-Customer-Classification-by-using-Machine-Learning/assets/122600889/9f64ab0d-2994-453c-bcab-efede7ce784f)

We analyzed the gender distribution among our users. Approximately 47.9% of users identify as Male, while 52.1% identify as Female. Additionally, there were 3 cases (0.3%) with missing gender information. This gender breakdown allows us to customize our marketing campaigns to resonate with different segments of our audience.

###Biavariate Analysis:

* Heatmap analysis:

![image](https://github.com/riyouuyt/Predict-Clicked-Ads-Customer-Classification-by-using-Machine-Learning/assets/122600889/887be7d4-d1e0-4eea-9201-604f36a22dcf)

* Scatterplot Analysis:
  
![image](https://github.com/riyouuyt/Predict-Clicked-Ads-Customer-Classification-by-using-Machine-Learning/assets/122600889/d6eea6f4-50ff-45b0-a8e2-e554b89a48af)

summary for the all the correlation columns above:

* Daily Time Spent on Site vs. Age: Negative correlation (-0.3314). Younger users tend to spend more time on the site.

* Daily Time Spent on Site vs. Area Income: Positive correlation (0.3083). Users from higher-income areas spend slightly more time on the site.

* Daily Time Spent on Site vs. Daily Internet Usage: Strong positive correlation (0.5183). More time on the site is associated with higher daily internet usage.

* Age vs. Area Income: Negative correlation (-0.1793). Younger users often reside in lower-income areas.

* Age vs. Daily Internet Usage: Negative correlation (-0.3705). Younger users have higher daily internet usage.

* Area Income vs. Daily Internet Usage: Positive correlation (0.3381). Higher area income relates to higher daily internet usage.

* Strongest Correlation: Daily Time Spent on Site and Daily Internet Usage (0.5183). A strong positive relationship exists.

* Age vs. Daily Time Spent on Site: Negative correlation (-0.3314). Older users spend less time on the site.

In summary, user engagement is positively linked to internet usage, with younger users being more active. Additionally, users from higher-income areas tend to engage more with the site. However, age and area income are negatively related, suggesting that younger users often reside in lower-income areas. These insights can inform marketing strategies and target demographic


## 📈 Customer Type and Behavior Analysis on Advertisement 📈

In our comprehensive analysis, we delved into customer behavior patterns and demographics, shedding light on their interaction with digital advertisements.

* **Ads Categories 📢:** We examined the distribution of users across various ad categories, helping us understand their preferences and interests. The top categories include `Otomotif`, `House`, `Health`, `Fashion`, `Food`, `Furniture`, `Travel`, `Electronic`, `Finance`, and `Bank`. This knowledge empowers us to tailor our marketing strategies to align with specific customer interests.

* **Gender Distribution 👫:** Our analysis revealed a balanced gender distribution, with approximately 47.9% of users identifying as Male and 52.1% as Female. There were 3 cases (0.3%) with missing gender information. This insight enables us to create targeted and inclusive marketing campaigns.

* **Customer Type 🎯:** Through behavioral analysis, we identified different customer types based on their interaction with advertisements. By categorizing users into various segments, we gain a deeper understanding of their preferences, needs, and responsiveness to our marketing efforts.

* **Age Groups 🎂:** We explored age intervals to determine which age ranges are more likely to engage with advertisements. This information aids in crafting age-specific marketing strategies for better results.


## Data Preprocessing

### Checking Null Values in Dataset 🕵️

In our dataset, we meticulously examined the presence of missing values. Here's a quick overview:

![image](https://github.com/riyouuyt/Predict-Clicked-Ads-Customer-Classification-by-using-Machine-Learning/assets/122600889/7dc3dacf-6bf3-47a9-b540-5d573d9bcf84)

📊 Columns with Null Values:
* `Daily Time Spent on Site` and `Area Income` each had a 1.3% null rate.
* `Daily Internet Usage` had a 1.1% null rate.
* `Male` exhibited a mere 0.3% of missing values.
* `Unnamed: 0`, `Age`, `Timestamp`, `Clicked on Ad`, `city`, `province`, and `category` were entirely free from null values.

Ensuring data integrity is essential for reliable analysis. We'll handle these missing values appropriately to maintain data quality. 🛠️

### 🧹 Data Cleaning and Preprocessing Summary 🧼

Our data underwent a comprehensive cleaning and preprocessing journey, ensuring its readiness for analysis. Here's what we accomplished:

* 🚮 Handling Missing Values: We meticulously addressed missing values, ensuring minimal data loss: `Daily Time Spent on Site`, `Area Income`, `Daily Internet Usage`, and `Male` columns had missing values, which were handled appropriately. `Unnamed: 0`, `Age`, `Timestamp`, `Clicked on Ad`, `city`, `province`, and `category` columns remained free from null values.

* 🧽 Duplicate Data: We identified and successfully removed duplicate rows from the dataset, leaving us with a pristine dataset, completely devoid of duplicates.

* 🔄 Feature Encoding: We encoded categorical columns `Gender` and `Clicked on Ad` using one-hot encoding, making them ready for machine learning models.

* 📛 Column Renaming: We renamed columns for clarity and consistency, including renaming `Gender` columns for better interpretation.

* 🔪 Column Removal: We removed the `Unnamed: 0` column, which didn't contribute to our analysis.

Now, with our dataset clean, structured, and encoded, we're primed for in-depth analysis and predictive modeling! 🚀🔍

## Prepare for The Modelling
* 🎯 Splitting Data: We divided our dataset into features (X) and the target variable (y). The 'Clicked on Ad_Yes' column serves as our target, while the remaining columns comprise our features.
* 
* 🕰️ Extracting Date Information: To unlock valuable insights, we extracted date-related details from the `Timestamp` column:
  * We converted the `Timestamp` column to datetime format to ensure accurate time-related calculations.
  * We created new columns, including `Year`, `Month`, `Week`, and `Day`, to dissect the timestamp data into various time components.

These meticulous preparations set the stage for our exploratory data analysis and predictive modeling, empowering us to uncover meaningful patterns and make informed decisions! 🚀📈

##Modelling
We conduct two experiments, one before normalization/standardization and another after applying these techniques, to understand how they influence our model's performance. This comparison allows us to make informed decisions about whether to include these data preprocessing steps in our predictive modeling pipeline.

In the first experiment, we work with raw, unnormalized data to assess the initial performance of the model. We then contrast this with the second experiment, where we apply normalization/standardization techniques to the data. By doing so, we aim to observe any notable improvements in model accuracy and performance and identify which features play a more significant role under different data transformation scenarios.

This comparative analysis will guide us in determining the most effective approach for handling our data, ensuring we build a predictive model that can reliably classify ad clicks with precision.

### Modelling Experiment

1. **Experiment 1 (Without Normalization/Standardization):**

![image](https://github.com/riyouuyt/Predict-Clicked-Ads-Customer-Classification-by-using-Machine-Learning/assets/122600889/65363119-0ee4-435f-9cd3-9a4670cbfa1c)

Model Observation:
* Low accuracy (49%) and poor performance.
* Top Features: Daily Time Spent on Site, Daily Internet Usage.

2. Experiment 2 (After Normalization/Standardization):

![image](https://github.com/riyouuyt/Predict-Clicked-Ads-Customer-Classification-by-using-Machine-Learning/assets/122600889/f850fd96-8116-4932-bc1e-12789d71eceb)

Model Observation:
* High accuracy (96%) and improved performance.
* Top Features: Daily Time Spent on Site, Daily Internet Usage.

Model Summary:

* Experiment 1 lacked data normalization and performed poorly in predicting ad clicks. The top features, Daily Time Spent on Site and Daily Internet Usage, were not sufficient for accurate predictions.
* Experiment 2, with data normalization, significantly improved accuracy and precision. It successfully identified potential ad clickers.
* The top features affecting predictions are Daily Time Spent on Site and Daily Internet Usage, highlighting their importance in understanding user behavior.
* Data normalization plays a crucial role in enhancing model performance, making it a recommended practice.

##Business Recommendation

### Feature Importance

![image](https://github.com/riyouuyt/Predict-Clicked-Ads-Customer-Classification-by-using-Machine-Learning/assets/122600889/61852971-25a8-47fe-ae16-8c22910d9977)

The feature importance analysis has identified that "Daily Internet Usage" and "Daily Time on Site" are the two most influential factors affecting user behavior and ad click-through rates. 🚀

* **Daily Internet Usage**: This feature is a critical determinant of user engagement. Users who spend more time online tend to be more active and attentive. This is a strong indicator of their responsiveness to ads.

* **Daily Time on Site**: The time users spend on the site directly impacts their exposure to ads. Longer visits provide more opportunities for users to interact with advertisements.

These two factors, "Daily Internet Usage" and "Daily Time on Site," are highly valuable in understanding user behavior and optimizing ad campaigns. 📊💡

### Age Group Distribution📊
Based on the feature importance, The age is pretty importance and can add some recommendation into the business distribution is predominantly concentrated in the "Adult" and "Senior" categories, with these two groups making up nearly the entire user base. The "Young" and "Elderly" groups represent only a small portion of the users.

![image](https://github.com/riyouuyt/Predict-Clicked-Ads-Customer-Classification-by-using-Machine-Learning/assets/122600889/7cd04a5c-8195-4072-801c-3c3a6cc68285)

Business Recommendation 🚀

* Target Adult and Senior Groups: 🎯 The marketing strategy should be primarily designed to target the "Adult" and "Senior" age groups since they make up the majority of users. Tailor advertisements, promotions, and content to resonate with the interests and preferences of these age groups.

* Explore Opportunities: 🔍 While the "Young" and "Elderly" groups are smaller in number, they still represent potential market segments. It may be worth exploring specific strategies to attract and engage these demographics, considering their unique interests and needs.

This business recommendation aims to optimize marketing efforts by focusing on the most significant user segments while exploring potential growth opportunities. 📈

## Business Simulation

💰Business Schemas💰

Company Profile: CarMend Pro 🚗🔧

CarMend Pro is a customer-focused automotive services company specializing in car maintenance and repair. We connect car owners with trusted mechanics, offering convenient and reliable solutions to keep their vehicles in top condition. Our transparent pricing, user-friendly platform, and dedication to innovation make us a leader in the automotive industry.

Marketing Team Target: 🎯
* Objective:
To drive traffic to the CarMend Pro website, generate interest in our services, and increase sales. 🚗💻📈

* Target Audience: 🎯
Car owners looking for automotive maintenance and repair services. 🚗🔧💡

### Simulation Story

Before the machine learning  for the classification customers exist, the marketing teams are assuming the prospect customer which there are:

* Interested Prospects (Assumed 30%): 🤩
  * Clicked on the ad and visited the website.
  * Spent more time on the site, viewed multiple pages.
  * Engaged with interactive tools like the "Price Estimator."

* Non-Interested Prospects (Assumed 70%): 😕
  *  Clicked on the ad but didn't engage much.
  * Bounced quickly from the landing page.

* Marketing Team's Assumed Clicked Customers: 📊
  * Total Clicks: 193
  * Interested Prospects (30% of clicks): 58
  * Non-Interested Prospects (70% of clicks): 135

### Expected Revenue Sales and CVR (Conversion Rate): 💰📈

* Marketing Team's Expected Results:
  * Total Conversions: 58 (assuming all 58 interested prospects converted).
  * Total Revenue Sales: $11,600 (58 conversions x $200 average transaction value).
  * Conversion Rate (CVR): 100% (since all expected interested prospects are assumed to convert).

* Non-Interested Prospects: 💰
  * Assuming a 2% conversion rate for non-interested prospects.
  * Expected Conversions: 3 (135 x 2%).
  * Average Transaction Value: $200
  * Revenue from Non-Interested Prospects: $600 (3 x $200).

* Total Expected Results:
  * Total Conversions: 61 (58 from interested + 3 from non-interested)
  * Total Revenue Sales: $12,200 ($11,600 from interested + $600 from non-interested)
  * Conversion Rate (CVR): Approximately 31.6% (61 conversions out of 193 clicks)


![Screenshot 2023-11-02 003123](https://github.com/riyouuyt/Predict-Clicked-Ads-Customer-Classification-by-using-Machine-Learning/assets/122600889/1355e767-e053-4ae5-b350-de3ef03c98a0)

Certainly, let's explain that the model was created using the DecisionTreeClassifier and compare the revenue and Conversion Rate (CVR) with and without the machine learning model

It achieved an impressive accuracy of 94.3%. The confusion matrix revealed the following:

True Positives (TP): 94
True Negatives (TN): 88
False Positives (FP): 6
False Negatives (FN): 5

### Comparison Revenue with Machine Learning and without Machine Learning

* Marketing Team's Expected Results:
  * Total Conversions: 58 (assuming all 58 interested prospects converted).
  * Total Revenue Sales: $11,600 (58 conversions x $200 average transaction value).
  * Conversion Rate (CVR): 100% (since all expected interested prospects are assumed to convert).

* Machine Learning Model's Results:
  * Total Conversions: 182 (as previously mentioned).
  * Total Revenue Sales: $36,400 (as previously mentioned).
  * Conversion Rate (CVR): Approximately 94.3% (as previously mentioned).

Comparison Summary:
With Machine Learning (ML) 🤖:

The model predicts significantly higher conversions and revenue than the marketing team's initial expectations. ML delivers a Conversion Rate (CVR) of around 94.3%, which is lower than the assumed 100% but still quite high. 📈💰Without Machine Learning (ML) The marketing team's expected results would fall short, achieving only 58 conversions and $11,600 in revenue. The CVR remains high, but it doesn't reach 100%. 🙅‍♂️🤷‍♂️
