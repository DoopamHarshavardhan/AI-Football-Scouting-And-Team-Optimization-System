__AI Football Scouting & Team Optimization Dashboard__

An AI-powered football scouting and team optimization system built using Data Analytics, Machine Learning, and Streamlit.
The project helps analyze players, predict future performance, identify playing archetypes, recommend similar players, and build optimized teams.

-------------------------------------------------------------------------------------------
__Key Features :__

__Player Analysis__

-> Detailed player profile with age, position, overall rating, potential, and growth

-> Skill analysis (pace, shooting, passing, dribbling, defending, physical)

-> Visual charts for skills and playing style

->Rule-based AI role recommendation

----------------

__Machine Learning – Future Overall Prediction__

-> Supervised ML (Random Forest Regressor)

-> Predicts a player’s future/peak overall rating

-> Displays expected growth and model confidence

-> Includes feature-importance based explanation

---------------------------



__Player Archetype Detection__

-> Unsupervised ML (KMeans Clustering)

-> Groups players into archetypes such as:

-> Playmaker

-> Goal Poacher

-> Defensive Anchor

-> Pace Winger

-> Box-to-Box Midfielder

-> Integrated into player profile for playing-style understanding

---------------------------

__Similar Player Recommendation__

-> K-Nearest Neighbors (KNN)

-> Recommends top 5 players with similar playing style

-> Displayed in the sidebar for quick scouting comparison

-------------------------------

__Pitch Visualization__

-> Visual representation of player positions on a football pitch

-> Accurate positional mapping for team formation

--------------------------------
__Team Builder__

-> Build a complete playing XI

-> Hybrid ML + rule-based decision system

-> Player replacement suggestions based on role fit and performance

-> Ensures one player per position with squad balance

------------------------------------
__Machine Learning Techniques Used:__

Feature_________________________________ML_________________________Type	Algorithm

Future Overall Prediction______________Supervised Learning________Random Forest Regressor

Player Archetype Clustering___________Unsupervised Learning_____KMeans

Similar Player Recommendation_______Instance-based Learning___KNN

-----------------------------------------
__Tech Stack__
-> Python

-> Streamlit – interactive dashboard

-> Pandas, NumPy – data processing

-> Scikit-learn – ML models

-> Matplotlib / Seaborn – visualization

-> Joblib – model persistence

------------------------------------------

__AI-Football-Scouting-Dashboard/__

├── app.py

├── data.csv

├── requirements.txt

├── README.md

├── future_overall_rf_model.joblib

├── model_features.joblib

├── player_archetype_kmeans.joblib

├── archetype_scaler.joblib

├── archetype_features.joblib

├── similar_player_knn.joblib

├── similarity_scaler.joblib

├── similarity_features.joblib

├── future_overall_prediction.ipynb

├── PLAYER_ARCHETYPE_CLUSTERING.ipynb

├── SIMILAR_PLAYER_RECOMMENDATION.ipynb

-----------------------------
__How to Run the Project Locally__

1️. Clone the repository

  git clone https://github.com/DoopamHarshavardhan/AI-Football-Scouting-Dashboard.git

  cd AI-Football-Scouting-Dashboard

2️. Install dependencies

pip install -r requirements.txt

3️. Run the Streamlit app

streamlit run app.py

------------------------------
__Machine Learning Workflow__

-> Data preprocessing and feature engineering (Jupyter Notebook)

-> Model training and evaluation

-> Model persistence using joblib

-> Real-time inference inside Streamlit dashboard

-----------------------------------

__Project Highlights__

-> Combines data analytics + multiple ML paradigms

-> Clean separation between notebooks (training) and app (deployment)

-> Hybrid ML-assisted decision system for team building

-> Designed with product-style UX, not just experimentation

-------------------------

__Author__

Doopam Harshavardhan

B.Tech (4th Year)

Aspiring Data Scientist / ML Engineer

----------
__License__

This project is for educational and learning purposes.
