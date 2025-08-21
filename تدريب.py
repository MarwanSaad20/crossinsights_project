from crossinsights.models.placeholder import train_model

# نفترض أن لديك ratings_clean.csv
import pandas as pd
ratings = pd.read_csv("C:/crossinsights_project/crossinsights/data/processed/ratings_clean.csv")

svd_model = train_model(ratings)
