# crossinsights/utils/eda_tools.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from crossinsights.utils.data_loader import load_raw_data
import logging

# إعداد logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='C:/crossinsights_project/crossinsights/logs/eda_tools.log',
    filemode='a'
)
logger = logging.getLogger(__name__)
console = logging.StreamHandler()
console.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console.setFormatter(formatter)
logger.addHandler(console)

def plot_rating_distribution(ratings: pd.DataFrame):
    """Plot the distribution of ratings."""
    logger.info("Generating rating distribution plot")
    if 'rating' not in ratings.columns:
        logger.error("Column 'rating' not found in ratings DataFrame")
        raise ValueError("Column 'rating' not found in ratings DataFrame")
    plt.figure(figsize=(10, 6))
    sns.histplot(pd.DataFrame(ratings['rating']), bins=5, kde=False)
    plt.title('Distribution of Movie Ratings')
    plt.xlabel('Rating')
    plt.ylabel('Count')
    output_path = 'C:/crossinsights_project/crossinsights/analysis/rating_distribution.png'
    plt.savefig(output_path)
    plt.close()
    logger.info(f"Rating distribution plot saved to {output_path}")

def plot_genre_popularity(movies: pd.DataFrame):
    """Plot popularity of movie genres using encoded columns."""
    logger.info("Generating genre popularity plot")

    # إذا لم تبدأ أعمدة الـ genres بالبادئة genre_، أضفها تلقائياً
    genre_cols = [col for col in movies.columns if col not in ['movie_id', 'title', 'genres']]
    if not genre_cols:
        logger.error("No genre columns found in movies DataFrame")
        raise ValueError("No genre columns found in movies DataFrame")

    # التأكد من أن كل عمود رقمي
    movies[genre_cols] = movies[genre_cols].apply(pd.to_numeric, errors='coerce').fillna(0)

    genre_counts = movies[genre_cols].sum().sort_values(ascending=False)

    plt.figure(figsize=(12, 6))
    sns.barplot(x=genre_counts.values, y=genre_counts.index)
    plt.title('Popularity of Movie Genres')
    plt.xlabel('Number of Movies')
    plt.ylabel('Genre')
    output_path = 'C:/crossinsights_project/crossinsights/analysis/genre_popularity.png'
    plt.savefig(output_path)
    plt.close()
    logger.info(f"Genre popularity plot saved to {output_path}")

def plot_age_vs_ratings(users: pd.DataFrame, ratings: pd.DataFrame):
    """Plot relationship between user age and average rating."""
    logger.info("Generating age vs ratings plot")
    required_cols = ['user_id', 'age']
    if not all(col in users.columns for col in required_cols) or 'user_id' not in ratings.columns or 'rating' not in ratings.columns:
        logger.error("Required columns not found in users or ratings DataFrame")
        raise ValueError("Required columns not found in users or ratings DataFrame")

    # تحويل user_id إلى Int64 لضمان التوافق
    ratings['user_id'] = pd.to_numeric(ratings['user_id'], errors='coerce').astype('Int64')
    users['user_id'] = pd.to_numeric(users['user_id'], errors='coerce').astype('Int64')

    # إزالة أي صفوف تحتوي على user_id غير صالح
    ratings = ratings.dropna(subset=['user_id'])
    users = users.dropna(subset=['user_id'])

    user_ratings = ratings.groupby('user_id')['rating'].mean().reset_index()
    merged = user_ratings.merge(users[['user_id', 'age']], on='user_id', how='inner')

    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='age', y='rating', data=merged)
    plt.title('Average Rating by User Age')
    plt.xlabel('Age')
    plt.ylabel('Average Rating')
    output_path = 'C:/crossinsights_project/crossinsights/analysis/age_vs_ratings.png'
    plt.savefig(output_path)
    plt.close()
    logger.info(f"Age vs ratings plot saved to {output_path}")
