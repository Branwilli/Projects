import os
import re 
import json

import numpy as np
import pandas as pd
import yfinance as yf
import warnings

from datetime import date, timedelta, timezone
from newspaper import Article
from dotenv import load_dotenv
from transformers import pipeline
from gdeltdoc import GdeltDoc, Filters

from indicators import add_features, add_targets, standardize_df, filter_data, smooth_labels

warnings.filterwarnings('ignore')
load_dotenv()

API = os.getenv('NEWS_API')
IMPACT_KEYWORDS = {

    # =========================
    # CENTRAL BANK / MONETARY POLICY
    # =========================
    'interest rate': 1.00, 'rate hike': 1.00, 'rate cut': 1.00, 'federal reserve': 1.00,
    'fed': 0.95, 'ecb': 0.95, 'central bank': 0.90, 'monetary policy': 0.90, 'hawkish': 0.90,
    'dovish': 0.90,

    # =========================
    # MACROECONOMIC DATA
    # =========================
    'inflation': 0.90, 'cpi': 0.90, 'employment': 0.85, 'nonfarm payroll': 1.00, 'unemployment': 0.85,
    'gdp': 0.80, 'growth': 0.75, 'recession': 0.90, 'forecast': 0.65, 'outlook': 0.65,

    # =========================
    # MARKET STRUCTURE / TECHNICALS
    # =========================
    'breakout': 0.70, 'trend line': 0.55, 'support': 0.50, 'resistance': 0.50, 'bullish': 0.65, 'bearish': 0.65,
    'decline': 0.60, 'rebound': 0.60, 'highs': 0.40, 'lows': 0.40, 'volatility': 0.75, 'bias': 0.50,

    # =========================
    # FOREX / CURRENCIES
    # =========================
    'gbpusd': 1.00, 'usd': 0.85, 'currency': 0.60, 'exchange rate': 0.85,

    # =========================
    # COMMODITIES
    # =========================
    'oil': 0.85, 'gold': 0.70,

    # =========================
    # EQUITIES / RISK APPETITE
    # =========================
    'stocks': 0.55, 'equities': 0.55, 'nasdaq': 0.60, 'dow jones': 0.60, 's&p 500': 0.70,
    'dax': 0.50, 'kospi': 0.40, 'composite index': 0.45,

    # =========================
    # GEOPOLITICS / GLOBAL SHOCKS
    # =========================
    'conflict': 0.90, 'hostilities': 0.85, 'gulf': 0.75, 'geopolitical risk': 0.95
}


def get_title(headline):
    """
    Extracts the 'title' string property from a raw news headline dictionary object.
    """
    return  headline['title']


def get_content(article):
    """
    Extracts the summary or content string field from a raw news article dictionary.
    """
    return article['summary']

def get_published(article):
    """
    Extracts the 'published' date field from a raw news article dictionary.
    """
    return article['published']


def fetch_forex_news(pair='GBPUSD=X', limit=10):
    """
    Fetches the latest macroeconomic or ticker-specific news articles from Yahoo Finance 
    """
    try:
        ticker = yf.Ticker(pair)    # Initialize the Ticker object for the specified currency pair
        news_items = ticker.news    # Retrieve the news items associated with the ticker

        if not news_items:
            print('No articles found.')
            return []
        
        articles = []

        # Iterate through the news items, extracting relevant fields and applying fallback defaults
        for item in news_items[:limit]:

            content = item.get('content', {})

            article_data = {
                'title': content.get('title'),
                'publisher': content.get('provider', {}).get('displayName'),
                'published': content.get('pubDate'),
                'summary': content.get('summary')
            }

            articles.append(article_data)

        return articles
    
    # Catch any exceptions that occur during the fetching and parsing process
    except Exception as e:
        print('Failed to fetch Yahoo Finance news: ', {e})
        return []


def fetch_gdelt_news(keyword='GBP USD'):
    """
    Connects to the GDELT Document API endpoint, extracts high-relevance URLs matching 
    macroeconomic keywords within specific countries, and uses the newspaper3k library 
    to parse article content, titles, and creation timelines.
    """
    try:
        # Establish date range for the past 105 days to ensure a robust dataset while avoiding stale news
        today = date.today().strftime("%Y-%m-%d")
        yesterday = (date.today() - timedelta(days=105)).strftime("%Y-%m-%d")
        gd = GdeltDoc() # Initialize the GDELT Document API client
    
        f = Filters(
            keyword=keyword,
            start_date=yesterday,
            end_date=today,
            language='English'
        )

        articles = gd.article_search(f) # Execute the search query 

        # Filter for specific countries and select only the URL and seen date for further processing
        urls = articles.loc[articles['sourcecountry'].isin(['United Kingdom', 'United States']), ['url', 'seendate']]
        urls = urls.to_records(index=False).tolist()    # Convert the filtered DataFrame to a list of tuples

        content = []

        # Content Extraction Loop: Iterates through the list of URLs, attempts to download and parse each article 
        for url, _ in urls:
            try:
                article = Article(url)
                article.download()
                article.parse()
                article.nlp() 

                article_dict = {
                    'title': article.title,
                    'published': article.publish_date,
                    'summary': article.summary
                }

                content.append(article_dict)

            except Exception as e:
                print(f"Skipping broken URL {url}: {e}")
                continue
            
            finally:
                if hasattr(gd, 'session'):
                    gd.session.close()  # Ensure the session is closed after processing each URL
    
        return content
    
    except Exception as e:
        print('GDELT fetch error: ', e)
        return pd.DataFrame()
    
    
def extract_title_relevance(text):
    relevant = []
    total_score = 0.0

    article = get_title(text)  

    if not article:
        return [], 0.0
    
    sentence = article.lower() 

    for term, weight in IMPACT_KEYWORDS.items():

        if term.lower() in sentence:
            relevant.append(sentence)
            total_score += weight

    return relevant, round(total_score, 4)


def extract_content_relevance(text):
    relevant = []
    total_score = 0.0

    article = get_content(text)

    if not article:
        return [], 0.0
    
    sentence = re.split(r'(?<=[.!?]) +', article)

    for s in sentence:
        s_lower = s.lower()
        sentence_score = 0.0
        matched = False

        for term, weight in IMPACT_KEYWORDS.items():
            if term.lower() in s_lower:
                matched = True
                sentence_score += weight

    if matched:
        relevant.append(s_lower)
        total_score += sentence_score

    return relevant, round(total_score, 4)


def calculate_recency_score(published, half_life_hour=24, min_score=0.05):
    
    try:
        published = pd.to_datetime(published, utc=True, unit='s', errors='coerce')

        if pd.isna(published):
            return min_score
        
        now = pd.Timestamp.now(tz=timezone.utc)

        age_hours = (now - published).total_seconds() / 3600.0
        age_hours = np.clip(age_hours, a_min=0, a_max=None)
        score = np.exp(-np.log(2) * age_hours / half_life_hour)

        score = np.maximum(score, min_score)
        score = np.nan_to_num(score, nan=min_score)

        return np.round(score, 4)
        
    except Exception as e:
        print('Recency score error: ', e)
        return min_score


def get_sentiment(articles, impact_score, recency_score):
    pipe = pipeline("text-classification", model="ProsusAI/finbert")

    total_score = []

    for entry in articles:
        try:
            sentiment = pipe(entry)

        except Exception as e:
            print(f"Sentiment error:  {e}")
            continue
        
        label = sentiment[0]['label'].lower()
        score = sentiment[0]['score']

        if score < 0.6:
            continue 

        if label == 'positive':
            val = 1
        elif label == 'negative':
            val = -1
        else:
            val = 0

        total_score.append(float((val * score) + (impact_score * recency_score)))

    if not total_score:
        return 0
    
    return np.clip(np.mean(total_score), -1, 1)


def sentiment_training_df(directory):

    for file in os.listdir(directory):
        if not file.endswith('.json'):
            continue

        path = os.path.join(directory, file)

        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            rows = []
            for i, article in enumerate(data):

                title = article.get('title', '')
                summary = article.get('summary', '')
                published = article.get('published', '')

                published = pd.to_datetime(published, utc=True, errors='coerce')

                article_dict = {
                    'title': title, 
                    'summary': summary,
                    'published': published
                }

                title_relevance, t_score = extract_title_relevance(article_dict)
                content_relevance, c_score = extract_content_relevance(article_dict)
                recency_impact = calculate_recency_score(published)

                title_sentiment_score = get_sentiment(title_relevance, t_score, recency_impact)
                content_sentiment_score = get_sentiment(content_relevance, c_score, recency_impact)
                score = (title_sentiment_score * 0.35) + (content_sentiment_score * 0.65)

                rows.append({'date': pd.to_datetime(published), 'score': score})
            
            df = pd.DataFrame(rows)
            df = (df.groupby('date').mean().reset_index())
            df.columns = ['datetime', 'sentiment_score']

            return df

        except Exception as e:
            print(f"Error processing {file}: {e}")
            continue


def sentiment_df(artices):
    rows = []
    for article in artices:
        published_date = get_published(article)
        
        if published_date is None or pd.isna(published_date) or str(published_date).strip() == "":
            #print("Skipping article due to missing publish date.")
            continue 
        
        published_date = pd.to_datetime(published_date, utc=True, errors='coerce')

        title_relevance, t_score = extract_title_relevance(article)
        content_relevance, c_score = extract_content_relevance(article)  

        recency_impact = calculate_recency_score(published_date)
        title_sentiment_score = get_sentiment(title_relevance, t_score, recency_impact)
        content_sentiment_score = get_sentiment(content_relevance, c_score, recency_impact)
        score = (title_sentiment_score * 0.35) + (content_sentiment_score * 0.65)

        rows.append({'date': pd.to_datetime(published_date), 'score': score})

    if not rows:
        return pd.DataFrame(columns=['datetime', 'sentiment_score'])
    
    df = pd.DataFrame(rows)
    df = (df.groupby('date').mean().reset_index())
    df.columns = ['datetime', 'sentiment_score']

    return df


def merge_df(sentiment_df, forex_df):
    forex_working = forex_df.copy()
    sentiment_working = sentiment_df.copy()

    forex_working['merge_date'] = forex_working.index.date
    sentiment_working['merge_date'] = pd.to_datetime(sentiment_working['datetime']).dt.date
    
    sentiment_daily = (
        sentiment_working.groupby('merge_date')['sentiment_score']
        .mean()
        .reset_index()
    )

    merged_df = forex_working.merge(sentiment_daily, on='merge_date', how='left')

    merged_df['sentiment_score'] = merged_df['sentiment_score'].ffill().fillna(0.0)

    merged_df = merged_df.drop(columns=['merge_date'])
    merged_df.index = forex_df.index
    
    return merged_df


def training_news_data(start, end, keyword='GBP USD'):
    #try:
        gd = GdeltDoc()

        f = Filters(
            keyword=keyword,
            start_date=start,
            end_date=end,
            language='English'
        )

        articles = gd.article_search(f)

        # Filter for UK and US sources
        urls = articles.loc[
            articles['sourcecountry'].isin(['United Kingdom', 'United States']),
            ['url', 'seendate']
        ]
        urls = urls.to_records(index=False).tolist()

        content = []

        # Extract content from each article
        for url, _ in urls:
            try:
                article = Article(url)
                article.download()
                article.parse()
                article.nlp()

                article_dict = {
                    'url': url,
                    'title': article.title,
                    'summary': article.summary,
                    'date': str(article.publish_date) if article.publish_date else None
                }
                content.append(article_dict)

            except Exception as e:
                print(f"Skipping broken URL {url}: {e}")
                continue

        # Close session once after processing
        if hasattr(gd, 'session'):
            gd.session.close()

        # Save results to JSON file
        os.makedirs("training_fundamentals", exist_ok=True)
        with open("training_fundamentals/fundamentals.json", "w", encoding="utf-8") as f:
            json.dump(content, f, ensure_ascii=False, indent=4)

        #return content  # Return the list for further use

    #except Exception as e:
        #print('GDELT fetch error: ', e)
        #return []



def process_and_load_articles(directory):
    all_loaded_articles = []

    for file in os.listdir(directory):
        if not file.endswith(".txt"):
            continue

        path = os.path.join(directory, file)
        
        if os.path.getsize(path) == 0:
            continue
            
        with open(path, 'r', encoding='utf-8') as f:
            raw_content = f.read().strip()

        if not raw_content:
            continue

        entries = raw_content.split(',,') if ',,' in raw_content else raw_content.split('\n\n')
        valid_articles = []

        for entry in entries:
            clean_entry = entry.replace('\n', ' ').strip()
            if not clean_entry:
                continue
                
            # Tries to find target keys dynamically anywhere in the text segment
            title_search = re.search(r"title:\s*(.*?)(?=\s*(summary:|published:|title:|$))", clean_entry, re.IGNORECASE)
            summary_search = re.search(r"summary:\s*(.*?)(?=\s*(published:|title:|summary:|$))", clean_entry, re.IGNORECASE)
            published_search = re.search(r"published:\s*(.*?)(?=\s*(title:|summary:|published:|$))", clean_entry, re.IGNORECASE)
                    
            if title_search or summary_search:
                # Capture whatever data survives, defaulting missing components safely
                article_dict = {
                    "title": title_search.group(1).strip() if title_search else "Unknown Title",
                    "summary": summary_search.group(1).strip() if summary_search else clean_entry,
                    "published": published_search.group(1).strip() if published_search else "Unknown Date"
                }
                valid_articles.append(article_dict)
                all_loaded_articles.append(article_dict)
        
            else:
                print(f"Failed completely on raw line inside: {file}")

        if valid_articles:
            new_filename = os.path.splitext(file)[0] + ".json"
            new_path = os.path.join(directory, new_filename)
            
            with open(new_path, 'w', encoding='utf-8') as json_file:
                json.dump(valid_articles, json_file, indent=4, ensure_ascii=False)
                
            print(f"Fixed & Converted: '{file}' -> '{new_filename}'")

    return all_loaded_articles

if __name__ == "__main__":
    training_news_data('2026-02-23', '2026-03-04')