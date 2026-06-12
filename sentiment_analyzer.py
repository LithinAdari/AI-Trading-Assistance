import yfinance as yf
import feedparser
import re
import requests
from datetime import datetime

# Financial sentiment lexicons
POSITIVE_WORDS = {
    "surge", "jump", "rise", "soar", "gain", "profit", "growth", "beat", "bullish",
    "buy", "strong", "positive", "expansion", "win", "contract", "partnership",
    "upgrade", "outperform", "record", "higher", "approve", "green", "clean",
    "order", "secures", "commission", "commissioned", "jv", "boost", "leads",
    "expansion", "solar", "wind", "electric", "ev", "charging", "gigafactory"
}

NEGATIVE_WORDS = {
    "drop", "fall", "slump", "plunge", "decline", "loss", "miss", "bearish",
    "sell", "weak", "negative", "contraction", "lose", "lawsuit", "investigation",
    "downgrade", "underperform", "debt", "deficit", "shrink", "risk", "crisis",
    "penalty", "fine", "slash", "slashes", "probe", "delay", "failed", "debt-ridden"
}

def clean_text(text: str) -> str:
    """Lowercase text and strip punctuation."""
    if not text:
        return ""
    text = text.lower()
    return re.sub(r"[^\w\s]", " ", text)

def score_text_sentiment(text: str) -> float:
    """
    Score the sentiment of a text block between -1.0 (very negative) and 1.0 (very positive).
    Uses a simple lexicon matching algorithm.
    """
    cleaned = clean_text(text)
    words = cleaned.split()
    
    pos_count = sum(1 for w in words if w in POSITIVE_WORDS)
    neg_count = sum(1 for w in words if w in NEGATIVE_WORDS)
    
    # Check for multi-word phrases
    phrases_pos = [
        "earnings beat", "revenue increase", "net profit", "capacity expansion", 
        "solar plant", "green energy", "wind farm", "electric vehicle", "charging station"
    ]
    phrases_neg = [
        "earnings miss", "revenue drop", "net loss", "order cancellation", 
        "project delay", "power cut", "regulatory hurdle"
    ]
    
    for phrase in phrases_pos:
        if phrase in cleaned:
            pos_count += 2
    for phrase in phrases_neg:
        if phrase in cleaned:
            neg_count += 2
            
    total = pos_count + neg_count
    if total == 0:
        return 0.0
    
    return (pos_count - neg_count) / total

def get_ticker_news_sentiment(ticker: str) -> dict:
    """
    Fetch news and analyze sentiment for a stock ticker.
    Attempts to use yfinance.Ticker(ticker).news first,
    then falls back to Yahoo RSS feed, and finally returns a neutral score.
    """
    headlines_data = []
    total_sentiment = 0.0
    
    # Method 1: yfinance News API
    try:
        yf_ticker = yf.Ticker(ticker)
        raw_news = yf_ticker.news
        if raw_news and isinstance(raw_news, list):
            for article in raw_news:
                content = article.get("content", {})
                if not content:
                    continue
                title = content.get("title", "")
                summary = content.get("summary", content.get("description", ""))
                pub_date_str = content.get("pubDate", "")
                link = content.get("canonicalUrl", content.get("clickThroughUrl", ""))
                
                if not title:
                    continue
                    
                full_text = f"{title} {summary}"
                score = score_text_sentiment(full_text)
                
                # Format date nicely: E.g., '2026-02-04T14:19:42Z' -> '2026-02-04 14:19'
                date_formatted = pub_date_str
                try:
                    dt = datetime.strptime(pub_date_str[:16], "%Y-%m-%dT%H:%M")
                    date_formatted = dt.strftime("%Y-%m-%d %H:%M")
                except ValueError:
                    pass
                    
                headlines_data.append({
                    "title": title,
                    "summary": summary,
                    "published": date_formatted,
                    "score": round(score, 3),
                    "link": link
                })
                total_sentiment += score
    except Exception as e:
        # Fallback to RSS or proceed to Method 2
        pass
        
    # Method 2: RSS Feed fallback if yfinance news is empty
    if not headlines_data:
        try:
            # RSS feed URL (stripping suffix for RSS search query if needed, or query direct)
            # For RSS, US tickers work fine, but Indian suffix may need to be included
            rss_url = f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker}"
            headers = {'User-Agent': 'Mozilla/5.0'}
            r = requests.get(rss_url, headers=headers, timeout=5)
            if r.status_code == 200:
                feed = feedparser.parse(r.content)
                for entry in feed.entries:
                    title = entry.get("title", "")
                    summary = entry.get("description", entry.get("summary", ""))
                    pub_date_str = entry.get("published", entry.get("pubDate", ""))
                    link = entry.get("link", "")
                    
                    full_text = f"{title} {summary}"
                    score = score_text_sentiment(full_text)
                    
                    headlines_data.append({
                        "title": title,
                        "summary": summary,
                        "published": pub_date_str,
                        "score": round(score, 3),
                        "link": link
                    })
                    total_sentiment += score
        except Exception:
            pass

    num_stories = len(headlines_data)
    avg_score = (total_sentiment / num_stories) if num_stories > 0 else 0.0
    
    return {
        "ticker": ticker,
        "average_sentiment": round(avg_score, 3),
        "headlines": headlines_data[:10]  # Return top 10 articles
    }

if __name__ == "__main__":
    print("Testing Updated yfinance news Sentiment scraper for TATAPOWER.NS:")
    res = get_ticker_news_sentiment("TATAPOWER.NS")
    print(f"Average sentiment: {res['average_sentiment']}")
    print(f"Fetched {len(res['headlines'])} articles.")
    for idx, story in enumerate(res['headlines'][:3]):
        print(f"{idx+1}. [{story['published']}] (Score: {story['score']}) {story['title']}")
