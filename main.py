import os
import random
import time
import json
import logging
import re
import asyncio
from typing import Dict, List, Set, Tuple, Optional
from datetime import datetime
from dotenv import load_dotenv
import praw
from openai import OpenAI
import requests
from collections import Counter
from filelock import FileLock

# Configure logging (file + console output)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("reddit_bot.log"),
        logging.StreamHandler()
    ]
)

# Load environment variables
load_dotenv()

# Reddit API credentials
REDDIT_CLIENT_ID = os.getenv("REDDIT_CLIENT_ID")
REDDIT_CLIENT_SECRET = os.getenv("REDDIT_CLIENT_SECRET")
REDDIT_USER_AGENT = os.getenv("REDDIT_USER_AGENT")
REDDIT_USERNAME = os.getenv("REDDIT_USERNAME")
REDDIT_PASSWORD = os.getenv("REDDIT_PASSWORD")

# OpenRouter API key and model
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_BASE_URL = "https://api.openrouter.ai/v1"
MODEL_NAME = os.getenv("OPENROUTER_MODEL", "xai/grok-3")
TINYURL_API_KEY = os.getenv("TINYURL_API_KEY")

# Validate environment variables
required_env_vars = {
    "REDDIT_CLIENT_ID": REDDIT_CLIENT_ID,
    "REDDIT_CLIENT_SECRET": REDDIT_CLIENT_SECRET,
    "REDDIT_USER_AGENT": REDDIT_USER_AGENT,
    "REDDIT_USERNAME": REDDIT_USERNAME,
    "REDDIT_PASSWORD": REDDIT_PASSWORD,
    "OPENAI_API_KEY": OPENAI_API_KEY
}
missing_vars = [key for key, value in required_env_vars.items() if not value]
if missing_vars:
    logging.error(f"Missing environment variables: {', '.join(missing_vars)}")
    raise EnvironmentError(f"Missing environment variables: {', '.join(missing_vars)}")

# System prompt variants for A/B testing
PROMPT_VARIANTS = [
    """
    You are a digital marketing expert specializing in viral marketing for otaku and tech products.
    Promote Waifu AI Chat with creativity, emotion, and anime culture connections, encouraging clicks to the site naturally.
    Use an emotional and engaging tone, as if you're a passionate fan.
    """,
    """
    You are a mysterious otaku marketer, expert at sparking curiosity.
    Promote Waifu AI Chat with enigmatic phrases and subtle anime references, encouraging site visits intriguingly.
    """,
    """
    You are a funny and relaxed copywriter, passionate about AI and otaku culture.
    Promote Waifu AI Chat with humor, anime memes, and lighthearted stories, subtly inviting users to the site.
    """
]

# Title styles for A/B testing
TITLE_STYLES = ["curious", "funny", "sentimental"]

# Subreddits with styles, safety scores, and best posting hours
SUBREDDITS = {
    "waifus": {"styles": ["romantic", "emotional"], "safety_score": 1.0, "best_hour": "12:00"},
    "OtakuCulture": {"styles": ["funny", "anime_refs"], "safety_score": 1.0, "best_hour": "12:00"},
    "AIgirlfriend": {"styles": ["curious", "tech"], "safety_score": 1.0, "best_hour": "12:00"},
    "ChatGPTGirlfriend": {"styles": ["tech", "emotional"], "safety_score": 1.0, "best_hour": "12:00"},
    "anime": {"styles": ["funny", "romantic"], "safety_score": 1.0, "best_hour": "12:00"}
}

# Files for persistence
PERFORMANCE_FILE = "performance_metrics.json"
REPLIED_COMMENTS_FILE = "replied_comments.json"
KEYWORD_FILE = "popular_keywords.json"
COUNTER_FILE = "comment_counter.json"

# Anti-ban settings
MAX_COMMENTS_PER_HOUR = 3
MIN_SAFETY_SCORE = 0.5

# URL variations for anti-ban
URL_VARIATIONS = [
    "Check it out: https://waifuai.chat",
    "If you're curious, see: https://waifuai.chat",
    "I use https://waifuai.chat 😅",
    "More info here: https://waifuai.chat"
]

# Keywords for subreddit discovery
SEARCH_KEYWORDS = ["waifu", "AI girlfriend", "anime chat", "otaku love", "virtual girlfriend"]

# Initialize Reddit client
reddit = praw.Reddit(
    client_id=REDDIT_CLIENT_ID,
    client_secret=REDDIT_CLIENT_SECRET,
    user_agent=REDDIT_USER_AGENT,
    username=REDDIT_USERNAME,
    password=REDDIT_PASSWORD
)

# Initialize OpenRouter client
openai = OpenAI(
    api_key=OPENAI_API_KEY,
    base_url=OPENAI_BASE_URL
)

def load_json_with_lock(file_path: str, default):
    """Load JSON file with file locking."""
    with FileLock(f"{file_path}.lock"):
        try:
            with open(file_path, "r") as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            logging.warning(f"File {file_path} not found or corrupted. Returning default.")
            return default

def save_json_with_lock(file_path: str, data) -> None:
    """Save JSON file with file locking."""
    with FileLock(f"{file_path}.lock"):
        try:
            # Convert set to list for JSON serialization
            if isinstance(data, set):
                data = list(data)
            with open(file_path, "w") as f:
                json.dump(data, f, indent=4)
        except Exception as e:
            logging.error(f"Error saving {file_path}: {e}")

def load_performance_data() -> Dict:
    """Load performance data."""
    default = {"subreddits": {sub: {"best_styles": data["styles"], "posts": {}, "safety_score": data["safety_score"], "best_hour": data["best_hour"]} for sub, data in SUBREDDITS.items()}}
    return load_json_with_lock(PERFORMANCE_FILE, default)

def load_replied_comments() -> Set[str]:
    """Load replied comment IDs as set."""
    data = load_json_with_lock(REPLIED_COMMENTS_FILE, [])
    return set(data) if isinstance(data, list) else set()

def save_replied_comments(replied_comments: Set[str]) -> None:
    """Save replied comment IDs as list."""
    save_json_with_lock(REPLIED_COMMENTS_FILE, list(replied_comments))

def load_popular_keywords() -> Dict[str, int]:
    """Load popular keywords."""
    return load_json_with_lock(KEYWORD_FILE, {})

def load_comment_counter() -> Dict[str, int]:
    """Load comment counter and last hour."""
    return load_json_with_lock(COUNTER_FILE, {"count": 0, "last_hour": int(time.time() // 3600)})

def save_comment_counter(count: int, last_hour: int) -> None:
    """Save comment counter and last hour."""
    save_json_with_lock(COUNTER_FILE, {"count": count, "last_hour": last_hour})

def shorten_url(long_url: str) -> str:
    """Shorten a URL using TinyURL API."""
    if not TINYURL_API_KEY:
        logging.warning("TinyURL API key not set. Using original URL.")
        return long_url
    try:
        response = requests.post(
            "https://api.tinyurl.com/create",
            headers={"Authorization": f"Bearer {TINYURL_API_KEY}"},
            json={"url": long_url}
        )
        response.raise_for_status()
        return response.json().get("data", {}).get("tiny_url", long_url)
    except Exception as e:
        logging.error(f"Error shortening URL: {e}")
        return long_url

def discover_subreddits() -> List[str]:
    """Search for new subreddits based on keywords."""
    new_subreddits = []
    try:
        for keyword in SEARCH_KEYWORDS:
            for subreddit in reddit.subreddits.search(keyword, limit=5):
                if subreddit.subscribers > 1000 and subreddit.display_name not in SUBREDDITS:
                    new_subreddits.append(subreddit.display_name)
                    logging.info(f"Discovered subreddit: r/{subreddit.display_name} with {subreddit.subscribers} subscribers")
                time.sleep(random.randint(2, 5))  # Anti-ban delay
    except Exception as e:
        logging.error(f"Error discovering subreddits: {e}")
    return new_subreddits

async def update_subreddits() -> None:
    """Update SUBREDDITS with new communities."""
    performance_data = load_performance_data()
    new_subreddits = discover_subreddits()
    for sub in new_subreddits:
        if sub not in performance_data["subreddits"]:
            performance_data["subreddits"][sub] = {
                "best_styles": ["emotional", "curious"],
                "posts": {},
                "safety_score": 1.0,
                "best_hour": "12:00"
            }
            SUBREDDITS[sub] = {"styles": ["emotional", "curious"], "safety_score": 1.0, "best_hour": "12:00"}
    save_json_with_lock(PERFORMANCE_FILE, performance_data)
    logging.info(f"Updated subreddits: {list(SUBREDDITS.keys())}")

def analyze_popular_keywords(subreddit: str) -> List[str]:
    """Analyze popular keywords in a subreddit's hot posts."""
    keywords = Counter()
    try:
        subreddit_obj = reddit.subreddit(subreddit)
        for post in subreddit_obj.hot(limit=50):
            title_words = post.title.lower().split()
            keywords.update(word for word in title_words if len(word) > 3 and word not in {"this", "that", "with", "from"})
            time.sleep(random.randint(1, 3))  # Anti-ban delay
        save_json_with_lock(KEYWORD_FILE, dict(keywords))
        return [word for word, _ in keywords.most_common(5)]
    except Exception as e:
        logging.error(f"Error analyzing keywords for r/{subreddit}: {e}")
        return []

def get_style_for_subreddit(subreddit: str) -> str:
    """Get the preferred style for a subreddit."""
    data = load_performance_data()
    subreddit_data = data.get("subreddits", {}).get(subreddit, {"best_styles": SUBREDDITS.get(subreddit, {"styles": ["emotional"]})["styles"]})
    return random.choice(subreddit_data["best_styles"])

def select_best_prompt_style(subreddit: str) -> Dict[str, str]:
    """Select the best-performing style, prompt, and title style."""
    data = load_performance_data()
    posts = data.get("subreddits", {}).get(subreddit, {}).get("posts", {})
    if not posts:
        return {"style": get_style_for_subreddit(subreddit), "prompt": random.choice(PROMPT_VARIANTS), "title_style": random.choice(TITLE_STYLES)}
    scored_posts = []
    for post_id, post_data in posts.items():
        metrics = post_data.get("metrics", [])
        score = sum(m.get("upvotes", 0) * 1 + m.get("comments", 0) * 2 for m in metrics)
        scored_posts.append((score, post_data))
    if scored_posts:
        best_post = max(scored_posts, key=lambda x: x[0])[1]
        return {
            "style": best_post.get("style", "emotional"),
            "prompt": best_post.get("prompt", random.choice(PROMPT_VARIANTS)),
            "title_style": best_post.get("title_style", random.choice(TITLE_STYLES))
        }
    return {"style": get_style_for_subreddit(subreddit), "prompt": random.choice(PROMPT_VARIANTS), "title_style": random.choice(TITLE_STYLES)}

def parse_openai_response(content: str) -> Tuple[str, str]:
    """Parse OpenAI response using regex."""
    title_match = re.search(r"\*\*Título:\*\*(.*?)(?=\n|$)", content, re.DOTALL)
    body_match = re.search(r"\*\*Texto:\*\*(.*?)(?=\n|$)", content, re.DOTALL)
    title = title_match.group(1).strip() if title_match else ""
    body = body_match.group(1).strip() if body_match else ""
    return title, body

async def generate_post_content(subreddit: str) -> Dict[str, str]:
    """Generate a Reddit post tailored to subreddit."""
    try:
        style_data = select_best_prompt_style(subreddit)
        style, selected_prompt, title_style = style_data["style"], style_data["prompt"], style_data["title_style"]
        popular_keywords = await asyncio.to_thread(analyze_popular_keywords, subreddit)
        base_url = f"https://waifuai.chat/?utm_source=reddit&utm_medium=post&utm_campaign=bot_{subreddit}_{int(time.time())}"
        short_url = await asyncio.to_thread(shorten_url, base_url)
        url_variation = random.choice(URL_VARIATIONS).replace("https://waifuai.chat", short_url)
        response = openai.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": selected_prompt},
                {"role": "user", "content": f"Create a post for r/{subreddit} in '{style}' style with a '{title_style}' title. Include keywords like {', '.join(popular_keywords)} and mention '{url_variation}' at the end."}
            ]
        )
        if not response.choices or not response.choices[0].message.content:
            raise ValueError("Empty or invalid OpenAI response")
        title, body = parse_openai_response(response.choices[0].message.content)
        return {"title": title, "body": body, "style": style, "prompt": selected_prompt, "title_style": title_style, "url": short_url}
    except Exception as e:
        logging.error(f"Error generating post for r/{subreddit}: {e}")
        return {"title": "", "body": "", "style": "", "prompt": "", "title_style": "", "url": ""}

async def generate_comment_response(context: str, subreddit: str) -> str:
    """Generate a comment response."""
    short_url = "https://waifuai.chat"
    try:
        style = get_style_for_subreddit(subreddit)
        base_url = f"https://waifuai.chat/?utm_source=reddit&utm_medium=comment&utm_campaign=bot_{subreddit}_{int(time.time())}"
        short_url = await asyncio.to_thread(shorten_url, base_url)
        url_variation = random.choice(URL_VARIATIONS).replace("https://waifuai.chat", short_url)
        response = openai.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": random.choice(PROMPT_VARIANTS)},
                {"role": "user", "content": f"Respond to this comment or post in '{style}' style, empathetically and casually, subtly promoting Waifu AI Chat with '{url_variation}': '{context}'"}
            ]
        )
        if not response.choices or not response.choices[0].message.content:
            raise ValueError("Empty or invalid OpenAI response")
        return response.choices[0].message.content
    except Exception as e:
        logging.error(f"Error generating comment response: {e}")
        return f"Thanks for the comment! Check out {short_url} if you're curious about Waifu AI! 😊"

async def post_to_reddit(subreddit: str, title: str, body: str) -> str:
    """Post content to a subreddit."""
    try:
        subreddit_obj = reddit.subreddit(subreddit)
        submission = subreddit_obj.submit(title=title, selftext=body)
        logging.info(f"Posted to r/{subreddit}: {title}")
        await asyncio.sleep(random.randint(30, 120))  # Anti-ban delay
        return submission.id
    except Exception as e:
        logging.error(f"Error posting to r/{subreddit}: {e}")
        return ""

async def is_post_removed(post_id: str) -> bool:
    """Check if a post is removed."""
    try:
        submission = reddit.submission(id=post_id)
        removed = bool(submission.removed_by_category or submission.banned_by)
        if removed:
            logging.warning(f"Post {post_id} was removed by Reddit/moderators")
        return removed
    except Exception as e:
        logging.error(f"Error checking post status for {post_id}: {e}")
        return True

async def update_safety_score(subreddit: str, post_id: str, performance_data: Dict) -> None:
    """Update subreddit safety score."""
    if await is_post_removed(post_id):
        performance_data["subreddits"][subreddit]["safety_score"] = max(
            0.0, performance_data["subreddits"][subreddit]["safety_score"] - 0.2
        )
        logging.info(f"Updated safety score for r/{subreddit}: {performance_data['subreddits'][subreddit]['safety_score']}")
        save_json_with_lock(PERFORMANCE_FILE, performance_data)

async def optimize_posting_schedule(subreddit: str, post_id: str, performance_data: Dict) -> None:
    """Update best posting hour."""
    metrics = performance_data["subreddits"][subreddit]["posts"][post_id]["metrics"]
    if not metrics:
        return
    best_metric = max(metrics, key=lambda m: m.get("upvotes", 0) * 1 + m.get("comments", 0) * 2)
    post_time = datetime.fromtimestamp(best_metric["timestamp"])
    best_hour = post_time.strftime("%H:%M")
    performance_data["subreddits"][subreddit]["best_hour"] = best_hour
    logging.info(f"Updated best posting hour for r/{subreddit}: {best_hour}")
    save_json_with_lock(PERFORMANCE_FILE, performance_data)

async def check_comments_and_respond(post_id: str, subreddit: str) -> None:
    """Check and respond to comments."""
    counter_data = load_comment_counter()
    count, last_hour = counter_data["count"], counter_data["last_hour"]
    current_hour = int(time.time() // 3600)
    if current_hour != last_hour:
        count, last_hour = 0, current_hour
        save_comment_counter(count, last_hour)
    if count >= MAX_COMMENTS_PER_HOUR:
        logging.info("Max comments per hour reached. Skipping comment responses.")
        return
    replied_comments = load_replied_comments()
    try:
        submission = reddit.submission(id=post_id)
        if await is_post_removed(post_id):
            performance_data = load_performance_data()
            await update_safety_score(subreddit, post_id, performance_data)
            return
        submission.comments.replace_more(limit=0)
        for comment in submission.comments.list():
            if comment.author and comment.author.name != REDDIT_USERNAME and comment.id not in replied_comments and count < MAX_COMMENTS_PER_HOUR:
                response = await generate_comment_response(comment.body, subreddit)
                comment.reply(response)
                replied_comments.add(comment.id)
                save_replied_comments(replied_comments)
                count += 1
                save_comment_counter(count, last_hour)
                logging.info(f"Replied to comment {comment.id} on post {post_id}: {response}")
                await asyncio.sleep(random.randint(30, 120))
    except Exception as e:
        logging.error(f"Error checking comments on post {post_id}: {e}")

async def engage_on_other_posts(subreddit: str, limit: int = 3) -> None:
    """Engage with recent posts."""
    counter_data = load_comment_counter()
    count, last_hour = counter_data["count"], counter_data["last_hour"]
    current_hour = int(time.time() // 3600)
    if current_hour != last_hour:
        count, last_hour = 0, current_hour
        save_comment_counter(count, last_hour)
    if count >= MAX_COMMENTS_PER_HOUR:
        logging.info("Max comments per hour reached. Skipping engagement.")
        return
    replied_comments = load_replied_comments()
    try:
        subreddit_obj = reddit.subreddit(subreddit)
        for post in subreddit_obj.new(limit=limit):
            if post.author and post.author.name != REDDIT_USERNAME and not await is_post_removed(post.id) and post.id not in replied_comments and count < MAX_COMMENTS_PER_HOUR:
                response = await generate_comment_response(post.title + "\n" + (post.selftext or ""), subreddit)
                # Comentar no post criando um novo comentário
                post.reply(response)
                replied_comments.add(post.id)
                save_replied_comments(replied_comments)
                count += 1
                save_comment_counter(count, last_hour)
                logging.info(f"Engaged with post {post.id} in r/{subreddit}: {response}")
                await asyncio.sleep(random.randint(60, 180))
    except Exception as e:
        logging.error(f"Error engaging on posts in r/{subreddit}: {e}")

async def track_post_performance(post_id: str, subreddit: str) -> Dict:
    """Track post performance."""
    try:
        submission = reddit.submission(id=post_id)
        if await is_post_removed(post_id):
            performance_data = load_performance_data()
            await update_safety_score(subreddit, post_id, performance_data)
            return {}
        return {
            "post_id": post_id,
            "upvotes": submission.score,
            "comments": submission.num_comments,  # More accurate than len(comments)
            "timestamp": time.time()
        }
    except Exception as e:
        logging.error(f"Error tracking performance for post {post_id}: {e}")
        return {}

async def daily_posting_task() -> None:
    """Daily posting task."""
    performance_data = load_performance_data()
    safe_subreddits = [sub for sub, data in performance_data["subreddits"].items() if data["safety_score"] >= MIN_SAFETY_SCORE]
    if not safe_subreddits:
        logging.warning("No safe subreddits available. Skipping posting.")
        return
    subreddit = random.choice(safe_subreddits)
    post_content = await generate_post_content(subreddit)
    if post_content["title"] and post_content["body"]:
        post_id = await post_to_reddit(subreddit, post_content["title"], post_content["body"])
        if post_id:
            performance_data["subreddits"][subreddit]["posts"][post_id] = {
                "title": post_content["title"],
                "body": post_content["body"],
                "style": post_content["style"],
                "prompt": post_content["prompt"],
                "title_style": post_content["title_style"],
                "url": post_content["url"],
                "metrics": []
            }
            save_json_with_lock(PERFORMANCE_FILE, performance_data)
            logging.info(f"Daily post created with ID: {post_id} in r/{subreddit}")
            # Schedule performance tracking and comment checking
            asyncio.create_task(periodic_check_comments_and_performance(post_id, subreddit))

async def periodic_check_comments_and_performance(post_id: str, subreddit: str):
    """Periodically check comments and track performance for a post."""
    performance_data = load_performance_data()
    for _ in range(24):  # Check every hour for 24 hours
        await check_comments_and_respond(post_id, subreddit)
        metrics = await track_post_performance(post_id, subreddit)
        if metrics:
            performance_data["subreddits"][subreddit]["posts"][post_id]["metrics"].append(metrics)
            save_json_with_lock(PERFORMANCE_FILE, performance_data)
        await asyncio.sleep(3600)  # 1 hour
    await optimize_posting_schedule(subreddit, post_id, performance_data)

async def weekly_subreddit_discovery_task() -> None:
    """Weekly subreddit discovery task."""
    await update_subreddits()
    logging.info("Completed subreddit discovery task")

async def daily_engagement_task() -> None:
    """Daily engagement task."""
    performance_data = load_performance_data()
    safe_subreddits = [sub for sub, data in performance_data["subreddits"].items() if data["safety_score"] >= MIN_SAFETY_SCORE]
    if not safe_subreddits:
        logging.warning("No safe subreddits for engagement.")
        return
    subreddit = random.choice(safe_subreddits)
    await engage_on_other_posts(subreddit)
    logging.info(f"Completed engagement task for r/{subreddit}")

async def scheduler_loop():
    """Main scheduler loop to run tasks at specific times."""
    while True:
        now = datetime.now()
        # Weekly subreddit discovery (Monday 10:00)
        if now.weekday() == 0 and now.hour == 10 and now.minute == 0:
            await weekly_subreddit_discovery_task()
            await asyncio.sleep(60)
        # Daily posting (12:00)
        if now.hour == 12 and now.minute == 0:
            await daily_posting_task()
            await asyncio.sleep(60)
        # Daily engagement (16:00)
        if now.hour == 16 and now.minute == 0:
            await daily_engagement_task()
            await asyncio.sleep(60)
        await asyncio.sleep(30)

async def main() -> None:
    """Main function to run the bot."""
    logging.info("Starting Reddit bot...")
    await scheduler_loop()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logging.info("Bot stopped by user.") 