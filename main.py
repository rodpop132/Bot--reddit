import os
import random
import time
import json
import logging
import re
import asyncio
from typing import Dict, List, Set, Tuple, Optional
from datetime import datetime, timedelta
from dotenv import load_dotenv
import praw
from openai import OpenAI
import requests
from collections import Counter
from filelock import FileLock
import threading
from flask import Flask, render_template_string, jsonify, request
from praw.models import Comment

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

app = Flask(__name__)

# Guardar últimas comunidades descobertas em memória para painel
last_discovered_subreddits = []

@app.route("/")
def index():
    logging.info("[Flask] Endpoint '/' accessed. Bot está rodando!")
    return "Bot está rodando!"

# Função para rodar o Flask
def run_flask():
    port = int(os.environ.get("PORT", 10000))
    logging.info(f"[Flask] Starting Flask server on port {port}")
    app.run(host="0.0.0.0", port=port)

def load_json_with_lock(file_path: str, default):
    """Load JSON file with file locking."""
    logging.debug(f"[IO] Loading JSON from {file_path}")
    with FileLock(f"{file_path}.lock"):
        try:
            with open(file_path, "r") as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            logging.warning(f"File {file_path} not found or corrupted. Returning default.")
            return default

def save_json_with_lock(file_path: str, data) -> None:
    """Save JSON file with file locking."""
    logging.debug(f"[IO] Saving JSON to {file_path}")
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
            logging.info(f"[Discovery] Searching subreddits for keyword: {keyword}")
            for subreddit in reddit.subreddits.search(keyword, limit=5):
                if subreddit.subscribers > 1000 and subreddit.display_name not in SUBREDDITS:
                    new_subreddits.append(subreddit.display_name)
                    logging.info(f"Discovered subreddit: r/{subreddit.display_name} with {subreddit.subscribers} subscribers")
                time.sleep(random.randint(2, 5))  # Anti-ban delay
    except Exception as e:
        logging.error(f"Error discovering subreddits: {e}")
    logging.info(f"[Discovery] New subreddits found: {new_subreddits}")
    return new_subreddits

async def update_subreddits() -> None:
    """Update SUBREDDITS with new communities."""
    logging.info("[Discovery] Updating subreddit list...")
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
    logging.info(f"[Discovery] Updated subreddits: {list(SUBREDDITS.keys())}")

def analyze_popular_keywords(subreddit: str) -> List[str]:
    """Analyze popular keywords in a subreddit's hot posts."""
    logging.info(f"[Keyword] Analyzing popular keywords for r/{subreddit}")
    keywords = Counter()
    try:
        subreddit_obj = reddit.subreddit(subreddit)
        for post in subreddit_obj.hot(limit=50):
            title_words = post.title.lower().split()
            keywords.update(word for word in title_words if len(word) > 3 and word not in {"this", "that", "with", "from"})
            time.sleep(random.randint(1, 3))  # Anti-ban delay
        save_json_with_lock(KEYWORD_FILE, dict(keywords))
        logging.info(f"[Keyword] Top keywords for r/{subreddit}: {keywords.most_common(5)}")
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

# Função aprimorada para engajar em novos subreddits
async def engage_new_subreddits(new_subreddits):
    for sub in new_subreddits:
        logging.info(f"[Discovery] Engajando automaticamente em novo subreddit: r/{sub}")
        # Tenta comentar em posts recentes
        try:
            await engage_on_other_posts(sub, limit=2)
        except Exception as e:
            logging.error(f"[Discovery] Falha ao engajar em r/{sub}: {e}")
        # Tenta postar se permitido
        try:
            post_content = await generate_post_content(sub)
            if post_content["title"] and post_content["body"]:
                await post_to_reddit(sub, post_content["title"], post_content["body"])
        except Exception as e:
            logging.error(f"[Discovery] Falha ao postar em r/{sub}: {e}")

# Função aprimorada de descoberta periódica
async def periodic_subreddit_discovery():
    while True:
        logging.info("[Discovery] Rodando busca periódica de subreddits...")
        new_subreddits = discover_subreddits()
        if new_subreddits:
            global last_discovered_subreddits
            last_discovered_subreddits = new_subreddits[-10:] + last_discovered_subreddits
            last_discovered_subreddits = last_discovered_subreddits[:10]
            await engage_new_subreddits(new_subreddits)
        await update_subreddits()
        await asyncio.sleep(1800)  # 30 minutos

# Aprimorar geração de conteúdo para conversão
async def generate_post_content(subreddit: str) -> dict:
    logging.info(f"[Post] Gerando conteúdo otimizado para conversão para r/{subreddit}")
    try:
        style_data = select_best_prompt_style(subreddit)
        style, selected_prompt, title_style = style_data["style"], style_data["prompt"], style_data["title_style"]
        popular_keywords = await asyncio.to_thread(analyze_popular_keywords, subreddit)
        base_url = f"https://waifuai.chat/?utm_source=reddit&utm_medium=post&utm_campaign=bot_{subreddit}_{int(time.time())}"
        short_url = await asyncio.to_thread(shorten_url, base_url)
        url_variation = random.choice([
            f"Descubra agora: {short_url}",
            f"Clique e conheça: {short_url}",
            f"Experimente grátis: {short_url}",
            f"Veja como funciona: {short_url}",
            f"Acesse: {short_url} e viva a experiência!"
        ])
        cta = random.choice([
            "Não perca essa chance!",
            "Garanta sua experiência exclusiva!",
            "Entre agora e surpreenda-se!",
            "O futuro do chat AI está aqui!",
            "Venha fazer parte dessa revolução!"
        ])
        response = openai.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": selected_prompt},
                {"role": "user", "content": f"Crie um post para r/{subreddit} no estilo '{style}' com título '{title_style}'. Inclua palavras-chave como {', '.join(popular_keywords)} e mencione '{url_variation}' no final. Use uma chamada para ação forte como '{cta}'."}
            ]
        )
        if not response.choices or not response.choices[0].message.content:
            raise ValueError("Empty or invalid OpenAI response")
        title, body = parse_openai_response(response.choices[0].message.content)
        logging.info(f"[Post] Post otimizado gerado para r/{subreddit}: Title='{title[:40]}...' Body length={len(body)}")
        return {"title": title, "body": body, "style": style, "prompt": selected_prompt, "title_style": title_style, "url": short_url}
    except Exception as e:
        logging.error(f"[Conversao] Erro ao gerar post para r/{subreddit}: {e}")
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
        logging.info(f"[Comment] Generated comment response for r/{subreddit}")
        return response.choices[0].message.content
    except Exception as e:
        logging.error(f"Error generating comment response: {e}")
        return f"Thanks for the comment! Check out {short_url} if you're curious about Waifu AI! 😊"

async def post_to_reddit(subreddit: str, title: str, body: str) -> str:
    """Post content to a subreddit."""
    try:
        logging.info(f"[Reddit] Posting to r/{subreddit}: {title[:40]}...")
        subreddit_obj = reddit.subreddit(subreddit)
        submission = subreddit_obj.submit(title=title, selftext=body)
        logging.info(f"[Reddit] Posted to r/{subreddit}: {submission.id}")
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
    logging.info(f"[Comment] Checking comments for post {post_id} in r/{subreddit}")
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
            # Ignorar objetos MoreComments
            if not isinstance(comment, Comment):
                continue
            if comment.author and comment.author.name != REDDIT_USERNAME and comment.id not in replied_comments and count < MAX_COMMENTS_PER_HOUR:
                logging.info(f"[Comment] Responding to comment {comment.id} on post {post_id}")
                response = await generate_comment_response(comment.body, subreddit)
                comment.reply(response)
                replied_comments.add(comment.id)
                save_replied_comments(replied_comments)
                count += 1
                save_comment_counter(count, last_hour)
                logging.info(f"[Comment] Replied to comment {comment.id} on post {post_id}: {response}")
                await asyncio.sleep(random.randint(30, 120))
    except Exception as e:
        logging.error(f"Error checking comments on post {post_id}: {e}")

async def engage_on_other_posts(subreddit: str, limit: int = 3) -> None:
    """Engage with recent posts."""
    logging.info(f"[Engage] Engaging on other posts in r/{subreddit}")
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
                logging.info(f"[Engage] Commenting on post {post.id} in r/{subreddit}")
                response = await generate_comment_response(post.title + "\n" + (post.selftext or ""), subreddit)
                post.reply(response)
                replied_comments.add(post.id)
                save_replied_comments(replied_comments)
                count += 1
                save_comment_counter(count, last_hour)
                logging.info(f"[Engage] Engaged with post {post.id} in r/{subreddit}: {response}")
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
    logging.info("[Task] Starting daily posting task...")
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
            logging.info(f"[Task] Daily post created with ID: {post_id} in r/{subreddit}")
            # Schedule performance tracking and comment checking
            asyncio.create_task(periodic_check_comments_and_performance(post_id, subreddit))
    else:
        logging.warning(f"[Task] Failed to generate post content for r/{subreddit}")

async def periodic_check_comments_and_performance(post_id: str, subreddit: str):
    """Periodically check comments and track performance for a post."""
    logging.info(f"[Task] Starting periodic check for post {post_id} in r/{subreddit}")
    performance_data = load_performance_data()
    for i in range(24):  # Check every hour for 24 hours
        logging.info(f"[Task] Hourly check {i+1}/24 for post {post_id}")
        await check_comments_and_respond(post_id, subreddit)
        metrics = await track_post_performance(post_id, subreddit)
        if metrics:
            performance_data["subreddits"][subreddit]["posts"][post_id]["metrics"].append(metrics)
            save_json_with_lock(PERFORMANCE_FILE, performance_data)
        await asyncio.sleep(3600)  # 1 hour
    await optimize_posting_schedule(subreddit, post_id, performance_data)
    logging.info(f"[Task] Finished periodic check for post {post_id} in r/{subreddit}")

async def weekly_subreddit_discovery_task() -> None:
    """Weekly subreddit discovery task."""
    logging.info("[Task] Starting weekly subreddit discovery task...")
    await update_subreddits()
    logging.info("[Task] Completed subreddit discovery task")

async def daily_engagement_task() -> None:
    """Daily engagement task."""
    logging.info("[Task] Starting daily engagement task...")
    performance_data = load_performance_data()
    safe_subreddits = [sub for sub, data in performance_data["subreddits"].items() if data["safety_score"] >= MIN_SAFETY_SCORE]
    if not safe_subreddits:
        logging.warning("No safe subreddits for engagement.")
        return
    subreddit = random.choice(safe_subreddits)
    await engage_on_other_posts(subreddit)
    logging.info(f"[Task] Completed engagement task for r/{subreddit}")

async def scheduler_loop():
    """Main scheduler loop to run tasks at specific times."""
    global scheduler_alive
    scheduler_alive = True
    logging.info("[Scheduler] Starting scheduler loop...")
    while True:
        now = datetime.now()
        # Weekly subreddit discovery (Monday 10:00)
        if now.weekday() == 0 and now.hour == 10 and now.minute == 0:
            logging.info("[Scheduler] Running weekly subreddit discovery task...")
            await weekly_subreddit_discovery_task()
            await asyncio.sleep(60)
        # Daily posting (12:00)
        if now.hour == 12 and now.minute == 0:
            logging.info("[Scheduler] Running daily posting task...")
            await daily_posting_task()
            await asyncio.sleep(60)
        # Daily engagement (16:00)
        if now.hour == 16 and now.minute == 0:
            logging.info("[Scheduler] Running daily engagement task...")
            await daily_engagement_task()
            await asyncio.sleep(60)
        await asyncio.sleep(30)

async def main() -> None:
    """Main function to run the bot."""
    logging.info("[Main] Starting Reddit bot...")
    await asyncio.gather(
        scheduler_loop(),
        periodic_subreddit_discovery()
    )

@app.route("/health")
def health():
    # Checa variáveis de ambiente
    required_env_vars = [
        "REDDIT_CLIENT_ID", "REDDIT_CLIENT_SECRET", "REDDIT_USER_AGENT",
        "REDDIT_USERNAME", "REDDIT_PASSWORD", "OPENAI_API_KEY"
    ]
    missing = [v for v in required_env_vars if not os.getenv(v)]
    env_ok = len(missing) == 0

    # Checa se o log está sendo atualizado (modificado nos últimos 5 minutos)
    log_path = "reddit_bot.log"
    try:
        mtime = os.path.getmtime(log_path)
        from time import time
        log_recent = (time() - mtime) < 300  # 5 minutos
    except Exception:
        log_recent = False

    # Checa se o scheduler está rodando (simples: variável global setada no loop)
    global scheduler_alive
    alive = scheduler_alive if 'scheduler_alive' in globals() else False

    status = env_ok and log_recent and alive
    return jsonify({
        "status": status,
        "env_ok": env_ok,
        "missing_env": missing,
        "log_recent": log_recent,
        "scheduler_alive": alive
    })

@app.route("/dashboard")
def dashboard():
    # Healthcheck para status visual
    import requests as pyrequests
    try:
        health = pyrequests.get("http://127.0.0.1:10000/health", timeout=2).json()
    except Exception:
        health = {"status": False, "env_ok": False, "missing_env": [], "log_recent": False, "scheduler_alive": False}
    if health["status"]:
        status_html = "<div class='alert alert-success'>✅ Bot operando normalmente!</div>"
    else:
        msg = []
        if not health["env_ok"]:
            msg.append(f"Variáveis de ambiente faltando: {', '.join(health['missing_env'])}")
        if not health["log_recent"]:
            msg.append("Arquivo de log não está sendo atualizado!")
        if not health["scheduler_alive"]:
            msg.append("Scheduler principal não está rodando!")
        status_html = f"<div class='alert alert-danger'>❌ Problemas detectados:<br>{'<br>'.join(msg)}</div>"

    # Lê os últimos 100 logs
    log_path = "reddit_bot.log"
    try:
        with open(log_path, "r", encoding="utf-8") as f:
            lines = f.readlines()[-100:]
    except Exception:
        lines = ["Log file not found or unreadable."]
    logs = "".join(f"<div class='logline'>{line}</div>" for line in lines)

    # Métricas básicas e engajamento por subreddit
    try:
        performance = load_performance_data()
        total_posts = sum(len(sub["posts"]) for sub in performance["subreddits"].values())
        total_subreddits = len(performance["subreddits"])
        total_comments = 0
        engajamento = []
        for subname, sub in performance["subreddits"].items():
            post_count = len(sub["posts"])
            comment_count = sum(sum(m.get("comments", 0) for m in post.get("metrics", [])) for post in sub["posts"].values())
            engajamento.append((subname, post_count, comment_count))
            for post in sub["posts"].values():
                total_comments += sum(m.get("comments", 0) for m in post.get("metrics", []))
    except Exception:
        total_posts = total_subreddits = total_comments = 0
        engajamento = []

    # Status das tarefas (simples: última execução = hora atual, próxima = estimada)
    now = datetime.now()
    status = [
        {"name": "Descoberta periódica de subreddits", "last": now.strftime('%Y-%m-%d %H:%M'), "next": (now + timedelta(minutes=30)).strftime('%Y-%m-%d %H:%M')},
        {"name": "Postagem diária", "last": now.strftime('%Y-%m-%d %H:%M'), "next": (now.replace(hour=12, minute=0) if now.hour < 12 else (now + timedelta(days=1)).replace(hour=12, minute=0)).strftime('%Y-%m-%d %H:%M')},
        {"name": "Engajamento diário", "last": now.strftime('%Y-%m-%d %H:%M'), "next": (now.replace(hour=16, minute=0) if now.hour < 16 else (now + timedelta(days=1)).replace(hour=16, minute=0)).strftime('%Y-%m-%d %H:%M')},
    ]
    status_rows = ''.join(
        f"<tr><td>{t['name']}</td><td>{t['last']}</td><td>{t['next']}</td></tr>" for t in status
    )
    engajamento_rows = ''.join(
        f"<tr><td>{sub}</td><td>{posts}</td><td>{comments}</td></tr>" for sub, posts, comments in engajamento
    )
    discovered_html = ''.join(f'<span class="badge bg-info text-dark me-1">r/{sub}</span>' for sub in last_discovered_subreddits)

    html = f"""
    <!DOCTYPE html>
    <html lang='en'>
    <head>
        <meta charset='utf-8'>
        <meta name='viewport' content='width=device-width, initial-scale=1'>
        <title>Reddit Bot Dashboard</title>
        <link href='https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css' rel='stylesheet'>
        <style>
            body {{ background: #181a1b; color: #eee; }}
            .logbox {{ background: #23272b; border-radius: 8px; padding: 1em; height: 350px; overflow-y: scroll; font-family: monospace; font-size: 0.95em; }}
            .logline {{ white-space: pre; }}
            .metricbox {{ background: #23272b; border-radius: 8px; padding: 1em; margin-bottom: 1em; }}
            .table-dark th, .table-dark td {{ color: #eee; }}
        </style>
    </head>
    <body>
    <div class='container py-4'>
        <h1 class='mb-4'>🤖 Reddit Bot Dashboard</h1>
        {status_html}
        <div class='row mb-4'>
            <div class='col-md-4'>
                <div class='metricbox'>
                    <h5>Posts feitos</h5>
                    <div class='display-6'>{total_posts}</div>
                </div>
            </div>
            <div class='col-md-4'>
                <div class='metricbox'>
                    <h5>Comentários respondidos</h5>
                    <div class='display-6'>{total_comments}</div>
                </div>
            </div>
            <div class='col-md-4'>
                <div class='metricbox'>
                    <h5>Subreddits ativos</h5>
                    <div class='display-6'>{total_subreddits}</div>
                </div>
            </div>
        </div>
        <h4>Últimas comunidades descobertas</h4>
        <div class='mb-3'>{discovered_html if discovered_html else '<span class="text-muted">Nenhuma descoberta recente.</span>'}</div>
        <h4>Engajamento por Subreddit</h4>
        <table class='table table-dark table-striped mb-4'>
            <thead><tr><th>Subreddit</th><th>Posts</th><th>Comentários</th></tr></thead>
            <tbody>
                {engajamento_rows}
            </tbody>
        </table>
        <h4>Status das Tarefas</h4>
        <table class='table table-dark table-striped mb-4'>
            <thead><tr><th>Tarefa</th><th>Última Execução</th><th>Próxima Execução</th></tr></thead>
            <tbody>
                {status_rows}
            </tbody>
        </table>
        <h4>Logs Recentes</h4>
        <div class='logbox' id='logbox'>{logs}</div>
        <button class='btn btn-secondary mt-2' onclick='refreshLogs()'>Atualizar Logs</button>
    </div>
    <script>
    function refreshLogs() {{
        fetch('/logs')
            .then(r => r.json())
            .then(data => {{
                document.getElementById('logbox').innerHTML = data.logs;
            }})
            .catch(err => {{
                document.getElementById('logbox').innerHTML = '<div class="text-danger">Erro ao carregar logs.</div>';
            }});
    }}
    setInterval(refreshLogs, 5000);
    </script>
    </body>
    </html>
    """
    return html

@app.route("/logs")
def logs_api():
    log_path = "reddit_bot.log"
    try:
        with open(log_path, "r", encoding="utf-8") as f:
            lines = f.readlines()[-100:]
        if not lines:
            lines = ["Nenhum log disponível."]
    except Exception:
        lines = ["Log file not found or unreadable."]
    logs = "".join(f"<div class='logline'>{line}</div>" for line in lines)
    return jsonify({"logs": logs})

if __name__ == "__main__":
    logging.info("[Main] Starting Flask thread...")
    flask_thread = threading.Thread(target=run_flask)
    flask_thread.daemon = True
    flask_thread.start()
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logging.info("Bot stopped by user.") 
