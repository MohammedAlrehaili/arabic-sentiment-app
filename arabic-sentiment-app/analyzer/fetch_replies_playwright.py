from playwright.sync_api import sync_playwright
from datetime import datetime
import pandas as pd

def fetch_replies(tweet_url, max_replies=50):
    data = []

    with sync_playwright() as p:

        browser = p.chromium.launch_persistent_context(
            user_data_dir="C:\\Users\\Kvendy\\AppData\\Local\\Google\\Chrome\\User Data",
            headless=False,
            args=["--profile-directory=Default"]
        )
        page = browser.pages[0] if browser.pages else browser.new_page()

        print(f"📡 Visiting: {tweet_url}")
        page.goto(tweet_url, timeout=60000)

        page.wait_for_timeout(5000)

        retries = 0
        last_len = 0
        while len(data) < max_replies and retries < 10:
            page.keyboard.press("PageDown")
            page.wait_for_timeout(1000)

            replies = page.query_selector_all("article")

            for r in replies:
                try:
                    username = r.query_selector("div[dir='ltr'] span").inner_text()
                    text = r.query_selector("div[data-testid='tweetText']").inner_text()
                    timestamp = r.query_selector("time").get_attribute("datetime")
                    date = datetime.fromisoformat(timestamp.replace("Z", "+00:00")).strftime("%Y-%m-%d")

                    reply_data = {
                        "user_id": username,
                        "text": text,
                        "date": date,
                        "platform": "Twitter"
                    }

                    if reply_data not in data:
                        data.append(reply_data)
                except:
                    continue

            if len(data) == last_len:
                retries += 1
            else:
                retries = 0
            last_len = len(data)

        browser.close()
        return data


if __name__ == "__main__":
    tweet_url = "https://x.com/bricksdept/status/1913545057925227003"
    replies = fetch_replies(tweet_url, max_replies=50)

    print(f"\n✅ Total replies scraped: {len(replies)}")

    df = pd.DataFrame(replies)
    df.to_csv("media/twitter_comments.csv", index=False, encoding="utf-8-sig")
    print("📁 Saved to media/twitter_comments.csv")
