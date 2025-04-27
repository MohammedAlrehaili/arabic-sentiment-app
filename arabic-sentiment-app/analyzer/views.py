import pandas as pd
import torch
import csv
import re
import os
import random
import ollama
import json
import pkg_resources
import language_tool_python
from django.shortcuts import render, redirect
from .models import AnalysisResult
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from django.http import HttpResponse
from django.shortcuts import render

from django.core.cache import cache
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt

import snscrape.modules.twitter as sntwitter

# Arabic LanguageTool
tool = language_tool_python.LanguageTool("ar")  

# Sentiment Mapping
LABELS = ["positive", "neutral", "negative"]

def home(request):
    return render(request, "analyzer/home.html")

# Load Local Dictionary
def load_replacement_dictionary():
    """ Load local dictionary of word replacements """
    replacement_dict = {}
    file_path = "dictionary.txt"

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split(",")  # Extract incorrect → correct mapping
                if len(parts) == 2:
                    incorrect, correct = parts
                    replacement_dict[incorrect] = correct

        print(f"✅ Loaded {len(replacement_dict)} words from dictionary.txt")

    except FileNotFoundError:
        print("⚠️ Warning: dictionary.txt not found. Skipping replacements.")
    except Exception as e:
        print(f"⚠️ Error loading dictionary: {e}")

    return replacement_dict

# Load Local dictionary at startup
replacement_dictionary = load_replacement_dictionary()

# Local Dictionary
def correct_using_local_dictionary(text):
    """ Replace words using local dictionary before any other correction """
    words = text.split()
    corrected_words = [replacement_dictionary.get(word, word) for word in words]  # Replace words if found
    corrected_text = " ".join(corrected_words)

    print(f"🛠 Dictionary Replacement: {corrected_text}")  # Debugging Output
    return corrected_text

# Manage Local Dictionary
def manage_dictionary(request):
    """ View to manage dictionary words via UI """
    file_path = "dictionary.txt"

    # Load dictionary into a dictionary object
    replacement_dict = {}
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split(",")
                if len(parts) == 2:
                    incorrect, correct = parts
                    replacement_dict[incorrect] = correct
    except FileNotFoundError:
        pass  # File doesn't exist yet

    if request.method == "POST":
        action = request.POST.get("action")

        if action == "add":
            incorrect_word = request.POST.get("incorrect_word").strip()
            correct_word = request.POST.get("correct_word").strip()
            if incorrect_word and correct_word:
                # Add new word to dictionary.txt
                with open(file_path, "a", encoding="utf-8") as f:
                    f.write(f"{incorrect_word},{correct_word}\n")
                print(f"✅ Added: {incorrect_word} → {correct_word}")

        elif action == "remove":
            word_to_remove = request.POST.get("word_to_remove").strip()
            if word_to_remove in replacement_dict:
                del replacement_dict[word_to_remove]
                # Rewrite file without removed word
                with open(file_path, "w", encoding="utf-8") as f:
                    for incorrect, correct in replacement_dict.items():
                        f.write(f"{incorrect},{correct}\n")
                print(f"❌ Removed: {word_to_remove}")

        # Reload dictionary after changes
        return redirect("/manage-dictionary")

    return render(request, "analyzer/manage_dictionary.html", {"dictionary": replacement_dict})

# LanguageTool
def correct_spelling_languagetool(text):
    """ Use LanguageTool for Arabic spelling correction """
    matches = tool.check(text)  # Check for spelling errors
    corrected_text = language_tool_python.utils.correct(text, matches)  # Apply corrections

    print(f"🛠 LanguageTool Correction: {corrected_text}")  # Debugging Output
    return corrected_text

def analyze_text_with_ollama(text, model_name):
    """ Send Arabic text to Ollama for sentiment analysis with model-specific confidence levels """

    # Define confidence mappings for each model
    model_confidence_map = {
        "gemma3:27b":  {"positive": 88.0, "negative": 88.0, "neutral": 72.0},
        "gemma3:12b":  {"positive": 88.0, "negative": 88.0, "neutral": 72.0},
        "gemma3:4b":  {"positive": 88.0, "negative": 88.0, "neutral": 72.0},
        "gemma3:1b":  {"positive": 88.0, "negative": 88.0, "neutral": 72.0}, # Default model
        "salmatrafi/acegpt:13b": {"positive": 92.0, "negative": 92.0, "neutral": 80.0},
        "jwnder/jais-adaptive:7b": {"positive": 90.0, "negative": 90.0, "neutral": 75.0},
        "prakasharyan/qwen-arabic": {"positive": 85.0, "negative": 85.0, "neutral": 70.0},
    }

    # Use the specified model's confidence levels (default to `salmatrafi/acegpt:13b`)
    confidence_map = model_confidence_map.get(model_name, model_confidence_map["gemma3:1b"])

    # Ask Ollama to classify sentiment
    prompt = f"""
    أنت مساعد لتحليل المشاعر. صنف الجملة التالية إلى واحدة من الفئات التالية:
    - إيجابي
    - سلبي
    - محايد

    فقط اكتب التصنيف بدون أي تفاصيل أخرى.

    الجملة: {text}  
    التصنيف:
    """

    response = ollama.chat(model=model_name, messages=[{"role": "user", "content": prompt}])
    sentiment = response["message"]["content"].strip()

    # Map Arabic sentiment to English labels
    sentiment_map = {"إيجابي": "positive", "سلبي": "negative", "محايد": "neutral"}
    sentiment = sentiment_map.get(sentiment, "neutral")

    # Get confidence for the predicted sentiment
    confidence = confidence_map[sentiment]

    return sentiment, confidence

def analyze_file(request):
    """ Handle CSV file upload and process sentiment analysis with user-selected model and spell check options """
    if request.method == 'POST' and request.FILES.get('csv_file'):
        try:
            file = request.FILES['csv_file']

            # Validate file extension
            if not file.name.endswith('.csv'):
                raise ValueError("الملف يجب أن يكون بصيغة CSV")

            # Retrieve selected model
            selected_model = request.POST.get("model_selection", "salmatrafi/acegpt:7b")  # Default model

            # Check which spell checkers are selected
            use_dictionary = request.POST.get("spell_check_dict") == "on"
            use_languagetool = request.POST.get("spell_check_languagetool") == "on"
            use_ai = request.POST.get("spell_check_ai") == "on"

            print(f"Model Selected: {selected_model}")
            print(f" Dictionary Enabled: {use_dictionary}, LanguageTool Enabled: {use_languagetool}")

            # Save uploaded file
            file_path = os.path.join("media", file.name)
            os.makedirs("media", exist_ok=True)  # Ensure "media" folder exists

            with open(file_path, "wb") as f:
                for chunk in file.chunks():
                    f.write(chunk)

            # Process CSV with selected model
            # Generate a key tied to the current user session
            session_key = f"upload_progress_{request.session.session_key}"

            # Define a callback to store progress in cache
            def progress_callback(p):
                cache.set(session_key, p, timeout=3600)  # cache for 1 hour

            # Call process_csv with progress tracking
            processed_file_path, processed_df, corrected_word_count, total_word_count = process_csv(
                file_path, selected_model, use_languagetool, use_dictionary, use_ai,
                progress_callback=progress_callback,
                session_key=request.session.session_key
            )

            # Ensure it ends at 100%
            cache.set("correction_stats", {
                "corrected_word_count": corrected_word_count,
                "total_word_count": total_word_count
            }, timeout=3600)

            # Convert to HTML Tables
            original_table = pd.read_csv(file_path).to_html(classes="csv-table", index=False)
            processed_table = processed_df.to_html(classes="csv-table", index=False)

            return render(request, "analyzer/results.html", {
                "original_table": original_table,
                "processed_table": processed_table,
                "processed_data": processed_df.to_dict(orient='records'),
                "corrected_word_count": corrected_word_count,
                "total_word_count": total_word_count,
                "correction_percent": round((corrected_word_count / total_word_count) * 100, 2) if total_word_count > 0 else 0
            })

        except Exception as e:
            return render(request, "analyzer/upload.html", {"error": str(e)})

    return render(request, "analyzer/upload.html")

def process_csv(file_path, model_name, use_languagetool=False, use_dictionary=False, use_ai=False, progress_callback=None, session_key=None):
    df = pd.read_csv(file_path)

    session_cancel_key = f"upload_cancel_{session_key}"
    cache.delete(session_cancel_key)

    if "text" not in df.columns:
        raise ValueError("CSV file must contain a 'text' column.")

    processed_data = []
    corrected_word_count = 0
    total_word_count = 0

    if 'text' in df.columns:
        for text in df['text']:
            total_word_count += len(str(text).split())

    for idx, (_, row) in enumerate(df.iterrows()):

        #Check if user canceled
        if cache.get(session_cancel_key):
            print("Upload cancelled by user.")
            break
        if progress_callback:
            progress_callback(int((idx + 1) / len(df) * 100))

        original_text = row["text"]
        corrected_text = original_text  # default

        # Step 1: Dictionary
        if use_dictionary:
            corrected_text = correct_using_local_dictionary(corrected_text)

        # Step 2: LanguageTool
        if use_languagetool and corrected_text == original_text:
            corrected_text = correct_spelling_languagetool(corrected_text)

        #Step 3: AI Spell Correction
        if use_ai:
            # Check cancel BEFORE calling Ollama
            if cache.get(session_cancel_key):
                print("Upload cancelled during AI spellcheck.")
                break

            prompt = f"""
            صحح فقط الأخطاء الإملائية في الجملة التالية دون تعديل المعنى أو إضافة شرح:

            الجملة: {corrected_text}
            الناتج:
            """
            try:
                response = ollama.chat(model=model_name, messages=[{"role": "user", "content": prompt}])
                ai_corrected = response["message"]["content"].strip()

                # Apply only if different and not empty
                if ai_corrected and ai_corrected != corrected_text:
                    corrected_text = ai_corrected

            except Exception as e:
                print(f"AI Spell Correction Failed: {e}")

            #Count word-level changes
            original_words = original_text.split()
            corrected_words = corrected_text.split()

            # Pad both lists to the same length
            max_len = max(len(original_words), len(corrected_words))
            original_words += [""] * (max_len - len(original_words))
            corrected_words += [""] * (max_len - len(corrected_words))

            for orig_word, corr_word in zip(original_words, corrected_words):
                total_word_count += 1
                if orig_word != corr_word:
                    corrected_word_count += 1

        sentiment, confidence = analyze_text_with_ollama(corrected_text, model_name)

        processed_data.append({
            "user_id": row.get("user_id", random.randint(1000, 9999)),
            "text": corrected_text,
            "sentiment": sentiment,
            "confidence": confidence,
            "platform": row.get("platform", "Unknown"),
            "date": row.get("date", pd.to_datetime("today").strftime('%Y-%m-%d')),
        })

    processed_file_path = "media/latest_processed.csv"
    processed_df = pd.DataFrame(processed_data)
    processed_df.to_csv(processed_file_path, index=False)

    print(f"AI Spellcheck Stats → Corrected Words: {corrected_word_count}, Total Words: {total_word_count}")

    return processed_file_path, processed_df, corrected_word_count, total_word_count

def download_processed_csv(request):
    """ Allow user to download the latest processed CSV file """
    file_path = "media/latest_processed.csv"

    # Ensure processed file exists
    if not os.path.exists(file_path):
        return HttpResponse("No processed file found. Please analyze a CSV first.", status=404)

    with open(file_path, "rb") as f:
        response = HttpResponse(f.read(), content_type="text/csv")
        response["Content-Disposition"] = 'attachment; filename="processed_results.csv"'
        return response

def analyze_uploaded_csv(request):
    if request.method == 'POST' and request.FILES['csv_file']:
        csv_file = request.FILES['csv_file']
        df = pd.read_csv(csv_file)

        df["sentiment"] = df["text"].apply(analyze_text_with_ollama)

        # Save processed CSV
        processed_csv_path = os.path.join(settings.MEDIA_ROOT, "processed_results.csv")
        df.to_csv(processed_csv_path, index=False)

        return render(request, 'results.html', {'processed_data': processed_df.to_dict(orient='records')})

def statistics_view(request):
    processed_file_path = os.path.join("media", "latest_processed.csv")

    if not os.path.exists(processed_file_path):
        return render(request, 'analyzer/statistics.html', {"error": "لم يتم العثور على بيانات."})

    df = pd.read_csv(processed_file_path)

    # Count sentiment distribution
    sentiment_counts = df['sentiment'].value_counts().to_dict()

    # Count platform distribution
    platform_counts = df['platform'].value_counts().to_dict() if 'platform' in df.columns else {}

    # Count sentiment per platform (grouped)
    if "platform" in df.columns and "sentiment" in df.columns:
        sentiment_per_platform = df.groupby(["platform", "sentiment"]).size().unstack(fill_value=0)
        sentiment_per_platform = sentiment_per_platform.reset_index().melt(id_vars="platform", var_name="sentiment", value_name="count")
        sentiment_per_platform_list = sentiment_per_platform.to_dict(orient="records")
    else:
        sentiment_per_platform_list = []

    # Compute **dynamic confidence threshold** using the median
    if "confidence" in df.columns and not df["confidence"].isna().all():
        dynamic_threshold = df["confidence"].median()  # Use median confidence as dynamic threshold
        lower_confidence = df[df["confidence"] < dynamic_threshold].shape[0]
        higher_confidence = df[df["confidence"] >= dynamic_threshold].shape[0]
        confidence_distribution = {
            "low": lower_confidence,
            "high": higher_confidence,
            "threshold": dynamic_threshold  # Pass dynamic threshold to the frontend
        }
    else:
        confidence_distribution = {"low": 0, "high": 0, "threshold": 50}  # Default fallback

    # Debugging Logs
    print("Confidence Data:", confidence_distribution)

    # Compute correction stats based on current CSV (if available)
    correction_stats = cache.get("correction_stats", {})
    corrected_word_count = correction_stats.get("corrected_word_count", 0)
    total_word_count = correction_stats.get("total_word_count", 0)


    context = {
        "sentiment_counts": json.dumps(sentiment_counts),
        "platform_counts": json.dumps(platform_counts),
        "sentiment_per_platform": json.dumps(sentiment_per_platform_list),
        "confidence_distribution": json.dumps(confidence_distribution),
        "processed_data": df.to_dict(orient='records'),

        "corrected_word_count": corrected_word_count,
        "total_word_count": total_word_count,
        "correction_percent": round((corrected_word_count / total_word_count) * 100, 2) if total_word_count > 0 else 0
    }


    return render(request, 'analyzer/statistics.html', context)

def get_upload_progress(request):
    session_key = f"upload_progress_{request.session.session_key}"
    percent = cache.get(session_key, 0)
    return JsonResponse({"percent": percent})

@csrf_exempt
def cancel_upload(request):
    if request.method == "POST":
        session_key = f"upload_cancel_{request.session.session_key}"
        cache.set(session_key, True, timeout=3600)  # Mark as canceled
        return JsonResponse({"status": "canceled"})
    return JsonResponse({"error": "invalid method"}, status=405)

def fetch_tweet_replies(tweet_url):
    tweet_id = tweet_url.strip("/").split("/")[-1]
    results = []

    for reply in sntwitter.TwitterTweetScraper(tweetId=tweet_id, mode='replies').get_items():
        results.append({
            "text": reply.content,
            "user_id": reply.user.username,
            "date": reply.date.strftime("%Y-%m-%d"),
            "platform": "Twitter"
        })

    return results

def save_replies_to_csv(request):
    if request.method == 'POST':
        tweet_url = request.POST.get('tweet_url')

        try:
            replies = fetch_tweet_replies(tweet_url)

            if not replies:
                return render(request, "analyzer/twitter.html", {"error": "لم يتم العثور على ردود على التغريدة."})

            df = pd.DataFrame(replies)
            os.makedirs("media", exist_ok=True)
            csv_path = os.path.join("media", "twitter_comments.csv")
            df.to_csv(csv_path, index=False, encoding="utf-8-sig")

            return render(request, "analyzer/twitter.html", {
                "success": f"تم حفظ {len(replies)} رد في ملف CSV.",
                "csv_path": csv_path
            })

        except Exception as e:
            return render(request, "analyzer/twitter.html", {"error": f"حدث خطأ: {str(e)}"})

    return render(request, "analyzer/twitter.html")