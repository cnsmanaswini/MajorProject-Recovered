from dotenv import load_dotenv
load_dotenv()

from ai.pipeline.loader import preload_models
from ai.pipeline.analyzer import analyze_text

preload_models()

texts = [
    "Listening to Joey really helps me and my anger.",
    "HUGE CONGRATULATIONS TO NICOLE WINNING BIG BROTHER 18! sorry not sorry #bbmichelle",
    "bts' trilogy MV is my all time fav, quite gloomy but beautiful as well",
]

for t in texts:
    result = analyze_text(t)
    print(t, "->", result.emotion, result.emotion_score)