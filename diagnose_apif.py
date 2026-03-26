# add to diagnose_apif.py or run standalone
import requests, os
from dotenv import load_dotenv
load_dotenv()
r = requests.get("https://v3.football.api-sports.io/status",
    headers={"x-rapidapi-key": os.getenv("API_FOOTBALL_KEY"),
             "x-rapidapi-host": "v3.football.api-sports.io"})
print(r.json())