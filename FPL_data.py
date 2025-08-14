import requests
import pandas as pd
from time import sleep
import os

BASE_URL = "https://fantasy.premierleague.com/api"
OUTPUT_FILE = "fpl_players_full_stats.csv"

# Load already processed player IDs if file exists
if os.path.exists(OUTPUT_FILE):
    existing_df = pd.read_csv(OUTPUT_FILE)
    processed_ids = set(existing_df["PlayerID"])
    all_data = existing_df.to_dict(orient="records")
    print(f"✅ Loaded {len(processed_ids)} already processed players.")
else:
    processed_ids = set()
    all_data = []
    print("📄 No existing file found. Starting fresh.")

# 1. Get all players and teams
bootstrap = requests.get(f"{BASE_URL}/bootstrap-static/").json()
players = bootstrap['elements']
teams = {team['id']: team['name'] for team in bootstrap['teams']}
positions = {pos['id']: pos['singular_name'] for pos in bootstrap['element_types']}

# 2. Loop through players
for i, p in enumerate(players, start=1):
    player_id = p['id']

    # Skip if already processed
    if player_id in processed_ids:
        print(f"[{i}/{len(players)}] Skipping {p['first_name']} {p['second_name']} (already processed)")
        continue

    player_name = f"{p['first_name']} {p['second_name']}"
    club = teams[p['team']]
    pos = positions[p['element_type']]
    price = p['now_cost'] / 10  # in millions

    try:
        # Get detailed history for last season stats
        details = requests.get(f"{BASE_URL}/element-summary/{player_id}/").json()
        history_past = details.get('history_past', [])
        last_season_stats = history_past[-1] if history_past else {}

        # Combine both current season & last season data
        row = {
            "PlayerID": player_id,
            "Name": player_name,
            "Club": club,
            "Position": pos,
            "Price (£m)": price,

            # --- Current season data ---
            "Current Form": p['form'],
            "Selected By (%)": p['selected_by_percent'],
            "ICT Index": p['ict_index'],
            "Total Points (Current)": p['total_points'],
            "Goals (Current)": p['goals_scored'],
            "Assists (Current)": p['assists'],
            "Clean Sheets (Current)": p['clean_sheets'],
            "Minutes (Current)": p['minutes'],

            # --- Last season data ---
            "Last Season": last_season_stats.get('season_name'),
            "Minutes (Last Season)": last_season_stats.get('minutes'),
            "Goals (Last Season)": last_season_stats.get('goals_scored'),
            "Assists (Last Season)": last_season_stats.get('assists'),
            "Clean Sheets (Last Season)": last_season_stats.get('clean_sheets'),
            "Total Points (Last Season)": last_season_stats.get('total_points')
        }

        # Append to list and save immediately
        all_data.append(row)
        pd.DataFrame(all_data).to_csv(OUTPUT_FILE, index=False)

        processed_ids.add(player_id)
        print(f"[{i}/{len(players)}] ✅ Saved {player_name}")

    except Exception as e:
        print(f"[{i}/{len(players)}] ❌ Error processing {player_name}: {e}")

    sleep(0.5)  # avoid hammering FPL's servers
