import random
import json
import numpy as np
# load nba players profiles

def load_players():
    file_path = "model_data/NBAplayer.json"
    with open(file_path) as f:
        data = json.load(f)
    return data

def is_player_in_team(player_name, team):
    return any(team_member["name"] == player_name["player"] for team_member in team)

def select_team_players(lowest_score_strong_team,
                        highest_score_weak_team,
                        anonymous=False):
    # select a strong team with with overall rating > upper_bound
    # select a weak team with overall rating < lower_bound
    # each team select 10 players. 
    strong_lower = lowest_score_strong_team
    weak__upper = highest_score_weak_team
    players = load_players()
    positions = ["PG", "SG", "SF", "PF", "C"]
    strong_team = [] # profile of strong team element={name, general_abilities}
    weak_team = []
    
    # select Starter players for both team
    for p, pos in enumerate(positions):
            
        
        strong_player_list = [player for player in players if player["biographic information"] is not None 
                      and pos in player['biographic information']['Position'] 
                      and player["general abilities"]["Overall"] >= strong_lower 
                      and not is_player_in_team(player, strong_team)]
        weak_team_list = [player for player in players if player["biographic information"] is not None 
                        and pos in player['biographic information']['Position'] 
                        and player["general abilities"]["Overall"] <= weak__upper 
                        and not is_player_in_team(player, weak_team)]
        
        strong_player = random.choice(strong_player_list)
        weak_player = random.choice(weak_team_list)
        
        if len(strong_team) != p+1:
            strong_team.append({"name": strong_player["player"], "position":strong_player['biographic information']['Position']\
                            , "general_abilities": strong_player["general abilities"]})
        if len(weak_team) != p+1:
            weak_team.append({"name": weak_player["player"],"position":weak_player['biographic information']['Position']\
                            , "general_abilities": weak_player["general abilities"]})

    # select bench players for both team
    strong_team_exist_player = [player['name'] for player in strong_team]
    weak_team_exist_player = [player['name'] for player in weak_team]
    cur_pos = random.choice(positions) # randomly assign position
    for player in players:
        if player['biographic information'] is None or player['player'] in strong_team_exist_player or player['player'] in weak_team_exist_player:
            continue
        if cur_pos not in player['biographic information']['Position']:
            continue
        if player["general abilities"]["Overall"] >= strong_lower and len(strong_team) < 10:
            strong_team.append({"name": player["player"], "position":player['biographic information']['Position'], "general_abilities": player["general abilities"]})
            cur_pos = random.choice(positions)
        if player["general abilities"]["Overall"] < weak__upper and len(weak_team) < 10:
            weak_team.append({"name": player["player"], "position":player['biographic information']['Position'], "general_abilities": player["general abilities"]}) 
            cur_pos = random.choice(positions)\
    
    # anomymous the player name
    if anonymous:
        total_palayer = len(strong_team) + len(weak_team)
        random_player_id = np.random.choice(np.arange(1,100), size=total_palayer, replace=True)
        for p, player in enumerate(strong_team+weak_team):
            player['name'] = f"Player{random_player_id[p]}"
    return_data_obj = {"strong_team": strong_team, "weak_team": weak_team}
        
    return return_data_obj

def load_team_profile(team_data_obj):
    def cal_average_power(team):
        return sum([(player["general_abilities"]["Inside Scoring"] + player["general_abilities"]["Outside Scoring"])/2 for player in team])/len(team)

    def pick_5_players( playerlist):
        picke_player = {}
        for player in playerlist:
            if "/" in player["position"]:
                positions = player["position"].split("/")
            else:
                positions = [player["position"]]
            for pos in positions:
                if len(picke_player) == 5:
                    break
                if pos not in picke_player.keys():
                    player['position'] = pos
                    picke_player[pos] = player
                    break
        return picke_player
    
    strong_team_score = cal_average_power(team_data_obj["strong_team"])
    weak_team_score = cal_average_power(team_data_obj["weak_team"])
    
    team_player_dict = {
        "team1": pick_5_players(team_data_obj["strong_team"]),
        "team2": pick_5_players(team_data_obj["weak_team"])
    }
    
    return [strong_team_score, weak_team_score], team_player_dict