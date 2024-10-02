
import os
import re
import json
import math
import yaml
import time
import pickle
import random

import scipy.stats as stats
import networkx as nx

from openai import OpenAI

current_directory = os.getcwd()
# initial openai client
config = yaml.safe_load(open("config/openai_key.yaml"))
client = OpenAI(api_key=config["api-key"])
DEFAULT_ENGINE = "gpt-4o-mini"
PARAMETERS = config['parameters']
random.seed(42)


RATIO2ALPHA = {
    "1:2": -0.3,
    "1:3": 0,
    "real": 0,
    "1:5": 0.9,
    "1:4": 0.5,
}

def openai_request(**kwargs):
    try:
        response = client.chat.completions.create(**kwargs)
    except Exception as e:
        print(f"Error: {e}")
        return None
    return response

def load_pickle(file_path):
    with open(file_path, 'rb') as file:
        var = pickle.load(file)
    return var

def load_json(file_path):
    with open(file_path, 'r') as file:
        var = json.load(file)
    return var

# Define a function to build a directed graph from a list of event sequences
def build_tree_with_probabilities(paths):
    G = nx.DiGraph()
    edge_counts = {}
    
    for path in paths:
        for i in range(len(path) - 1):
            edge = (path[i], path[i + 1])
            if edge not in edge_counts:
                edge_counts[edge] = 0
            edge_counts[edge] += 1

    node_counts = {}
    for (u, v), count in edge_counts.items():
        if u not in node_counts:
            node_counts[u] = 0
        node_counts[u] += count
    
    for (u, v), count in edge_counts.items():
        probability = count / node_counts[u]
        G.add_edge(u, v, weight=probability)
    
    return G


def random_choice_with_prob(candidates, prob_list=False, key_word=False):
    # prob_list default to be uniform distribution
    if not prob_list:
        prob_list = [1/len(candidates)]*len(candidates)
    if key_word:
        for i in range(len(candidates)):
            if key_word in candidates[i]:
                return candidates[i]
    return random.choices(candidates, prob_list)[0]

def make_or_miss(prob):
    """
        Given team's overall score[0-100], return "make" or "miss" event
    """
    prob = (prob / 100) * (0.58-0.36) + 0.36 # average of highest team attack FG% 50.7, 3P% 37.9%, FT% 85.9%
    return random.choices(["make", "miss"], [prob, 1-prob])[0]

def modify_num_play_each_turn(density):
    """
        adjust gaussian distribution to generate number of plays in each turn
        original distribution: mean = 1.65, std = 0.92
        density: [0,1]
        mean = 1.65*(1 +  2 * density), std = 0.92
        output: number of plays for this turn
    """
    mean = 1.65 * (1 + 2 * density)
    std = 0.92
    distribution = stats.norm(mean, std)
    prob_list=  [distribution.pdf(i) for i in range(1, 11)]
    num_play = random_choice_with_prob(range(1, 11), prob_list)
    return num_play


def num_ele_in_list(list, element):
    return len([ele for ele in list if ele == element])


def generate_turn(tree, quarter_init=False):
    # traverse from "stat" to "end" to generate a path
    end_node = "end"
    paths = []
    cur_node = "start"
    
    if quarter_init:
        cur_node = "vs"
        paths.append(cur_node)

    # generate a path from start to end
    while cur_node != end_node:
        next_node_list = [v for v in tree[cur_node].keys()]
        if num_ele_in_list(paths, "make") == 2:
            # if two "make" events are consecutive, the next event should not be "make"
            next_node_probs = [tree[cur_node][v]['weight'] if v != "make" else 0 for v in next_node_list]
        else:
            next_node_probs = [tree[cur_node][v]['weight'] for v in next_node_list]
        next_node = random.choices(next_node_list, next_node_probs)[0] # given possible children nodes and probabilities, randomly choose one
        if next_node != end_node:
            paths.append(next_node)
        cur_node = next_node

    return paths


def parse_points(text):
    text = text.lower()
    ddd = re.findall(r'\d+', text)
    if "free throw" in text:
        return 1
    elif len(ddd)>0:
        distance = int(ddd[0])
        if distance < 23:
            return 2
        elif distance == 23 and "three" not in text:
            return 2
        else:
            return 3
    elif "two" in text:
        return 2
    elif "three" in text:
        return 3
    elif "layup" in text or "dunk" in text or "tip" in text or "jump shot" in text or "jumper" in text or "hook shot" in text or "step back jump":
        return 2
    else:
        # print(text)
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": f"Calculate the points for this play description. \n\nExample 1:\nText: makes jump bank shot\nPoint: 2\n\nText: {text}\nPoints:"}
        ]
        PARAMETERS.update({"messages": messages, "model": DEFAULT_ENGINE})
        response = openai_request(**PARAMETERS)
        if response is not None:
            return int(response.choices[0].message.content.lower())
        return 0 

def conditional_turn_generator(graph, num_plays, key_event=False, quarter=False):   
    number_of_make = random.choices([1,2],[0.75, 0.25])[0]
    max_retry = 10000
    while True:
        path = generate_turn(graph, quarter_init=quarter)
        # ! GPT-3.5 reject unreasonable path
        if key_event == "miss":
            if key_event not in path or "make" in path:
                continue
        elif key_event == "make":
            if key_event not in path:
                continue
            elif num_ele_in_list(path, "make") == number_of_make:
                # print(f"number of make: {num_ele_in_list(path, 'make')}, expected: {number_of_make}")
                return path
        if "vs" in path and len(path) - 1 == num_plays:
            return path  
        elif len(path) == num_plays:
            return path
        max_retry -= 1
        if max_retry == 0:
            return path # if reach the max retry, randomly return the path


def path_template(path, verb_desc_dict):
    path_desc = []
    active_free_throw = False
    for pos,event in enumerate(path):
        if event not in verb_desc_dict.keys():
            print(f"Event {event} not in verb_desc_dict")
            continue

        desc_candidates = verb_desc_dict[event]

        # check if two "make" events are consecutive
        if event == "make" and pos < len(path) - 1 and path[pos + 1] == "make":
            active_free_throw = True
            path_desc.append(random_choice_with_prob(desc_candidates, key_word="free throw"))
            continue

        if active_free_throw:
            path_desc.append(random_choice_with_prob(desc_candidates, key_word="free throw"))
            continue

        # other event
        path_desc.append(random_choice_with_prob(desc_candidates))
    return path_desc


def convert_time_to_seconds(timestamp):
    if ":" in timestamp:
        time = timestamp.split(":")
        return int(time[0])*60 + int(time[1])
    else:
        return math.ceil(float(timestamp))

def fill_in_players(text, team_name, player_name_dict):
    # find all content in <> and replace with player name
    player_name = re.findall(r'<(.*?)>', text)
    for name in player_name:
        if "team" in name.lower():
            text = text.replace(f"<{name}>", team_name)
        elif "-" in name:
            pos = name.split("-")[1]
            if pos.upper() in player_name_dict.keys():
                text = text.replace(f"<{name}>", player_name_dict[pos.upper()]['name'])
            else: # if 
                pos_list = list(player_name_dict.keys())
                random_pos = random.choice(pos_list)
                text = text.replace(f"<{name}>", player_name_dict[random_pos]['name'])
        else:
            pos_list = list(player_name_dict.keys())
            random_pos = random.choice(pos_list)
            text = text.replace(f"<{name}>", player_name_dict[random_pos]['name'])
    return text

def convert_seconds_to_time(seconds):
    min = seconds // 60
    sec = seconds % 60
    if sec < 10:
        sec = f"0{sec}"
    return f"{min}:{sec}"

def get_timestamp(event_duration, path, start_time):
    time_stamps = []
    start_seconds = convert_time_to_seconds(start_time)
    cur_time = start_seconds
    for event in path:
        # return current time_stamps if cur_time < 0

        if event not in event_duration.keys():
            print(f"Event {event} not in event_duration")
            time_stamps.append(convert_seconds_to_time(start_seconds))
            continue
        
        # get duration list and prob
        dur_list = []
        prob_list = []
        for k,v in event_duration[event].items():
            if event not in ["make", "defensive goaltending violation"] and k == 0:
                continue
            dur_list.append(k)
            prob_list.append(v)
        duration = random_choice_with_prob(dur_list, prob_list)
        cur_time -= duration

        # if updated time is less the 0, means the quarter ends
        if cur_time < 0:
            return time_stamps

        time_stamps.append(convert_seconds_to_time(cur_time))
    return time_stamps

def load_data():
    # load data
    event_duration = load_pickle("model_data/event_duration.pkl")
    event_seqs = load_pickle("model_data/event_seqs.pkl")
    verb_to_desc = load_json("model_data/desc_template.json") # GPT-4 polished description
    markov_graph = build_tree_with_probabilities(event_seqs)
    return event_duration, verb_to_desc, markov_graph

def generate_game(quarter_id, alpha, player_name_dict, team_power):
    """Generate quarter of game with at most 150 turns
        player_name_dict; {"pos":player}
    """
    _density = alpha # density of scoring move
    total_game = []
    event_duration, verb_to_desc, markov_graph = load_data()
    cur_time_stamp = "12:00"
    team_name = ['team1', 'team2']
    total_scoring_move = 0
    total_move = 0
    for i in range(200): # at most 150 events in a game
        team_id = i%2
        # get istribution of number of plays in a path according to gaussian distribution
        num_of_plays = modify_num_play_each_turn(_density)
        # decide make or miss event
        _key_event = make_or_miss(team_power[team_id])
        cur_team = team_name[team_id]

        path = conditional_turn_generator(markov_graph, num_plays=num_of_plays, key_event=_key_event, quarter=False if quarter_id > 0 else True)
        
        # validate the number of plays in the path
        if len(path) == 0:
            continue
        
        # stats the ratio of scoring event
        total_scoring_move += len([ele for ele in path if ele == "make"])
        total_move += len(path)
        
        templates = path_template(path, verb_to_desc)
        timestamp = get_timestamp(event_duration, path, cur_time_stamp)
        # display generated game
        
        for pos, (time, play) in enumerate(zip(timestamp, templates[0:len(timestamp)])):
            
            # decide scores made by the play
            if path[pos] != "make":
                score_point = 0
                Scoring_play = False
            else:
                Scoring_play = True
                score_point = parse_points(play)

            # load players
            cur_players = player_name_dict[cur_team]
            play = fill_in_players(play, cur_team, cur_players)

            total_game.append({
                "team": cur_team,
                "time": time,
                "description": play,
                "ScoringPlay": Scoring_play,
                "points": score_point,
            })

        if len(timestamp) < len(path): # the generated path will be cut off when the quarter ends
            total_game.append({
                "team":None,
                "time": "0:0",
                "description": "end of quarter",
                "ScoringPlay": False,
                "points": 0
            })
            break
        cur_time_stamp = timestamp[-1]
    # print(f"total scoring move ratio: {total_scoring_move/total_move}")
    return total_game

def simulate_single_game(power_list, players_dict, alpha=0.5):
    random.seed(int(time.time()))
    total_game = []
    # game_init = True
    for qid in range(4):
        # # print(qid)
        # if qid > 0:
        #     game_init = False
        quarter_game = generate_game(qid, alpha=alpha, player_name_dict=players_dict, team_power=power_list)
        quarter_game.insert(0, {"team": None, "time": "12:00", "description": f"start of quarter {qid+1}", "ScoringPlay": False, "points": 0})
        total_game.append(quarter_game)

    # append player info at the end of game
    # total_game.append(players[game_id])
    player_dict = {}
    for team in players_dict.keys():
        player_list = players_dict[team].copy()
        player_dict[team] = [v['name'] for v in player_list.values()]
    # compute totalt score:
    team_score = {}
    for quarter in total_game:
        for play in quarter:
            if play['team'] is None:
                continue
            if play['team'] not in team_score.keys():
                team_score[play['team']] = 0
            else:
                team_score[play['team']] += play['points']

    # final_game
    simulated_game = {
        "pbp":total_game,
        "team_players": player_dict,
        "team_scores": team_score
        

    }
    return simulated_game