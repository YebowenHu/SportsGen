import os
import json
import tiktoken
import numpy as np
from glob import glob

enc = tiktoken.get_encoding("cl100k_base")

def is_number(input_value):
    try:
        # Try to convert the input to a float
        float(input_value)
        return True
    except ValueError:
        # If an exception is raised, the input is not a number
        return False

def load_folder(folder):
    file_list = []
    for fpath in glob(os.path.join(folder, "*.json")):
        file_list.append(fpath)
    return file_list

def load_quarters(file_list):
    # print(file_list[0])
    quarters = []
    for file in file_list:
        with open(file, 'r') as f:
            j_data = json.load(f)
            game_quarter = j_data["pbp"][0:4]
            quarters.extend(game_quarter)
    return quarters

def games_statistics(folder_path):
    """
        Analyzing quarter sattistics
        1. #plays/turn {ave_play, play_range}
        2. density: scoring move/total_move, {avg_density, density_range}
        3. #tokens {avg_token, token_range}
        4. #turns {avg_turn, turn_range}
    """
    file_list = load_folder(folder_path)
    # load all quarters
    quarters = load_quarters(file_list)
    statistic_dict = {"#plays":[], "density":[], "#tokens":[], "#turns":[],"#free throw":[], "#otherScore":[], "game_score":[]}

    for quarter in quarters:
        statistic_dict['#plays'].append(len(quarter))
        statistic_dict['density'].append(len([play for play in quarter if play['ScoringPlay']])/len(quarter))

        # count tokens
        total_text = "\n".join([play['time'] + "\t" + play['description'] for play in quarter])
        tokens = enc.encode(total_text)
        statistic_dict['#tokens'].append(len(tokens))

        # count turns
        cur_team = None
        total_turn = 0
        for play in quarter:
            if play['team'] is None:
                continue
            team = play['team']
            if team != cur_team:
                cur_team = team
                total_turn += 1
        statistic_dict['#turns'].append(total_turn)
        
        # count free throw
        points_list = [0,0]
        total_other = 0
        total_free_throw = 0
        for play in quarter:
            if "free throw" in play['description'] and play['ScoringPlay']:
                total_free_throw += 1
            elif play['ScoringPlay']:
                total_other += 1
                gain_points = play['points']
                gain_team = 0 if play['team'] == "team1" else 1
                points_list[gain_team] += gain_points
        statistic_dict['#free throw'].append(total_free_throw)
        statistic_dict['#otherScore'].append(total_other)
        # print(points_list)
        statistic_dict['game_score'].append(points_list)
    # calculate statistics
    ratio_value = (1 - np.mean(statistic_dict['density']))/np.mean(statistic_dict['density'])
    print("############# Simulations Statistics #############")
    print(f"S: NS is 1:{round(ratio_value, 1)}")
    print(f"average #plays: {np.mean(statistic_dict['#plays'])}, min: {min(statistic_dict['#plays'])}, max: {max(statistic_dict['#plays'])}")
    print(f"average #tokens: {np.mean(statistic_dict['#tokens'])}, min: {min(statistic_dict['#tokens'])}, max: {max(statistic_dict['#tokens'])}")
    print(f"average #turns: {np.mean(statistic_dict['#turns'])}, min: {min(statistic_dict['#turns'])}, max: {max(statistic_dict['#turns'])}")
    print(f"# free thow: {np.sum(statistic_dict['#free throw'])/len(statistic_dict['#plays'])},\
           # other score: {np.sum(statistic_dict['#otherScore'])/len(statistic_dict['#plays'])}")
    
    # calculate win rate
    team1_win = 0
    team2_win = 0
    draw = 0
    for i in statistic_dict['game_score']:
        if i[0] > i[1]:
            team1_win += 1
        elif i[0] < i[1]:
            team2_win += 1
        else:
            draw += 1
    print(f"team1 win rate: {team1_win/len(statistic_dict['game_score'])}, team2 win rate: {team2_win/len(statistic_dict['game_score'])}, draw rate: {draw/len(statistic_dict['game_score'])}")
    return