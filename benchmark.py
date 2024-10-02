import os
import json
import fire
from glob import glob

SYS_PROMPT = """You are a helpful assistant tasked with analyzing sports games. You have been given a play-by-play breakdown of an NBA basketball game between two teams.\n
The "Time" column shows the exact time on the game clock when each play took place. The game clock counts down, so this column displays times in a descending order.\n
The "Play" column describes the action that happened at the respective times. It provides details of specific plays, movements, and outcomes on the court.\n
Team players are listed in two rows, each row representing one of the two basketball teams involved in the game.
"""


def load_json(fpath):
    with open(fpath, 'r') as r:
        j_data = json.load(r)
        return j_data

    
def batchit(corpus, size=128):
    assert hasattr(corpus, "__iter__")
    assert size is None or isinstance(size, int) and size > 0
    batched_corpus = []
    batch = []
    for row in corpus:
        batch.append(row)
        if len(batch) == size:
            batched_corpus.append(batch.copy())
            batch.clear()
    if len(batch) > 0:
        batched_corpus.append(batch)
    return batched_corpus

def team_score(team_players):
    initial_scores = {k:0 for k in team_players.keys()}
    team_affiliations = ""
    for team in team_players.keys():
        team_affiliations += "\n" + team + ": " + ", ".join(team_players[team])
    
    task_prompt = f"Analyze the team-player affiliations and play-by-play descriptions below to determine the total points scored by each team.\nPlease explain your reasoning step by step and provide the final results in the following JSON format.\n{json.dumps(initial_scores)}\n\n#Team-Player Affiliations:{team_affiliations}\n\n#Play-by-Play Descriptions:\nTime\tPlay\n"

    return [task_prompt]

def initial_team_scores(pbp_data):
    init_score = {}
    for play in pbp_data[0]:
        if play['ScoringPlay']:
            init_score[play['team']] = 0
        if len(init_score.keys()) == 2:
            break
    return init_score


def generate_pbp_desc(pbp_data, step_size=False):
    # yield quarter description and ground truth 
    quarter_id = 0
    init_team_scores = initial_team_scores(pbp_data)
    for quarter in pbp_data:
        quarter_id += 1
        if step_size:
            quarter_data = batchit(quarter, step_size)
        else:
            quarter_data = [quarter]
        seg_id = 0
        # process
        for segment in quarter_data:
            pbp_desc = ""
            ground_truth = init_team_scores.copy()
            seg_id += 1
            for play in segment:
                play_desc = play['time'] + "\t" + play['description']
                pbp_desc += play_desc + '\n'

                # record ground truth
                if play['team'] not in ground_truth.keys() and play['ScoringPlay']:
                    ground_truth[play['team']] = play['points']
                elif play['ScoringPlay']:
                    ground_truth[play['team']] += play['points']
            yield f"{quarter_id}_{seg_id}", pbp_desc, ground_truth
    return

def player_scores(team_players):
    task_prompts = []
    team_affiliations = ""
    for team in team_players.keys():
        team_affiliations += "\n" + team + ": " + ", ".join(team_players[team])
    
    for team in team_players.keys():
        players = team_players[team]
        initial_scores = {p:0 for p in players}

        task_prompt = f"Analyze the team-player affiliations and play-by-play descriptions below to determine the total points scored by each player.\nPlease explain your reasoning step by step and provide the final results in the following JSON format.\n{json.dumps(initial_scores)}\n\n#Team-Player Affiliations:{team_affiliations}\n\n#Play-by-Play Descriptions:\nTime\tPlay\n"

        task_prompts.append((team, task_prompt))
    return task_prompts

def task_generate(game_folder, bench_name, steps, player_stats):
    if steps:
        bench_name += f"-step_{steps}"
    if player_stats:
        bench_name += "-player_stats"
        
    save_file = os.path.join("benchmarks",f"{bench_name}.json")
    
    if os.path.exists(save_file):
        print(f"File {save_file} already exists.")
        return
    
    # safe all evaluation instances in one file
    bench_data = []
    for fpath in glob(os.path.join(game_folder, "*.json")):
        jdata = load_json(fpath)
        game_name = fpath.split("/")[-1].split(".")[0]
        if player_stats:
            task_prompts = player_scores(jdata['team_players'])
        else:
            task_prompts = team_score(jdata['team_players'])
        for step_id, desc, g in generate_pbp_desc(jdata['pbp'], steps):
            if player_stats:
                bench_data.extend([{
                    "instance_id": game_name + f"_{task_prompt[0]}" + f"_{step_id}", 
                    "system_msg": SYS_PROMPT,
                    "prompt_msg": task_prompt[1] + desc,
                    "truth": g
                }] for task_prompt in task_prompts)
            else:
                bench_data.append({
                    "instance_id": game_name + f"_{step_id}", 
                    "system_msg": SYS_PROMPT,
                    "prompt_msg": task_prompts[0] + desc,
                    "truth": g
                })
    
    print(f"Load {len(bench_data)} instances from {game_folder}\nSave to {save_file}")
    with open(save_file, 'w') as w:
        for instance in bench_data:
            w.write(json.dumps(instance) + "\n")
    
    return

if __name__=="__main__":
    fire.Fire(task_generate)