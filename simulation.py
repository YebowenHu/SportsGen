
import os
import json
import fire

from tqdm import tqdm

from utils.NBAPlayer import select_team_players, load_team_profile
from utils.GameGenerator import simulate_single_game, RATIO2ALPHA
from utils.stats import games_statistics, is_number

def create_new_games(bench_name, bench_size, strong_team_strength, weak_team_strength, ratio, anonymous):
    save_dir = f"simulations/{bench_name}_{bench_size}_{ratio}"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    elif os.path.exists(save_dir) and len(os.listdir(save_dir)) > 0:
        print(f"Folder {save_dir} already exists and is not empty.")
        return
    
    # load density ratio
    if ratio not in RATIO2ALPHA and is_number(ratio):
        alpha = ratio
    elif ratio in RATIO2ALPHA:
        alpha = RATIO2ALPHA[ratio]
    else:
        raise ValueError(f"Invalid ratio: {ratio}")
    
    for game_id in tqdm(range(bench_size)):
        match_teams_obj = select_team_players(strong_team_strength, weak_team_strength, anonymous=anonymous)
        match_compare_scores, match_player_dict = load_team_profile(match_teams_obj)
        simulation = simulate_single_game(match_compare_scores, match_player_dict, alpha=alpha)
        with open(os.path.join(save_dir, f"game_{game_id}.json"), 'w') as f:
            f.write(json.dumps(simulation, indent=4))
    print(f"Game Simulation Completed: save to {save_dir}")
    games_statistics(save_dir)
    return
    
if __name__ == "__main__":
    fire.Fire(create_new_games)