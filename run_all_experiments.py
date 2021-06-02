import pickle
import os
import platform
from joblib import Parallel, delayed
import random
from deep_rl_for_swarms.ma_envs.envs.point_envs import attack as attack
import numpy as np


if __name__ == '__main__':

    def processinput(at, me, seed, nat_idx):
        # Prepare data output
        data = {}
        data["at"] = at  # Attack mode
        data["me"] = me  # Whether to use me or not
        data["seed"] = seed  # Seed number
        data["save_flag"] = save_flag
        data["nat"] = nr_at[nat_idx]
        data["ns"] = nr_sensors[nat_idx]
        if me == "no_com":
            data["comm_range"] = 0
        else:
            data["comm_range"] = comm_range
        data["base_dir"] = base_dir
        data["trpo_iterations"] = trpo_iterations
        data["batch_size"] = batch_size
        data["env_timesteps_phy"] = env_timesteps_phy
        data["at_th"] = at_th
        data["gamma"] = gam

        # Generate hash for interchange file
        hash = str(random.getrandbits(128))
        data_output = 'data' + hash + '.pickle'
        # Write data output
        with open(data_output, 'wb') as handle:
            pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

        if platform.system() == 'Windows':
            print("Running on Windows")
            _ = os.system('python run_attack_ma_trpo.py ' + hash)  # Windows order
        else:
            print("Running on Linux")
            _ = os.system('python3 run_attack_ma_trpo.py ' + hash)  # Linux order

        # Delete ancillary file
        os.remove(data_output)

    def processinput_policy(at, me, seed, nat):
        save_dir = os.path.normpath(base_dir + '/' + at + '_' + me + '_' + str(nat) + '/' + str(seed))
        data_output = os.path.normpath(save_dir + '/data_policy.pickle')
        if not replace and os.path.isfile(data_output):  # Data already exists
            pass
        else:
            # Prepare data output
            data = {}
            data["at"] = at
            data["me"] = me
            data["seed"] = seed
            data["nr_at"] = nat
            data["nr_sensors"] = nr_sensors[nr_at.index(nat)]
            data["comm_range"] = comm_range
            data["trpo_iterations"] = trpo_iterations
            data["env_timesteps_phy"] = env_timesteps_phy
            data["base_dir"] = base_dir
            data["nr_episodes"] = nr_episodes
            data["at_th"] = at_th
            data['gamma'] = gam

            # Generate hash for interchange file
            hash = str(random.getrandbits(128))
            data_output = 'data' + hash + '.pickle'
            # Write data output
            with open(data_output, 'wb') as handle:
                pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

            if platform.system() == 'Windows':
                _ = os.system('python policy_exam.py ' + hash)  # Windows order
            else:
                _ = os.system('python3 policy_exam.py ' + hash)  # Linux order

            # Delete ancillary file
            os.remove(data_output)

    def processinput_baselines(at, nat):

        env = attack.AttackEnv(nr_agents=nat, nr_agents_total=nr_sensors[nr_at.index(nat)],
                               obs_mode='sum_obs_multi', attack_mode=at, obs_radius=5000, world_size=1000,
                               phy_agent_memory=5, lambda_phy=at_th, K=5, L=1000, lambda_mac=at_th,
                               timesteps_limit=env_timesteps_phy, sum_rwd=True, df=gam)
        if at == 'phy':
            # Obtain the baselines
            print("Obtaining random baseline, for ", at, " with ", str(nat), " ASs")
            info_random = env.run_baseline('random', num_episodes=nr_episodes)
            print("Obtaining grid baseline, for ", at, " with ", str(nat), " ASs")
            info_grid = env.run_baseline('grid', num_episodes=nr_episodes)
            print("Obtaining always attack baseline, for ", at, " with ", str(nat), " ASs")
            info_always = env.run_baseline('always_attack', num_episodes=nr_episodes)
            print("Obtaining never attack baseline, for ", at, " with ", str(nat), " ASs")
            info_never = env.run_baseline('never_attack', num_episodes=nr_episodes)
            data = {"info_random": info_random, "info_grid": info_grid, "info_always": info_always,
                    "info_never": info_never}

            # Obtain fc error if there is no attack!!
            env = attack.AttackEnv(nr_agents=0, nr_agents_total=nr_sensors[nr_at.index(nat)],
                                   obs_mode='sum_obs_multi', attack_mode=at,
                                   obs_radius=5000, world_size=1000, phy_agent_memory=5, lambda_phy=at_th,
                                   K=5, L=1000, lambda_mac=at_th, timesteps_limit=env_timesteps_phy, sum_rwd=True,
                                   df=gam)
            fc_error = np.zeros(nr_episodes)
            for ep in range(nr_episodes):
                print("FC error: Running episode ", ep + 1, " of ", nr_episodes)
                if at == 'phy':
                    _ = env.reset()
                    done = False
                    while not done:
                        _, _, done, info = env.step([])
                fc_error[ep] = info["phy_fc_error_rate"]
            data["fc_error"] = fc_error

        elif at == 'mac':
            # Obtain the baselines
            print("Obtaining random baseline, for ", at, " with ", str(nat), " ASs")
            info_random = env.run_baseline('random', num_episodes=nr_episodes)
            print("Obtaining grid baseline, for ", at, " with ", str(nat), " ASs")
            info_grid = env.run_baseline('grid', num_episodes=nr_episodes)
            print("Obtaining grid baseline, for ", at, " with ", str(nat), " ASs")
            info_always = env.run_baseline('always_attack', num_episodes=nr_episodes)
            print("Obtaining grid baseline, for ", at, " with ", str(nat), " ASs")
            info_never = env.run_baseline('never_attack', num_episodes=nr_episodes)
            data = {"info_random": info_random, "info_grid": info_grid, "info_always": info_always,
                    "info_never": info_never}
        else:
            raise RuntimeError('Attack not recognized, ' + str(at))
        data_output = os.path.normpath(base_dir + '/data_baseline_' + at + '_' + str(nat) + '.pickle')
        # Write data output
        with open(data_output, 'wb') as handle:
            pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

    for at in ['mac', 'phy']:

        # Common parameters
        me_v = ["me", "no_me", "no_com", "mean"]
        nr_seed = 10
        comm_range = 5000  # Large enough to cover all sensors!!
        at_th = 0.5  # Reputation threshold parameter
        save_flag = True
        base_dir = os.path.normpath(os.getcwd() + '/logger_randomized')
        batch_size = 10000
        nr_episodes = 50  # Nr of episodes to run per seed during policy evaluation!
        replace = False  # Set to True if already stored values should be deleted
        gam = 0.995  # Discount factor
        nr_at = [1, 3, 10]
        nr_sensors = [11, 13, 21]
        trpo_iterations = 500

        env_timesteps_phy = 250

        # TRAINING LOOP!!!
        num_cores = 1 # Adjust to the number of threads available in your machine
        _ = Parallel(n_jobs=num_cores, verbose=5)\
            (delayed(processinput)(at=at, me=me, seed=seed, nat_idx=nat_idx)
             for me in me_v for seed in range(nr_seed) for nat_idx in range(len(nr_at)))

        # POLICY CHECK LOOP!!!
        _ = Parallel(n_jobs=num_cores, verbose=5)\
            (delayed(processinput_policy)(at=at, me=me, seed=seed, nat=nat)
             for me in me_v for seed in range(nr_seed) for nat in nr_at)

        # BASELINES LOOP!!
        _ = Parallel(n_jobs=num_cores, verbose=5) \
            (delayed(processinput_baselines)(at=at, nat=nat) for nat in nr_at)
