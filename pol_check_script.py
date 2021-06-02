import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"  # To run everything on CPU
from deep_rl_for_swarms.common.act_wrapper import ActWrapper
from deep_rl_for_swarms.policies import mlp_mean_embedding_policy
from deep_rl_for_swarms.policies import mlp_multi_mean_embedding_policy
from deep_rl_for_swarms.ma_envs.envs.point_envs import attack as attack
import pickle
from gym import spaces
import itertools


def policy_me_multi(name, ob_space, ac_space, index=None):
    return mlp_multi_mean_embedding_policy.MlpPolicy(name=name, ob_space=ob_space, ac_space=ac_space,
                                                     hid_size=[256, 256], feat_size=[[256], [256]], index=index)


def policy_mean(name, ob_space, ac_space, index=None):
    return mlp_multi_mean_embedding_policy.MlpPolicy(name=name, ob_space=ob_space, ac_space=ac_space,
                                                     hid_size=[256, 256], feat_size=[[], []], index=index)


def policy_no_me(name, ob_space, ac_space, index=None):
    return mlp_mean_embedding_policy.MlpPolicy_No_Mean_Emmbedding(name=name, ob_space=ob_space, ac_space=ac_space,
                                                                  hid_size=[256, 256], feat_size=[256], index=index)


def policy_check(me, seed, comm_range, base_dir, at, nr_at, trpo_iterations, env_timesteps_phy, at_th, gamma):
    save_dir = os.path.normpath(base_dir + '/' + at + '_' + me + '_' + str(nr_at) + '/' + str(seed))

    # Create environment and load policy

    pol_dir = os.path.normpath(save_dir + '/models_trpo/model_{}.pkl'.format(trpo_iterations))

    if me == "no_com":  # Do not sum the rewards!!
        sum_rwd = False
        comm_range = 0  # Make sure that there are NO OBSERVATIONS
    else:
        sum_rwd = True

    env = attack.AttackEnv(nr_agents=nr_at, nr_agents_total=nr_sensors, obs_mode='sum_obs_multi',
                           attack_mode=at,
                           obs_radius=comm_range, world_size=1000, phy_agent_memory=5, lambda_phy=at_th,
                           K=5, L=1000, lambda_mac=at_th, timesteps_limit=env_timesteps_phy, sum_rwd=sum_rwd,
                           df=gamma)

    ob_space = env.observation_space
    ac_space = env.action_space

    if me == "me":
        pol = policy_me_multi("pi", ob_space, ac_space)
    elif me == 'mean':
        pol = policy_mean("pi", ob_space, ac_space)
    else:
        pol = policy_no_me("pi", ob_space, ac_space)

    act_params = {'name': "pi", 'ob_space': ob_space, 'ac_space': ac_space}
    pi = ActWrapper(pol, act_params)
    pi.load(pol_dir, pol)
    # Load baseline data (if available) for plot in phy case
    if at == 'phy':
        dir = os.path.normpath(base_dir + '/data_baseline_' + at + '_' + str(nr_at) + '.pickle')
        with open(dir, 'rb') as handle:
            data = pickle.load(handle)
        ac_baseline = data['info_grid'][0]['actions'][0]
    # Simulate episodes
    for ep in range(nr_episodes):
        ob = env.reset()
        done = False
        r = 0
        while not done:
            ac, _ = pi.act(True, np.vstack(ob))
            #ac = np.ones(5)
            if isinstance(env.action_space, spaces.Box):
                ac = np.clip(ac, env.action_space.low, env.action_space.high)
            ob, rew, done, info_indiv = env.step(ac)
            r += np.mean(rew)
        # Analyse results
        if at == 'phy':
            env.phy_plot(save=False, show=True,
                         title=str(at) + '_' + str(me) + '_' + str(ep) + '_' + str(env.lambda_phy) + '_' + str(r))
        elif at == 'mac':
            env.mac_plot(save=False, show=True,
                         title=str(at) + '_' + str(me) + '_' + str(ep) + '_' + str(env.lambda_mac) + '_' + str(r))


if __name__ == '__main__':

    # This script uses the policy result for the attack

    def filter_seeds(nat, at):  # Filter seeds if required
        filtered_seeds = np.zeros([len(me_v), good_seeds], dtype=int)
        if good_seeds < nr_seed:
            # Obtain seeds that provide the best reward
            dir = os.path.normpath(base_dir + '/' + at + '_' + me_v[0] + '_' + str(nat) + '/' + str(0) +
                                   '/data_policy.pickle')
            with open(dir, 'rb') as handle:
                data = pickle.load(handle)
            nrep = data['nr_episodes']
            rwd = np.zeros([len(me_v), nr_seed, nrep])

            for seed, me in itertools.product(range(nr_seed), range(len(me_v))):
                dir = os.path.normpath(base_dir + '/' + at + '_' + me_v[me] + '_' + str(nat) + '/' + str(seed) +
                                       '/data_policy.pickle')
                with open(dir, 'rb') as handle:
                    data = pickle.load(handle)
                for rep in range(nrep):
                    rwd[me, seed, rep] = data['info'][rep]['mean_total_rwd']

            for me in range(len(me_v)):
                rewards_mean = np.mean(rwd[me, :, :], axis=1)  # Mean reward for each seed
                filtered_seeds[me, :] = np.argsort(rewards_mean)[-good_seeds:]
        else:
            for me in range(len(me_v)):
                filtered_seeds[me, :] = np.array(range(nr_seed))
        return filtered_seeds


    # Initial parameters
    at = "phy"
    me_v = ['me']
    nr_at = 5
    nr_sensors = 15
    trpo_iterations = 1000
    env_timesteps_phy = 250
    base_dir = os.path.normpath(os.getcwd() + '/logger_randomized')
    nr_episodes = 5
    comm_range = 5000
    at_th = 0.5
    gamma = 0.995

    nr_seed = 10
    good_seeds = 1
    data = {}
    seeds = filter_seeds(nr_at, at)

    for me in me_v:
        seed = seeds[me_v.index(me), 0]
        policy_check(me, seed, comm_range, base_dir, at, nr_at, trpo_iterations, env_timesteps_phy, at_th, gamma)

