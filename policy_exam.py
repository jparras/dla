import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"  # To run everything on CPU
from deep_rl_for_swarms.common.act_wrapper import ActWrapper
from deep_rl_for_swarms.policies import mlp_mean_embedding_policy
from deep_rl_for_swarms.policies import mlp_multi_mean_embedding_policy
from deep_rl_for_swarms.ma_envs.envs.point_envs import attack as attack
import sys
import pickle
from gym import spaces


def policy_me_multi(name, ob_space, ac_space, index=None):
    return mlp_multi_mean_embedding_policy.MlpPolicy(name=name, ob_space=ob_space, ac_space=ac_space,
                                                     hid_size=[256, 256], feat_size=[[256], [256]], index=index)


def policy_mean(name, ob_space, ac_space, index=None):
    return mlp_multi_mean_embedding_policy.MlpPolicy(name=name, ob_space=ob_space, ac_space=ac_space,
                                                     hid_size=[256, 256], feat_size=[[], []], index=index)


def policy_no_me(name, ob_space, ac_space, index=None):
    return mlp_mean_embedding_policy.MlpPolicy_No_Mean_Emmbedding(name=name, ob_space=ob_space, ac_space=ac_space,
                                                                  hid_size=[256, 256], feat_size=[256], index=index)


if __name__ == '__main__':

    hash = sys.argv[1]
    data_input = 'data' + hash + '.pickle'
    with open(data_input, 'rb') as handle:
        data = pickle.load(handle)
    at = data["at"]
    me = data["me"]
    seed = data["seed"]
    nr_at = data["nr_at"]
    nr_sensors = data["nr_sensors"]
    comm_range = data["comm_range"]
    trpo_iterations = data["trpo_iterations"]
    env_timesteps_phy = data["env_timesteps_phy"]
    base_dir = data["base_dir"]
    nr_episodes = data["nr_episodes"]
    at_th = data["at_th"]
    gamma = data['gamma']

    save_dir = os.path.normpath(base_dir + '/' + at + '_' + me + '_' + str(nr_at) + '/' + str(seed))

    # Create environment and load policy

    pol_dir = os.path.normpath(save_dir + '/models_trpo/model_{}.pkl'.format(trpo_iterations))

    if me == "no_com":  # Do not sum the rewards!!
        sum_rwd = False
        comm_range = 0  # Make sure that there are NO OBSERVATIONS
    else:
        sum_rwd = True

    env = attack.AttackEnv(nr_agents=nr_at, nr_agents_total=nr_sensors, obs_mode='sum_obs_multi', attack_mode=at,
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
    info = []
    # Simulate episodes
    for ep in range(nr_episodes):
        ob = env.reset()
        done = False
        while not done:
            ac, _ = pi.act(True, np.vstack(ob))
            if isinstance(env.action_space, spaces.Box):
                ac = np.clip(ac, env.action_space.low, env.action_space.high)
            ob, rew, done, info_indiv = env.step(ac)

        # Analyse results
        info.append(info_indiv)

    data["info"] = info
    data_output = os.path.normpath(save_dir + '/data_policy.pickle')
    # Write data output
    with open(data_output, 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print("Simulation with ", at, " attack, where me is ", me, ", with ", nr_at, " ASs and seed ", seed, ", is done")
