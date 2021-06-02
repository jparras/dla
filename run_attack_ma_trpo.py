import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"  # To run everything on CPU
from mpi4py import MPI
from deep_rl_for_swarms.common import logger
from deep_rl_for_swarms.policies import mlp_mean_embedding_policy
from deep_rl_for_swarms.policies import mlp_multi_mean_embedding_policy
from deep_rl_for_swarms.ma_envs.envs.point_envs import attack as attack
import dill
import pickle
import sys
from deep_rl_for_swarms.rl_algo.trpo_mpi import trpo_mpi_attack

def policy_me_multi(name, ob_space, ac_space, index=None):
    return mlp_multi_mean_embedding_policy.MlpPolicy(name=name, ob_space=ob_space, ac_space=ac_space,
                                                     hid_size=[256, 256], feat_size=[[256], [256]], index=index)


def policy_mean(name, ob_space, ac_space, index=None):
    return mlp_multi_mean_embedding_policy.MlpPolicy(name=name, ob_space=ob_space, ac_space=ac_space,
                                                     hid_size=[256, 256], feat_size=[[], []], index=index)


def policy_no_me(name, ob_space, ac_space, index=None):
    return mlp_mean_embedding_policy.MlpPolicy_No_Mean_Emmbedding(name=name, ob_space=ob_space, ac_space=ac_space,
                                                                  hid_size=[256, 256], feat_size=[256], index=index)


def train_trpo(log_dir=None, attack_mode=None, mean_embedding="me", save_flag=False, plot_flag=False,
               nat=1, ns=10, comm_range=0, trpo_iterations=100, batch_size=500, env_timesteps_phy=50,
               at_th=-1, gamma=0.995):

    import deep_rl_for_swarms.common.tf_util as U

    sess = U.single_threaded_session()
    sess.__enter__()

    rank = MPI.COMM_WORLD.Get_rank()
    if rank == 0:
        logger.configure(format_strs=['csv'], dir=log_dir)
    else:
        logger.configure(format_strs=[])
        logger.set_level(logger.DISABLED)

    if mean_embedding == "no_com":  # Do not sum the rewards!!
        sum_rwd = False
        comm_range = 0  # Make sure that there are NO OBSERVATIONS
    else:
        sum_rwd = True

    env = attack.AttackEnv(nr_agents=nat, nr_agents_total=ns, obs_mode='sum_obs_multi', attack_mode=attack_mode,
                           obs_radius=comm_range, world_size=1000, phy_agent_memory=5, lambda_phy=at_th,
                           K=5, L=1000, lambda_mac=at_th, timesteps_limit=env_timesteps_phy, sum_rwd=sum_rwd,
                           df=gamma)

    if save_flag:
        os.makedirs(os.path.normpath(log_dir + "/models_trpo"), exist_ok=True)
        with open(os.path.normpath(log_dir + '/models_trpo/env.pkl'), 'wb') as file:
            dill.dump(env, file)

    # TRPO parameters
    max_kl = 0.01
    cg_iters = 10
    cg_damping = 0.1
    vf_iters = 5
    vf_stepsize = 1e-3
    lam = 0.98

    if mean_embedding == "me":
        trpo_mpi_attack.learn(env, policy_me_multi, timesteps_per_batch=batch_size, max_kl=max_kl,
                              cg_iters=cg_iters, cg_damping=cg_damping, max_iters=trpo_iterations,
                              gamma=gamma, lam=lam, vf_iters=vf_iters, vf_stepsize=vf_stepsize,
                              save_dir=log_dir, save_flag=save_flag, plot_flag=plot_flag)
    elif mean_embedding == "mean":
        trpo_mpi_attack.learn(env, policy_mean, timesteps_per_batch=batch_size, max_kl=max_kl,
                              cg_iters=cg_iters, cg_damping=cg_damping, max_iters=trpo_iterations,
                              gamma=gamma, lam=lam, vf_iters=vf_iters, vf_stepsize=vf_stepsize,
                              save_dir=log_dir, save_flag=save_flag, plot_flag=plot_flag)
    else:
        trpo_mpi_attack.learn(env, policy_no_me, timesteps_per_batch=batch_size, max_kl=max_kl,
                              cg_iters=cg_iters, cg_damping=cg_damping, max_iters=trpo_iterations,
                              gamma=gamma, lam=lam, vf_iters=vf_iters, vf_stepsize=vf_stepsize,
                              save_dir=log_dir, save_flag=save_flag, plot_flag=plot_flag)
    env.close()


def main(attack_mode, mean_embedding, save_flag, seed, nat, ns, comm_range, base_dir, trpo_iterations, batch_size,
         env_timesteps_phy, at_th, gamma):

    log_dir = os.path.normpath(base_dir + '/' + attack_mode + '_' + mean_embedding + '_' + str(nat) + '/' + str(seed))
    train_trpo(log_dir=log_dir, attack_mode=attack_mode, mean_embedding=mean_embedding,
               save_flag=save_flag, nat=nat, ns=ns, comm_range=comm_range, trpo_iterations=trpo_iterations,
               batch_size=batch_size, env_timesteps_phy=env_timesteps_phy, at_th=at_th, gamma=gamma)


if __name__ == '__main__':

    # Read values for the simulation
    hash = sys.argv[1]
    data_input = 'data' + hash + '.pickle'
    with open(data_input, 'rb') as handle:
        data = pickle.load(handle)
    at = data["at"]
    me = data["me"]
    seed = data["seed"]
    save_flag = data["save_flag"]
    nat = data["nat"]
    ns = data["ns"]
    comm_range = data["comm_range"]
    base_dir = data["base_dir"]
    trpo_iterations = data["trpo_iterations"]
    batch_size = data["batch_size"]
    env_timesteps_phy = data["env_timesteps_phy"]
    at_th = data["at_th"]
    gamma = data['gamma']

    main(at, me, save_flag, seed, nat, ns, comm_range, base_dir, trpo_iterations, batch_size, env_timesteps_phy, at_th,
         gamma)
