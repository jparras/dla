import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import itertools
from scipy.stats import ttest_ind as t_test
from tikzplotlib import save as tikz_save
from deep_rl_for_swarms.ma_envs.envs.point_envs import attack as attack
import pickle


if __name__ == '__main__':

    def data_plot(input_raw, title, xlabel, ylabel, zero=False, lines=None, save=False, test=False, iters=500):

        color = ['b', 'k', 'r', 'g']

        # Smooth out curve
        l_conv = 30
        input = np.zeros([max(input_raw.shape[0], l_conv) - min(input_raw.shape[0], l_conv) + 1,
                          input_raw.shape[1], input_raw.shape[2]])
        for i1 in range(input.shape[1]):
            for i2 in range(input.shape[2]):
                input[:, i1, i2] = np.convolve(input_raw[:, i1, i2], np.ones(l_conv) / l_conv, mode='valid')
        # Downsample
        n_samples = 50
        if input.shape[0] > n_samples: # Downsample if file is too large!
            ds = int(input.shape[0] / n_samples)
            input = input[::ds, :, :]
        t = np.linspace(0, iters, input.shape[0])
        alpha_value = 0.3
        for i in range(len(me_v)):
            mean = np.mean(input[:, :, i], axis=1)
            std = np.std(input[:, :, i], axis=1)
            plt.plot(t, mean, color[i])
            plt.fill_between(t, mean - std, mean + std,
                             alpha=alpha_value, edgecolor=None, facecolor=color[i], label=me_v[i])

        if lines is not None:
            num_lines = len(lines)
            #plt.plot(t, lines*np.ones([len(t), num_lines]), 'k')
            for i in range(num_lines):
                plt.plot(t, lines[i] * np.ones(len(t)))
            '''
            plt.plot(t, lines[0] * np.ones(len(t)), 'k')
            plt.plot(t, lines[1] * np.ones(len(t)), 'k--')
            if num_lines == 3:
                plt.plot(t, lines[2] * np.ones(len(t)), 'cyan')
            '''

        if zero:  # Add a line at 0
            plt.plot(t, np.zeros(t.shape), 'k')

        #plt.legend(loc='best')

        if save:
            tikz_save(title + '.tikz') #, figureheight='\\figureheight', figurewidth='\\figurewidth')
            #plt.savefig(title + '.pdf')
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.show()

        data = np.around(np.mean(input[-1, :, :], axis=0), decimals=2)
        dstd = np.around(np.std(input[-1, :, :], axis=0), decimals=2)
        out_msg = title + ": "
        for i in range(len(me_v)):
            out_msg = out_msg + me_v[i] + ' = ' + str(data[i]) + ' +- ' + str(dstd[i]) + '; '
        if lines is not None:
            for i in range(num_lines):
                out_msg = out_msg + ' line ' + str(i + 1) + ' = ' + str(lines[i]) + '; '


        print(out_msg)
        if test:
            # Obtain all t_tests
            for t1 in range(len(me_v)):
                out_msg = "T1: " + me_v[t1]
                for t2 in range(len(me_v)):
                    d1 = input[-1, :, t1].flatten()
                    d2 = input[-1, :, t2].flatten()
                    t, p = t_test(d1, d2, equal_var=False)  # Welch test: t test with unequal variances
                    out_msg = out_msg + ' T2: ' + me_v[t2] + ' --> p = ' + str(p)
                print(out_msg)

    def extract_info(vals, title, comp=None, comp_titles=None):

        mean = np.around(np.mean(vals, axis=1), decimals=2)
        std = np.around(np.std(vals, axis=1), decimals=2)
        out_msg = "RESULTS for " + title + ": "
        for i in range(len(me_v)):
            out_msg = out_msg + me_v[i] + ' = ' + str(mean[i]) + ' +- ' + str(std[i]) + '; '
        if comp is None:
            pass
        else:
            for i in range(len(comp)):
                out_msg = out_msg + "; " + comp_titles[i] + ' = ' + str(comp[i])
        print(out_msg)
        # Welch test
        for t1 in range(len(me_v)):
            out_msg = "T1: " + me_v[t1]
            for t2 in range(len(me_v)):
                d1 = vals[t1, :].flatten()
                d2 = vals[t2, :].flatten()
                t, p = t_test(d1, d2, equal_var=False)  # Welch test: t test with unequal variances
                out_msg = out_msg + ' T2: ' + me_v[t2] + ' --> p = ' + str(p)
            print(out_msg)

    def obtain_ptt_theoretical(n):  # Use Biachi's equations to obtain theoretical Ptt
        w1 = 1
        m = 10  # Max backoff stage is 2 ** 10
        from scipy.optimize import fsolve

        def equations(p):
            tau1, p1 = p
            return tau1 - 2/(1 + w1 + 2 * w1 * sum([(2 * p1) ** j for j in range(m-1)])), p1 - 1 + (1 - tau1) ** (n-1)

        tau1, p1 = fsolve(equations, (0, 0))

        Ptr = 1 - (1 - tau1) ** n
        Ps1 = tau1 * (1 - tau1) ** (n - 1)
        Pc = Ptr - n * Ps1

        # Obtain duration of a time slot
        env = attack.AttackEnv(nr_agents=0, nr_agents_total=n, attack_mode='mac')
        _ = env.reset()
        Ts = 1  # Countdown value
        Tt = env.t_tx
        Tc = env.t_col

        si = Ps1 * env.fr_size / ((1-Ptr) * Ts + (n * Ps1) * Tt + Tc * Pc)

        return n * si

    def info_training(nat, at, filtered_seeds):
        print("\n TRAINING RESULTS FOR ATTACK ", at, " WITH ", nat, " ATTACKING SENSORS \n")
        # Create arrays to store values
        reward = np.zeros((trpo_iterations[at], good_seeds, len(me_v)))
        ag_caught = np.zeros((trpo_iterations[at], good_seeds, len(me_v)))
        ag_no_caught = np.zeros((trpo_iterations[at], good_seeds, len(me_v)))
        time = np.zeros((trpo_iterations[at], good_seeds, len(me_v)))

        if at == "phy":
            fc_err = np.zeros((trpo_iterations[at], good_seeds, len(me_v)))

        if at == "mac":
            number_of_tx = np.zeros((trpo_iterations[at], good_seeds, len(me_v)))
            number_of_col = np.zeros((trpo_iterations[at], good_seeds, len(me_v)))
            tx_prop_time_total = np.zeros((trpo_iterations[at], good_seeds, len(me_v)))
            total_bits_tx = np.zeros((trpo_iterations[at], good_seeds, len(me_v)))
            tx_prop_at = np.zeros((trpo_iterations[at], good_seeds, len(me_v)))
            tx_prop_normal = np.zeros((trpo_iterations[at], good_seeds, len(me_v)))

        for seed, mean_embedding in itertools.product(range(good_seeds), me_v):

            def assign_data(key):
                data = data_seed[key].values
                if len(data) >= trpo_iterations[at]:
                    data = data[0:trpo_iterations[at]]
                else:
                    data = np.pad(data, (0, trpo_iterations[at] - len(data)), 'constant')
                return data

            # Extract training info
            data_seed = pd.read_csv(os.path.normpath(root_dir + '/' + at + '_' + mean_embedding + '_' + str(nat) + '/'
                                                     + str(filtered_seeds[me_v.index(mean_embedding), seed])
                                                     + '/progress.csv'))
            idm = me_v.index(mean_embedding)
            reward[:, seed, idm] = assign_data("MtR")
            time[:, seed, idm] = assign_data("TimeElapsed")
            time[:, seed, idm] = time[:, seed, idm] - np.concatenate((np.zeros(1), time[:-1, seed, idm]))
            ag_caught[:, seed, idm] = assign_data("AttC")
            ag_no_caught[:, seed, idm] = assign_data("AttNC")

            if at == "phy":
                fc_err[:, seed, idm] = assign_data("Fce")

            if at == "mac":
                number_of_tx[:, seed, idm] = assign_data("Tmt")
                number_of_col[:, seed, idm] = assign_data("Tmc")
                tx_prop_time_total[:, seed, idm] = assign_data("Ptt")
                total_bits_tx[:, seed, idm] = assign_data("Tbt")
                tx_prop_at[:, seed, idm] = assign_data("MpbtA")
                tx_prop_normal[:, seed, idm] = assign_data("MpbtN")

        # Results
        if data_baselines is None:
            data_plot(reward, "total_rwd_" + at + '_' + str(nat), "TRPO Iteration", "Total reward", zero=False,
                      iters=trpo_iterations[at])
            data_plot(100 * ag_caught / (ag_caught + ag_no_caught), "at_disc_" + at + '_' + str(nat),
                      "TRPO Iteration", "Prop Ag caught", zero=False, iters=trpo_iterations[at])
        else:
            data_plot(reward, "total_rwd_" + at + '_' + str(nat), "TRPO Iteration", "Total reward", zero=False,
                      lines=data_baselines[at_v.index(at), nr_at.index(nat), 0:3], iters=trpo_iterations[at])
            data_plot(100 * ag_caught / (ag_caught + ag_no_caught), "at_disc_" + at + '_' + str(nat),
                      "TRPO Iteration", "Prop Ag caught", zero=False,
                      lines=data_baselines[at_v.index(at), nr_at.index(nat), 3:6], iters=trpo_iterations[at])

        if at == "phy":
            if data_baselines is None:
                data_plot(fc_err * 100, "primary_detection_" + at + '_' + str(nat),
                          "TRPO Iteration", "FC total error", iters=trpo_iterations[at])
            else:
                data_plot(fc_err * 100, "primary_detection_" + at + '_' + str(nat),
                          "TRPO Iteration", "FC total error",
                          lines=data_baselines[at_v.index(at), nr_at.index(nat), 6:11], iters=trpo_iterations[at])
        '''
        if at == "mac":
            tx_prop_normal_agents = (nr_sensors[nr_at.index(nat)] - nat) / nr_sensors[nr_at.index(nat)]
            data_plot(tx_prop_normal * total_bits_tx / 100, "mac_bits_tx_" + at + '_' + str(nat),
                      "TRPO Iteration", "NS bits tx",
                      lines=[tx_prop_normal_agents * ptt_th[nr_at.index(nat)] * t_mac_max / Rb])
        '''

    def info_trained(nat, at, filtered_seeds):  # Obtains info of the trained and stored NN values
        print("\n TRAINED RESULTS FOR ATTACK ", at, " WITH ", nat, " ATTACKING SENSORS \n")
        # Create list to store values
        info = [[] for _ in me_v]  # info[me][seed]['info'][repetition]['key'] to access to 'key' of each value!
        for seed, me in itertools.product(range(nr_seed), me_v):
            dir = os.path.normpath(root_dir + '/' + at + '_' + me + '_' + str(nat) + '/' + str(seed) +
                                   '/data_policy.pickle')
            with open(dir, 'rb') as handle:
                data = pickle.load(handle)
            info[me_v.index(me)].append(data)

        # Obtain reward values
        vals = [[] for _ in me_v]
        for me in range(len(me_v)):
            for seed, rep in itertools.product(list(filtered_seeds[me, :]), range(info[0][0]['nr_episodes'])):
                vals[me].append(info[me][seed]['info'][rep]['mean_total_rwd'])
        vals = np.array(vals)  # me x samples matrix
        if data_baselines is None:
            extract_info(vals, 'mean_total_rwd')
        else:
            extract_info(vals, 'mean_total_rwd', comp=data_baselines[at_v.index(at), nr_at.index(nat), 0: 3],
                         comp_titles=baselines_name)

        # Obtain agent discovered values
        vals = [[] for _ in me_v]
        for me in range(len(me_v)):
            for seed, rep in itertools.product(list(filtered_seeds[me, :]), range(info[0][0]['nr_episodes'])):
                vals[me].append(info[me][seed]['info'][rep]['attackers_caught'])
        vals = 100 * np.array(vals) / nat  # me x samples matrix
        extract_info(vals, 'caught_as', comp=data_baselines[at_v.index(at), nr_at.index(nat), 3:6],
                     comp_titles=baselines_name)

        if at == 'phy':
            # Obtain agent discovered values
            vals = [[] for _ in me_v]
            for me in range(len(me_v)):
                for seed, rep in itertools.product(list(filtered_seeds[me, :]), range(info[0][0]['nr_episodes'])):
                    vals[me].append(info[me][seed]['info'][rep]['phy_fc_error_rate'])
            vals = 100 * np.array(vals)
            if data_baselines is None:
                extract_info(vals, 'primary_detection')
            else:
                extract_info(vals, 'phy_fc_error_rate', comp=data_baselines[at_v.index(at), nr_at.index(nat), 6:10],
                             comp_titles=baselines_name + ['no_attack'])

        if at == 'mac':
            # Obtain total bits tx values
            vals = [[] for _ in me_v]
            for me in range(len(me_v)):
                for seed, rep in itertools.product(list(filtered_seeds[me, :]), range(info[0][0]['nr_episodes'])):
                    vals[me].append(info[me][seed]['info'][rep]['total_bits_tx'] *
                                    info[me][seed]['info'][rep]['mean_prop_bits_tx_no'] / 100)
            vals = np.array(vals) / 1e3  # me x samples matrix
            if data_baselines is None:
                extract_info(vals, 'total_bits_tx_normal_sensors')
            else:
                extract_info(vals, 'total_bits_tx_normal_sensors', comp=data_baselines[at_v.index(at), nr_at.index(nat), 10:],
                             comp_titles=baselines_name)
            tx_prop_normal_agents = (nr_sensors[nr_at.index(nat)] - nat) / nr_sensors[nr_at.index(nat)]
            print("Theoretical total bits tx if no attack: ",
                  tx_prop_normal_agents * ptt_th[nr_at.index(nat)] * t_mac_max / (1e3 * Rb))

    def filter_seeds(nat, at):  # Filter seeds if required
        filtered_seeds = np.zeros([len(me_v), good_seeds], dtype=int)
        if good_seeds < nr_seed:
            # Obtain seeds that provide the best reward
            dir = os.path.normpath(root_dir + '/' + at + '_' + me_v[0] + '_' + str(nat) + '/' + str(0) +
                                   '/data_policy.pickle')
            with open(dir, 'rb') as handle:
                data = pickle.load(handle)
            nrep = data['nr_episodes']
            rwd = np.zeros([len(me_v), nr_seed, nrep])

            for seed, me in itertools.product(range(nr_seed), range(len(me_v))):
                dir = os.path.normpath(root_dir + '/' + at + '_' + me_v[me] + '_' + str(nat) + '/' + str(seed) +
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
        print('Filtered seeds for ', at, ' are ', filtered_seeds)
        return filtered_seeds

    # Select main directory
    root_dir = os.path.normpath(os.getcwd() + '/logger_randomized')
    # If we want to save figures: go to data_plot and set plot=True in the parameters!
    t_mac_max = 5e5  # us of the mac simulation
    Rb = 1  # Mbps
    at_v = ['mac', 'phy']
    nr_at = [1, 3, 10]
    nr_sensors = [11, 13, 20]
    trpo_iterations = {"phy": 500, "mac": 500}
    me_v = ["me", "mean", "no_me", "no_com"]
    nr_seed = 10
    good_seeds = 1
    env_timesteps_phy = 250

    # Load baselines values if available
    if os.path.isfile(os.path.normpath(root_dir + '/data_baseline_' + at_v[0] + '_' + str(nr_at[0]) + '.pickle')):
        data_baselines = np.zeros([len(at_v), len(nr_at), 13])
        baselines_name = ["baseline random", "baseline_always", "baseline_never"]
        for at, nat in itertools.product(at_v, nr_at):
            dir = os.path.normpath(root_dir + '/data_baseline_' + at + '_' + str(nat) + '.pickle')
            with open(dir, 'rb') as handle:
                data = pickle.load(handle)
            data_baselines[at_v.index(at), nr_at.index(nat), 0] = np.mean([data['info_random'][i]['mean_total_rwd']
                                                                           for i in range(len(data['info_random']))])
            data_baselines[at_v.index(at), nr_at.index(nat), 1] = np.mean([data['info_always'][i]['mean_total_rwd']
                                                                           for i in range(len(data['info_always']))])
            data_baselines[at_v.index(at), nr_at.index(nat), 2] = np.mean([data['info_never'][i]['mean_total_rwd']
                                                                           for i in range(len(data['info_never']))])

            data_baselines[at_v.index(at), nr_at.index(nat), 3] = 100 * np.mean([data['info_random'][i]['attackers_caught'] / (data['info_random'][i]['attackers_caught'] + data['info_random'][i]['attackers_not_caught']) for i in range(len(data['info_random']))])
            data_baselines[at_v.index(at), nr_at.index(nat), 4] = 100 * np.mean([data['info_always'][i]['attackers_caught'] / (data['info_always'][i]['attackers_caught'] + data['info_always'][i]['attackers_not_caught']) for i in range(len(data['info_always']))])
            data_baselines[at_v.index(at), nr_at.index(nat), 5] = 100 * np.mean([data['info_never'][i]['attackers_caught'] / (data['info_never'][i]['attackers_caught'] + data['info_never'][i]['attackers_not_caught']) for i in range(len(data['info_never']))])

            if at == 'phy':
                data_baselines[at_v.index(at), nr_at.index(nat), 6] = 100 * np.mean(
                    [data['info_random'][i]['phy_fc_error_rate'] for i in range(len(data['info_random']))])
                data_baselines[at_v.index(at), nr_at.index(nat), 7] = 100 * np.mean(
                    [data['info_always'][i]['phy_fc_error_rate'] for i in range(len(data['info_always']))])
                data_baselines[at_v.index(at), nr_at.index(nat), 8] = 100 * np.mean(
                    [data['info_never'][i]['phy_fc_error_rate'] for i in range(len(data['info_never']))])

                data_baselines[at_v.index(at), nr_at.index(nat), 9] = 100 * np.mean(data["fc_error"])
            if at == 'mac':
                data_baselines[at_v.index(at), nr_at.index(nat), 10] = np.mean(
                    [data['info_random'][i]['total_bits_tx'] * data['info_random'][i]['mean_prop_bits_tx_no'] / 100 for
                     i in range(len(data['info_random']))]) / 1e3
                data_baselines[at_v.index(at), nr_at.index(nat), 11] = np.mean(
                    [data['info_always'][i]['total_bits_tx'] * data['info_always'][i]['mean_prop_bits_tx_no'] / 100 for
                     i in range(len(data['info_always']))]) / 1e3
                data_baselines[at_v.index(at), nr_at.index(nat), 12] = np.mean(
                    [data['info_never'][i]['total_bits_tx'] * data['info_never'][i]['mean_prop_bits_tx_no'] / 100 for
                     i in range(len(data['info_never']))]) / 1e3

    else:
        data_baselines = None

    # Obtain theoretical tx proportion
    ptt_th = []
    for n in nr_sensors:
        ptt_th.append(obtain_ptt_theoretical(n))

    # Show results
    for nat, at in itertools.product(nr_at, at_v):
        filtered_seeds = filter_seeds(nat, at)
        info_training(nat, at, filtered_seeds)  # Values during training

    for nat, at in itertools.product(nr_at, at_v):
        filtered_seeds = filter_seeds(nat, at)
        info_trained(nat, at, filtered_seeds)  # Values after training
