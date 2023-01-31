# -*- coding: utf-8 -*-
# ./adaptation/simulations.py

"""Some simulations for the agent and the task."""
import multiprocessing as mp
from itertools import product
import glob

import numpy as np
from matplotlib import pyplot as plt
import pandas as pd

import model
import task_hold_hand as thh
from pars import task as task_pars
import pars


def plot_contexts(pandata, axis=None, fignum=4):
    """Plots the inferred contexts as well as the true contexts. True contexts
    are plotted as background colors and the posterior over contexts as
    colored lines. The chromatic code is the same for both, but the alpha on
    the true contexts is lower for visual clarity.

    Parameters
    ----------
    pandata : DataFrame
    Data from both the agent and the task, i.e. the output of thh.join_pandas.

    """
    flag_makepretty = False
    if axis is None:
        fig, axis = plt.subplots(1, 1, clear=True, num=fignum)
        flag_makepretty = True
    else:
        plt.show(block=False)
    alpha = 0.1
    cue_range = [2.2, 3.2]
    real_range = [1.1, 2.1]
    infer_range = [0, 1]
    color_list = [(0, 0, 0, alpha), (1, 0, 0, alpha), (1, 1, 0, alpha), (0, 1, 1, alpha),
                  (0, 0, 1, alpha), (0, 1, 0, alpha)]
    con_strings = sorted([column for column in pandata.columns
                          if (column.startswith('con')
                              and not column.startswith('con_'))])
    all_cons = np.concatenate([pandata['ix_context'].unique(),
                               np.arange(len(con_strings))])
    all_cons = np.unique(all_cons)
    colors = {idx: color
              for idx, color in zip(all_cons, color_list)}
    real_con = np.array(pandata['ix_context'])
    con_breaks = np.nonzero(np.diff(real_con))[0] + 1
    con_breaks = np.concatenate([[0], con_breaks, [len(real_con) - 1]])
    cons = np.array([real_con[one_break] for one_break in con_breaks])
    # plot real context
    for c_con, n_con, ix_con in zip(con_breaks, con_breaks[1:], cons):
        axis.fill_between([c_con, n_con], [real_range[0]] * 2,
                          [real_range[1]] * 2,
                          color=colors[ix_con])
    axis.text(x=len(pandata) / 2, y=1.6, s='Real context',
              horizontalalignment='center', verticalalignment='center')
    # ['con{}'.format(idx) for idx in all_cons]
    conx = np.array(pandata.loc[:, con_strings])
    conx = conx * (infer_range[1] - infer_range[0]) + infer_range[0]
    con_breaks = np.nonzero(np.diff(real_con))[0] + 1
    con_breaks = np.concatenate([[0], con_breaks, [len(real_con) - 1]])
    cons = np.array([real_con[one_break] for one_break in con_breaks])

    # plot cues
    real_cues = np.array(pandata['cue'])
    cue_breaks = np.nonzero(np.diff(real_cues))[0] + 1
    cue_breaks = np.concatenate([[0], cue_breaks, [len(real_cues) - 1]])
    cues = np.array([real_cues[one_break] for one_break in cue_breaks])
    for c_cue, n_cue, ix_cue in zip(cue_breaks, cue_breaks[1:], cues):
        axis.fill_between([c_cue, n_cue], *cue_range, color=colors[ix_cue])
    axis.text(x=len(pandata) / 2, y=np.mean(cue_range), s='Cues',
              horizontalalignment='center', verticalalignment='center')
    # plot inferred context
    for ix_con in range(len(con_strings)):
        color = colors[ix_con][:-1] + (1, )
        axis.plot(conx[:, ix_con], color=color)
    axis.text(x=len(pandata) / 2, y=0.5, s='Inferred context',
              horizontalalignment='center', verticalalignment='center')
    # Plot breaks
    for c_break in con_breaks:
        axis.plot([c_break, c_break], [infer_range[0], cue_range[1]],
                  color='black', alpha=0.2)
    if flag_makepretty:
        axis.set_xticks(con_breaks)
        axis.set_yticks([0, 0.5, 1])
        axis.set_title('Context inference')
        axis.set_xlabel('Trial')
        axis.set_ylabel('Prob. of context')
    plt.draw()


def plot_adaptation(pandata, axis=None, fignum=5):
    """Plots inferred magnitudes, hand position and "adaptation".

    Parameters
    ----------
    pandata : DataFrame
    Data from a simulation that contains the following columns:
      'pos(t)' : hand position
      'mag_mu_x' : Estimate of the magnitude of the force in context x,
                   for x = {0, 1, 2}.

    """
    columns = sorted(list(pandata.columns))
    trial = np.arange(len(pandata))
    colors = ['black', 'red', 'blue']
    adapt_color = np.array([174, 99, 164]) / 256
    if axis is None:
        fig, axis = plt.subplots(1, 1, clear=True, num=fignum)
    axis.plot(np.array(pandata['pos(t)']), color='black', alpha=0.4,
              label='Error')
    axis.plot(np.array(pandata['pos(t)'] + pandata['action']),
              color=adapt_color, label='Adaptation')
    magmu = [np.array(pandata[column])
             for column in columns
             if column.startswith('mag_mu')]
    errors = [np.array(pandata[column])
              for column in columns
              if column.startswith('mag_sd')]
    for color_x, error_x, magmu_x in zip(colors, errors, magmu):
        axis.plot(magmu_x, color=color_x, label='{} model'.format(color_x))
        axis.fill_between(trial, magmu_x - error_x, magmu_x + error_x,
                          color=color_x, alpha=0.1)
    magmu = np.array(magmu)
    yrange = np.array([magmu.min(), magmu.max()]) * 1.1
    axis.set_ylim(yrange)
    plt.draw()


def sim_and_plot(agent, pars_task, return_data=False,
                 force_labels=None, axes=None, fignum=6):
    """Simulates the agent playing and makes a nice plot with context inference
    and adaptation and colors everywhere.

    """
    pandata, pandagent, agent = thh.run(agent, pars=pars_task)
    pandota = thh.join_pandas(pandata, pandagent)
    if axes is None:
        fig, axes = plt.subplots(2, 1, num=fignum, clear=True, sharex=True)
    plot_adaptation(pandota, axis=axes[0])
    plot_contexts(pandota, axis=axes[1])

    axes[0].set_ylabel('Adaptation (N)')
    # axes[0].set_yticks(ticks=fake_forces * 1.2)
    if force_labels is not None:
        axes[0].set_yticklabels(force_labels)
    axes[0].legend()

    axes[1].set_xlabel('Trial')
    axes[1].set_ylabel('p(context)', y=0.05, horizontalalignment='left')
    axes[1].set_yticks([0, 0.5, 1])

    if return_data:
        return pandota, agent


def kim_2015(plot=True, axis=None, fignum=11):
    """Simulates an experiment from Kim et al. 2015.

    Parameters
    ----------
    plot : bool
    Whether to simulate and plot. If True, will create the agent
    with the parameters agent_pars and simulate with sim_and_plot().

    Returns
    -------
    task : dictionary
    Parameters for the task. To be used directly as the input for tth.run().

    agent_pars : dictionary
    Parameters for the agent for both tasks. To be used directly with
    agent.RLMeanSD(). Note that it will not work for the other agents.

    """
    task = task_pars.copy()
    task['obs_noise'] = 3
    task['force_noise'] = 0.01 * np.ones(3)
    task['forces'] = [[0, 0], [-1, 40], [1, 40]]
    blocks = [0, 1, 2, 1, 2, 0, 2, 1, 2, 1, 0, 0, 2, 1, 2, 1,
              0, 1, 2, 1, 2, 0, 0, 1, 2, 1, 2, 0, 2, 1, 2, 1, 0]
    contexts = [[idx, 30] for idx in blocks]
    task['context_seq'] = pars.define_contexts(contexts)
    task['breaks'] = np.zeros(len(task['context_seq']))
    task['cues'] = task['context_seq']

    agent_pars = {'angles': [0, -1, 1],
                  'prediction_noise': 0,
                  'cue_noise': 0.0001,
                  'context_noise': 0.1,
                  'force_sds': np.ones(3),
                  'max_force': 60,
                  'hyper_sd': 10,
                  'obs_sd': 3,
                  'all_learn': True,
                  'learn_rate': 10,
                  'sample_action': True}
    if plot:
        agent = model.LRMeanSD(**agent_pars)
        sim_and_plot(agent, task, force_labels=[-40, 0, 40],
                     fignum=fignum)
    return task, agent_pars


def oh_2019(plot=True, axes=None, fignum=10):
    """Simulates the experiments from Oh and Schweighoffer 2019.

    Parameters
    ----------
    plot : bool
    Whether to simulate and plot. If True, will create the agent
    with the parameters agent_pars and simulate with sim_and_plot().

    Returns
    -------
    task_20 : dictionary
    Parameters for the task with adaptation=20. To be used directly
    as the input for tth.run().

    task_10 : dictionary
    Same as --task_20-- but for adaptation=10.

    agent_pars : dictionary
    Parameters for the agent for both tasks. To be used directly with
    agent.RLMeanSD(). Note that it will not work for the other agents.

    """
    task_20 = task_pars.copy()
    task_20['obs_noise'] = 2.8
    task_20['force_noise'] = 1 * np.ones(2)
    task_20['forces'] = [[0, 0], [1, 20]]
    contexts_20 = [[0, 20], [1, 60], [0, 40], [1, 50], [0, 20],
                   [1, 30], [0, 40], [1, 50]]
    task_20['context_seq'] = pars.define_contexts(contexts_20)
    task_20['breaks'] = np.zeros(len(task_20['context_seq']))
    task_20['cues'] = np.zeros(len(task_20['context_seq']), dtype=int)

    agent_pars = {'angles': [0, 0],
                  'cue_noise': 1 / 2,
                  'max_force': 50,
                  'hyper_sd': 5,
                  'obs_sd': 2.8,
                  'context_noise': 0.01,
                  'force_sds': np.ones(2),
                  'prediction_noise': 0.1,
                  'action_sd': 0.1,  # 2,
                  'all_learn': True,
                  'sample_action': True,
                  'learn_rate': 2}

    task_10 = task_20.copy()
    task_10['forces'] = [[0, 0], [1, 10]]
    if plot:
        if axes is None:
            fig, axes = plt.subplots(2, 2, num=fignum, clear=True,
                                     sharex=True, sharey=False)

        agent = model.LRMeanSD(**agent_pars)

        sim_and_plot(agent, task_20, axes=axes[:, 0])
        axes[0, 0].set_title('Adaptation: 20')
        axes[0, 0].set_ylim((-25, 25))
        agent = model.LRMeanSD(**agent_pars)
        sim_and_plot(agent, task_10, axes=axes[:, 1])
        axes[0, 1].set_title('Adaptation: 10')
        axes[0, 1].set_ylim((-25, 25))

    if plot:
        return task_20, task_10, agent_pars, agent
    else:
        return task_20, task_10, agent_pars


def davidson_2004(plot=True, axes=None, fignum=13):
    """Simulates the second experiment in Davidson_Scaling_2004."""
    trials = 160
    task_m8 = task_pars.copy()
    task_m8['obs_noise'] = 1.1  # 0.5
    task_m8['force_noise'] = 0.5 * np.ones(3)
    task_m8['forces'] = [[0, 0], [1, 4], [-1, 4]]  #
    task_m8['context_seq'] = pars.define_contexts([[2, trials],
                                                   [1, trials]])
    task_m8['cues'] = np.zeros(len(task_m8['context_seq']), dtype=int)
    task_m8['breaks'] = np.zeros(len(task_m8['context_seq']), dtype=int)

    task_p8 = task_m8.copy()
    task_p8['forces'] = [[0, 0], [1, 4], [1, 12]]

    task_p12 = task_p8.copy()
    task_p12['forces'] = [[0, 0], [1, 4], [1, 18]]

    task_m12 = task_m8.copy()
    task_m12['forces'] = [[0, 0], [1, 4], [-1, 10]]

    task_noo_m8 = task_m8.copy()
    task_noo_m8['forces'][0][1] = 20
    task_noo_p8 = task_p8.copy()
    task_noo_p8['forces'] = [[0, 20], [1, 4], [1, 12]]

    agent_pars_m8 = {'angles': [0, 4, -4],
                     'cue_noise': 1 / 3,
                     'max_force': 20,
                     'hyper_sd': 1001,
                     'obs_sd': 2.7,  # 2.6
                     'context_noise': 0.01,
                     'force_sds':  0.1 * np.ones(3),
                     'prediction_noise': 0.3,
                     'learn_rate': 1,
                     'all_learn': True,
                     'threshold_learn': 0.2,
                     'sample_action': False,
                     'action_sd': 1.2,
                     'prior_over_contexts': np.array([0.01, 0.01, 0.98])}
    agent_pars_p8 = agent_pars_m8.copy()
    agent_pars_p8['angles'] = [0, 4, 12]
    agent_pars_m12 = agent_pars_m8.copy()
    agent_pars_m12['angles'] = [0, 4, -10]
    agent_pars_p12 = agent_pars_p8.copy()
    agent_pars_p12['angles'] = [0, 4, 18]
    agent_pars_noo_p8 = agent_pars_p8.copy()
    agent_pars_noo_p8['angles'] = [25, 0, 0]
    agent_pars_noo_m8 = agent_pars_m8.copy()
    agent_pars_noo_m8['angles'] = [25, 0, 0]
    tasks = (task_m8, task_p8, task_m12, task_p12, task_noo_m8, task_noo_p8)
    agents = (agent_pars_m8, agent_pars_p8, agent_pars_m12, agent_pars_p12,
              agent_pars_noo_m8, agent_pars_noo_p8)
    if plot:
        if axes is None:
            fig, axes = plt.subplots(len(agents), 2, num=fignum, clear=True,
                                     sharex=True, sharey=False)
        for ix_agent, ag_par in enumerate(agents):
            c_axes = axes[ix_agent, :]
            agent = model.LRMeanSD(**ag_par)
            sim_and_plot(agent, tasks[ix_agent], axes=c_axes)
        return tasks, agents, agent
    return tasks, agents


def vaswani_2013(plot=True, axes=None, fignum=14):
    """Replicates the error clamp weirdness in Vaswani et al 2013,
    for the different groups but without the earlier phases of each
    experimental run.


    Parameters
    ----------
    plot : bool
    Whether to simulate and plot. If True, will create the agent
    with the parameters agent_pars and simulate with sim_and_plot().

    Returns
    -------
    task : dictionary
    Parameters for the task. To be used directly as the input for tth.run().

    agent_pars : dictionary
    Parameters for the agent for both tasks. To be used directly with
    agent.RLMeanSD(). Note that it will not work for the other agents.

    """
    num_trials = 300
    task_common = task_pars.copy()
    task_common['obs_noise'] = 0.1
    task_common['breaks'] = np.zeros(num_trials, dtype=int)
    task_common['cues'] = np.zeros(num_trials, dtype=int)
    task_common['forces'] = [[0, 0], [1, 1], [-1, 1], [-1, 0.5]]
    task_common['force_noise'] = 0.01 * np.ones(4)

    task_1 = task_common.copy()
    contexts = [[1, 100], [pars.CLAMP_INDEX, num_trials - 100]]
    task_1['context_seq'] = pars.define_contexts(contexts)

    task_2 = task_common.copy()
    contexts = [[0, 50], [1, 50], [pars.CLAMP_INDEX, num_trials - 100]]
    task_2['context_seq'] = pars.define_contexts(contexts)

    task_3 = task_common.copy()
    contexts = [[3, 50], [1, 50], [pars.CLAMP_INDEX, num_trials - 100]]
    task_3['context_seq'] = pars.define_contexts(contexts)

    task_4 = task_common.copy()
    contexts = [[2, 100], [pars.CLAMP_INDEX,  num_trials - 100]]
    task_4['context_seq'] = pars.define_contexts(contexts)

    force_sds = 0.1
    agent_pars = {'max_force': 20,
                  'hyper_sd': 1,
                  'obs_sd': 0.1,
                  'context_noise': 0.09,
                  'prediction_noise': 5.0,
                  'action_sd': 0.17,
                  'sample_action': True,
                  'learn_rate': 1}

    agent_1 = agent_pars.copy()
    agent_1['angles'] = [0, 1]
    agent_1['cue_noise'] = 1 / 2
    agent_1['force_sds'] = force_sds * np.ones(2)

    agent_2 = agent_1.copy()
    agent_2['prior_over_contexts'] = np.array([0.8, 0.2])  # np.array([0.505, 0.495])

    agent_3 = agent_pars.copy()
    agent_3['angles'] = [0, 1, -0.5]
    agent_3['cue_noise'] = 1 / 3
    agent_3['force_sds'] = force_sds * np.ones(3)
    agent_3['context_noise'] *= 2 / 3

    agent_4 = agent_1.copy()
    agent_4['angles'] = [0, -1]

    tasks = [task_1, task_2, task_3, task_4]
    agents = [agent_1, agent_2, agent_3, agent_4]
    if plot:
        if axes is None:
            fig, axes = plt.subplots(2, 4, num=fignum, clear=True,
                                     sharex=True, sharey=False)

        for idx, (c_task, c_agent_pars) in enumerate(zip(tasks, agents)):
            agent = model.LRMeanSD(**c_agent_pars)
            sim_and_plot(agent, c_task, axes=axes[:, idx])
            axes[0, idx].set_ylim((-1.2, 1.2))
        return tasks, agents, agent
    return tasks, agents


def model_showoff(plot=True, axes=None, fignum=15):
    """Runs a grid of simulations for different values of the parameters
    obs_noise and cue_noise.
    """
    obs_noises = np.array([0.5, 2])
    cue_noises = np.array([0.0, 0.33])
    numbers = np.array((len(obs_noises), len(cue_noises)))
    agents_pars = np.empty(numbers, dtype=object)
    tasks_pars = np.empty(numbers, dtype=object)

    task_common = task_pars.copy()
    task_common['forces'] = [[0, 0], [1, 4]]
    task_common['context_seq'] = pars.define_contexts([[1, 20],
                                                       [0, 10]])
    task_common['cues'] = task_common['context_seq']
    task_common['breaks'] = np.zeros(
        len(task_common['context_seq']), dtype=int)

    for ix_obs, c_obs in enumerate(obs_noises):
        for ix_cue, c_cue in enumerate(cue_noises):
            c_task = task_common.copy()
            c_task['obs_noise'] = c_obs
            c_agent_pars = {'max_force': 20,
                            'angles': [0, 1],
                            'force_sds': 0.01 * np.ones(2),
                            'hyper_sd': 5,
                            'obs_sd': 1,
                            'context_noise': 0.09,
                            'prediction_noise': 0.01,
                            'action_sd': 0.17,
                            'cue_noise': c_cue}
            agents_pars[ix_obs, ix_cue] = c_agent_pars
            tasks_pars[ix_obs, ix_cue] = c_task

    if plot:
        agents = np.empty(numbers, dtype=object)
        if axes is None:
            fig, axes = plt.subplots(*numbers, num=fignum, clear=True,
                                     sharex=True, sharey=True)
        for ix_obs in range(numbers[0]):
            for ix_cue in range(numbers[1]):
                agents[ix_obs, ix_cue] = model.LRMeanSD(**agents_pars[ix_obs,
                                                                      ix_cue])
                sim_and_plot(agent=agents[ix_obs, ix_cue],
                             pars_task=tasks_pars[ix_obs, ix_cue],
                             axes=[axes[ix_obs, ix_cue], axes[ix_obs, ix_cue]])
    idxs = [0, 2]
    return tasks_pars, agents_pars


def multiple_runs(runs, agent_pars, task_pars, names=None,
                  agent_class=None):
    """Simulates --runs-- runs of the experiments with each of the agents
    described by --agent_pars--, in the tasks described by --task_pars--.

    Returns a panda with a meaningless index, with the columns returned by
    agent_class.join_pandas, adding columns for trial number and group name
    (each group defined by an agent).

    Parameters
    ----------
    runs : int
    Number of runs of the experiment per agent to do.

    agent_pars : iterable
    Iterable of dictionaries, each dictionary containing the parameters for
    the agent. The elements of the dictionaries must match the parameters of
    --agent_class--. Its size must match that of --task_pars--.

    task_pars : iterable
    Iterable of dictionaries, each dictionary containing the parameters for
    the task in thh. Its size must match that of --agent_pars--.

    names : iterable
    List of names to identify each agent. The names are used for the "group"
    column of the returned panda. If None, range(num_agents) is used.

    agent_class : class
    Class for the agent to use. This function assumes children of
    model.LeftRightAgent, but anything that can work with thh.run should
    work.

    """
    if names is None:
        names = np.arange(len(agent_pars))
    if agent_class is None:
        agent_class = model.LRMeanSD
    data = {name: [] for name in names}
    looping = [task_pars, agent_pars, names]
    for idx, (task, ag_pars, name) in enumerate(zip(*looping)):
        for idx in range(runs):
            agent = agent_class(**ag_pars)
            pandata, pandagent, _ = thh.run(agent, pars=task)
            pandota = thh.join_pandas(pandata, pandagent)
            pandota['part'] = idx
            data[name].append(pandota)
        data[name] = pd.concat(data[name])
    data = pd.concat(data, axis=0, names=['Group', 'trial'])
    data.reset_index('Group', inplace=True)
    data.reset_index('trial', inplace=True)
    return data
