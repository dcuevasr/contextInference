# -*- coding: utf-8 -*-
# .adaptation/code/figures.py

"""Figures for the context inference motor adaptation paper. The default
parameter --fignum-- for each function defined here determines which figure it
is in the paper. Those that do not have one are not paper figures, but
auxiliary functions."""

import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from matplotlib import gridspec as gs
import seaborn as sns
import matplotlib as mpl

import simulations as sims
import model
import task_hold_hand as thh
import pars

FIGURE_FOLDER = '../article/figures/'

mpl.rcParams['font.size'] = 8
mpl.rcParams['figure.titlesize'] = 'small'
mpl.rcParams['lines.linewidth'] = 1


def model_showoff(fignum=2, show=True, do_a=True, save=False):
    """Plots a bunch of arbitrary simulations that show how the parameters of
    the model work and affect inference. Also leaves an empty space on top
    to insert the diagram of the generative model by hand.

    Parameters
    ----------
    do_a : bool
    Whether to add the model diagram to the figure. If True, the model diagram
    is assumed to be in FIGURE_FOLDER/generative.png. If False, the space will
    be left empty at the top.

    """
    runs = 50
    highlight_color = 'lightgray'
    colors = ['black', 'tab:olive', 'tab:blue', 'tab:orange', 'tab:green']
    tasks, agents_pars = sims.model_showoff(plot=False)
    numbers = tasks.shape
    figsize = (5, 6)
    fig = plt.figure(fignum, clear=True, figsize=figsize)
    height_ratios = 0.5 * np.ones(numbers[0] + 2)
    height_ratios[0] = 3
    height_ratios[1] = 0.3
    width_ratios = np.ones(numbers[1] + 2)
    width_ratios[0] = 2
    width_ratios[1] = 0.35
    gsbig = gs.GridSpec(numbers[0] + 2, numbers[1] + 2, figure=fig,
                        height_ratios=height_ratios, width_ratios=width_ratios,
                        hspace=0.1, wspace=0.1)
    axis_diagram = fig.add_subplot(gsbig[0, :])
    ctx_switch = np.diff(
        tasks.reshape(-1)[0]['context_seq']).nonzero()[0][0] + 1
    num_trials = len(tasks.reshape(-1)[0]['context_seq'])
    axes = np.empty(numbers, dtype=object)
    axes_state = fig.add_subplot(gsbig[2:, 0])
    for ix_row in np.arange(2, numbers[0] + 2):
        for ix_col in np.arange(2, numbers[1] + 2):
            axes[ix_row - 2, ix_col -
                 2] = fig.add_subplot(gsbig[ix_row, ix_col])
    states = []
    for ix_row in range(numbers[0]):
        for ix_col in range(numbers[1]):
            c_axes = axes[ix_row, ix_col]
            agent = model.LRMeanSD(**agents_pars[ix_row, ix_col])
            task = tasks[ix_row, ix_col]
            pandata, pandagent, _ = thh.run(agent, pars=task)
            pandota = thh.join_pandas(pandata, pandagent)
            pandota = sims.multiple_runs(runs=runs,
                                         agent_pars=[agents_pars[ix_row,
                                                                 ix_col], ],
                                         task_pars=[task, ])
            c_axes.axvline(ctx_switch, linestyle='--', color='black',
                           alpha=0.3)
            c_colors = [colors[0], colors[ix_row + ix_col * numbers[0] + 1]]
            c_cycler = plt.cycler('color', c_colors)
            c_axes.set_prop_cycle(c_cycler)
            sns.lineplot(data=pandota, x='trial', y='con0', errorbar='se',
                         ax=c_axes)
            sns.lineplot(data=pandota, x='trial', y='con1', errorbar='se',
                         ax=c_axes)
            pandota['cue_uncertainty'] = int(ix_row)
            pandota['obs_noise'] = int(ix_col)
            states.append(pandota)
    xticks = np.array([0, ctx_switch])
    megapanda = pd.concat(states, ignore_index=True)
    megapanda['one_noise'] = megapanda['cue_uncertainty'] + \
        megapanda['obs_noise'] * megapanda['cue_uncertainty'].max()
    megapanda['cat_noise'] = megapanda['one_noise'].astype(str) + '--'
    cycler = plt.cycler('color', colors[1:])
    axes_state.set_prop_cycle(cycler)
    sns.lineplot(data=megapanda, x='trial', y='mag_mu_1', errorbar='se', hue='cat_noise',
                 ax=axes_state)
    axes_state.set_title('Inferred state')
    axes_state.set_xlabel('Trial')
    axes_state.set_ylabel('Force')
    axes_state.get_legend().remove()
    for axis in axes.reshape(-1):
        axis.set_xticks([])
        axis.set_yticks([])
        axis.set_xlim((0, num_trials - 1))
        axis.set_xlabel('')
        axis.set_ylabel('')
        axis.spines['right'].set_visible(False)
        axis.spines['top'].set_visible(False)
    for rows in axes[-1, :]:
        rows.set_xlabel('Trial')
        rows.set_xticks(xticks)
    for cols in axes[:, 0]:
        cols.set_yticks([0, 0.5])  # , labels=['0', '.5'])
    cue_texts = ['Low', 'High']
    for tops, text in zip(axes[0, :], cue_texts):
        tops.set_title(text, verticalalignment='bottom')
    axes[0, 1].text(s='Cue Uncertainty', x=0, y=1.8,
                    transform=axes[0, 1].transAxes,
                    verticalalignment='center',
                    horizontalalignment='center',
                    backgroundcolor=highlight_color,
                    linespacing=0.1)
    obs_texts = ['Low', 'High']
    for lefts, text in zip(axes[:, -1], obs_texts):
        lefts.text(s=text, x=1.1, y=0.5, transform=lefts.transAxes,
                   rotation=270, horizontalalignment='center',
                   verticalalignment='center')
    axes[1, -1].text(s='Obs. Noise', x=1.25, y=0,
                     transform=axes[0, -1].transAxes,
                     rotation=270, verticalalignment='center',
                     backgroundcolor=highlight_color,)
    axes[-1, 0].set_ylabel(r'p(ctx)')

    # Import diagram of the model from png
    if do_a:
        diagram = plt.imread(FIGURE_FOLDER + '/generative.png')
        axis_diagram.axis('off')
        axis_diagram.imshow(diagram)

    # subplot labels
    offset_multiplier = 0.03
    offset = np.array([-1, figsize[1] / figsize[0]]) * offset_multiplier
    anchor_a = np.array(axis_diagram.get_position())[[0, 1], [0, 1]] + offset
    anchor_b = np.array(axes_state.get_position())[[0, 1], [0, 1]] + offset
    fig.text(s='A', x=anchor_a[0], y=anchor_a[1], fontdict={'size': 12})
    fig.text(s='B', x=anchor_b[0], y=anchor_b[1], fontdict={'size': 12})
    if save:
        plt.savefig(FIGURE_FOLDER + 'figure_{}.png'.format(fignum),
                    dpi=600, bbox_inches='tight')
        plt.savefig(FIGURE_FOLDER + 'figure_{}.svg'.format(fignum), format='svg')
    if show:
        plt.draw()
        plt.show(block=False)


def plot_adaptation(pandata, axis, colors):
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
    adapt_color = np.array([174, 99, 164]) / 256
    # axis.plot(np.array(pandata['pos(t)']), color='black', alpha=0.4,
    #           label='Error')
    magmu = [np.array(pandata[column])
             for column in columns
             if column.startswith('mag_mu')]
    errors = [np.array(pandata[column])
              for column in columns
              if column.startswith('mag_sd')]
    for color_x, error_x, magmu_x in zip(colors, errors, magmu):
        axis.plot(magmu_x, color=color_x, label='{} model'.format(color_x),
                  linewidth=2)
        # axis.fill_between(trial, magmu_x - 2 * error_x, magmu_x + 2 * error_x,
        #                   color=color_x, alpha=0.1)
    axis.plot(np.array(-pandata['hand'] - pandata['action']),
              color=adapt_color, label='Adaptation', zorder=1000)
    magmu = np.array(magmu)
    yrange = np.array([magmu.min(), magmu.max()]) * 1.1
    axis.set_ylim(yrange)
    plt.draw()


def oh_2019_kim_2015(fignum=3, show=True, save=False):
    """Reproduces the results from Oh_Minimizing_2019 and kim_neural_2015 and
    plots them in a format similar to their plots.

    An empty subplot is created to manually add the figure from their
    paper.

    """
    supp_fignum = f's{fignum}'
    kim_reps = 20  # As in Kim 2015
    oh_reps = 11  # As in Oh 2019
    context_color = np.ones(3) * 0.5
    ad_color = np.ones(3) * 0.3
    colors = ['black', 'tab:green', 'tab:orange']
    cycler = plt.cycler('color', colors)

    figsize = (7, 4)
    fig = plt.figure(fignum, clear=True, figsize=figsize)
    magri = gs.GridSpec(3, 5, width_ratios=[1, 0.17, 0.5, 1, 1], wspace=0.05,
                        hspace=0.25, figure=fig)
    axes = np.empty((3, 4), dtype=object)
    for ix_col, col in enumerate([0, 2, 3, 4]):
        for ix_row in range(3):
            if ix_col == 2 or ix_col == 3:
                sharex = axes[ix_row, 1]
                sharey = axes[ix_row, 1]
            else:
                sharex = sharey = None
            axes[ix_row, ix_col] = fig.add_subplot(magri[ix_row, col],
                                                   sharex=sharex,
                                                   sharey=sharey)

    # Axes for the supplementary figures
    fig_supp, ax_supp = plt.subplots(3, 1, num=supp_fignum, clear=True, sharex=True, squeeze=False)
    axes = np.hstack([axes, ax_supp])

    # a)
    trials_kim = np.arange(300)  # It IS used in the query below
    task_kim, agent_pars = sims.kim_2015(plot=False)
    pandota_kim = sims.multiple_runs(kim_reps, agent_pars=[agent_pars],
                                     task_pars=[task_kim])
    pandota_kim = pandota_kim.query(f'trial in @trials_kim')
    kimcon = ['con0', 'con1', 'con2']
    contexts = pandota_kim['ix_context'].values
    contexts[contexts == pars.CLAMP_INDEX] = 0
    contexts[contexts == 1] = -40
    contexts[contexts == 2] = 40
    _one_oh(pandota_kim, colors, context_color, 40, axes[0, :], -40)
    axes[0, 2].set_yticks([-40, 0, 40])
    axes[0, 3].text(s='Angle', x=-0.35, y=0.5,
                    transform=axes[0, 1].transAxes, rotation=90,
                    horizontalalignment='center',
                    verticalalignment='center')
    axes[0, 1].set_title('Inferred state')
    axes[0, 2].set_title('Obs. Adapt.')
    axes[0, 0].set_title('Context')

    # b)
    task_20, task_10, agent_pars = sims.oh_2019(plot=False)
    agent = model.LRMeanSD(**agent_pars)
    agent.all_learn = True
    pandota_20 = sims.multiple_runs(oh_reps, agent_pars=[agent_pars],
                                    task_pars=[task_20])
    _one_oh(pandota_20, colors, context_color, 20, axes[1, :])

    # c)
    agent = model.LRMeanSD(**agent_pars)
    agent.all_learn = True
    pandota_10 = sims.multiple_runs(oh_reps, agent_pars=[agent_pars],
                                    task_pars=[task_10])
    _one_oh(pandota_10, colors, context_color, 10, axes[2, :])

    # Subplot labels
    axes[0, 0].text(x=-0.1, y=1.05, s='A',
                    transform=axes[0, 0].transAxes,
                    fontdict={'size': 12})
    axes[1, 0].text(x=-0.1, y=1.05, s='B',
                    transform=axes[1, 0].transAxes,
                    fontdict={'size': 12})
    axes[2, 0].text(x=-0.1, y=1.05, s='C',
                    transform=axes[2, 0].transAxes,
                    fontdict={'size': 12})

    # Axes shenanigans
    for axis in axes[:, 0].reshape(-1):
        axis.set_yticks([0, 1])
        axis.set_ylabel('p(ctx)')
    for axis in axes[:, -3:-1].reshape(-1):
        yticks = axis.get_yticks().astype(int)
        axis.set_yticklabels(yticks, visible=False)
    axes[0, -2].set_title('Exp. data')
    axes[-1, -1].set_xlabel('Trial')
    for row in [0, 1]:
        for axis in axes[row, :].reshape(-1):
            axis.set_xticklabels(axis.get_xticklabels(), visible=False)
    for axis in axes.reshape(-1):
        axis.spines['right'].set_visible(False)
        axis.spines['top'].set_visible(False)
    # fig.align_ylabels(axes[:, 0])
    if save:
        fig.savefig(FIGURE_FOLDER + 'figure_{}.svg'.format(fignum),
                    dpi=600, bbox_inches='tight')
        fig_supp.savefig(FIGURE_FOLDER + '{}.svg'.format(supp_fignum),
                         format='svg', bbox_inches='tight')
    if show:
        plt.draw()
        plt.show(block=False)


def _one_oh(pandota, colors, context_color, adapt, axes, adapt2=None):
    """Plots one full row of the Oh simulations.

    """
    ohcon = ['con0', 'con1']
    if not (adapt2 is None):
        ohcon += ['con2']
    pandota_u = pandota.loc[pandota['part'] == 0]
    contexts = pandota_u['ix_context'].values
    contexts[contexts == pars.CLAMP_INDEX] = 0
    contexts[contexts == 1] = adapt
    if not (adapt2 is None):
        contexts[contexts == 2] = adapt2
    contexts_20 = contexts
    switches = np.nonzero(np.diff(contexts_20))[0]
    switches = np.concatenate([[0], switches, [len(contexts)]])
    magmustr = np.unique([column for column in pandota.columns if
                          column.startswith('mag_mu')])
    axes[2].plot(contexts, color=context_color)
    axes[4].plot(contexts, color=context_color)  # Supp. mat
    axes[3].plot(contexts, color=context_color)
    pandota['adapt'] = -pandota['pos(t)'] - pandota['action']
    sns.lineplot(data=pandota, x='trial', y='adapt', ax=axes[2], color='black')
    sns.lineplot(data=pandota, x='trial', y='adapt', ax=axes[4], color='black')  # supp
    for c_con, c_state, c_color in zip(ohcon, magmustr, colors):
        sns.lineplot(data=pandota, x='trial', y=c_con, color=c_color,
                     ax=axes[0], errorbar='se')
        sns.lineplot(data=pandota, x='trial', y=c_state, color=c_color,
                     ax=axes[1], errorbar='se')
    for axis in axes:
        axis.set_xlabel('Trial')
        axis.set_ylabel('')

    axes[2].text(s='Angle', x=-0.35, y=0.5,
                 transform=axes[1].transAxes, rotation=90,
                 horizontalalignment='center',
                 verticalalignment='center')
    axes[0].set_ylim([0, 1.2])
    ylim = axes[0].get_ylim()
    axes[0].vlines(switches, *ylim, linestyle='--', color='black',
                   linewidth=0.5)


def davidson_2004(fignum=4, show=True, save=False):
    """Reproduces the results from Davidson_Scaling_2004, leaving an empty
    subplot to put in the results from their paper.

    """
    supp_fignum = f's{fignum}'
    repeats = 32  # No. of participants per group
    figsize = (6, 4)
    colors = [np.array((95, 109, 212)) / 256,
              np.array((212, 198, 95)) / 256]
    # colors = {'-A': np.array((95, 109, 212)) / 256,
    #           '3A': np.array((212, 198, 95)) / 256}
    ran = [161, 200]  # Trials of importance for this figure

    fig = plt.figure(num=fignum, clear=True, figsize=figsize)
    fig_supp, ax_supp = plt.subplots(3, 1, num=supp_fignum, clear=True, sharex=True, squeeze=False)
    
    gsbig = gs.GridSpec(2, 1, figure=fig, height_ratios=[1.5, 1], hspace=0.4)
    gstop = gsbig[0].subgridspec(2, 4, width_ratios=[1, 0.25, 2, 2],
                                 wspace=0.05)
    gsbot = gsbig[1].subgridspec(2, 4, width_ratios=[1, 1.8, 1, 1.8],
                                 wspace=0.25)

    axes = []  # Gets turned into np.array later
    axes.append(fig.add_subplot(gstop[:, 2]))
    axes.append(fig.add_subplot(gstop[0, 0]))
    axes.append(fig.add_subplot(gstop[1, 0]))
    axes.append(fig.add_subplot(gstop[:, 3]))
    axes = np.array(axes)

    baxes = []  # Gets turned into np.array later
    baxes.append(fig.add_subplot(gsbot[:, 1]))
    baxes.append(fig.add_subplot(gsbot[0, 0]))
    baxes.append(fig.add_subplot(gsbot[1, 0]))
    baxes.append(fig.add_subplot(gsbot[:, 3]))
    baxes.append(fig.add_subplot(gsbot[0, 2]))
    baxes.append(fig.add_subplot(gsbot[1, 2]))
    baxes = np.array(baxes)

    tasks_all, agents_pars_all = sims.davidson_2004(plot=False)
    agent_pars = agents_pars_all[:2]  # For top row
    tasks = tasks_all[:2]             # For top row
    agent_sims = agents_pars_all[2:]  # For bottom row
    tasks_sims = tasks_all[2:]        # For bottom row

    names = ['-A', '3A']
    labels = [['O', 'A', '-A'], ['O', 'A', '3A']]
    _davidson_trio(repeats, colors, agent_pars, tasks, names, axes, labels,
                   ran)

    names = ['-2A', '4A']
    labels = [['_O', '_A', '-2A'], ['_O', '_A', '4A']]
    _davidson_trio(repeats, colors, agent_sims[:2], tasks_sims[:2],
                   names, baxes[:3], labels, ran)
    names = ['-A', '3A']
    labels = [['_O', 'A', '-A'], ['_O', 'A', '3A']]
    _davidson_trio(repeats, colors, agent_sims[2:], tasks_sims[2:],
                   names, baxes[3:], labels, ran)
    for axis in axes.reshape(-1):
        axis.set_xlabel('')
        axis.spines['top'].set_visible(False)
        axis.spines['right'].set_visible(False)
    axes[3].set_xlabel('Trials since switch')
    for axis in baxes.reshape(-1):
        axis.set_xlabel('')
        axis.spines['top'].set_visible(False)
        axis.spines['right'].set_visible(False)
    axes[1].set_title('p(ctx)', horizontalalignment='center')
    axes[3].set_title('Exp. data')
    axes[3].set_yticklabels([])
    axes[1].set_xticks([])
    baxes[0].set_xlabel('Trials since switch')
    baxes[4].set_ylabel('')
    baxes[5].set_ylabel('')
    baxes[1].set_xticks([])
    baxes[4].set_xticks([])

    # subplot labels
    offset_multiplier = 0.03
    offset = np.array([-1, figsize[1] / figsize[0]]) * offset_multiplier
    anchor_a = np.array(axes[1].get_position())[[0, 1], [0, 1]] + offset
    anchor_b = np.array(baxes[1].get_position())[[0, 1], [0, 1]] + offset
    anchor_c = np.array(baxes[4].get_position())[[0, 1], [0, 1]] + offset
    fig.text(s='A', x=anchor_a[0], y=anchor_a[1], fontdict={'size': 12})
    fig.text(s='B', x=anchor_b[0], y=anchor_b[1], fontdict={'size': 12})
    fig.text(s='C', x=anchor_c[0], y=anchor_c[1], fontdict={'size': 12})

    if save:
        fig.savefig(FIGURE_FOLDER + 'figure_{}.svg'.format(fignum),
                    dpi=600, bbox_inches='tight')
        fig_supp.savefig(FIGURE_FOLDER + 'figure_{}.svg'.format(supp_fignum),
                         format='svg', bbox_inches='tight')
    if show:
        plt.draw()
        plt.show(block=False)


def _davidson_trio(repeats, color_list, agent_pars, tasks, names, axes, labels,
                   ran):
    """Top row of plots for davidson_2004().

    There should be three axes. The last two are used to plot context
    inference, the first for adaptation. --agent_pars--, --names-- and
    --tasks-- are assumed to have the same size (2, ), corresponding to the two
    groups in each simulated experiment.

    --labels-- contains the labes for each plot, which means that there have to
      be three labels per each agent, e.g. [['a', 'b', 'c'], ...]. Note that
      prefixing a label with an underscore makes it disappear from the legend.

    """
    colors = {key: color for key, color in zip(names, color_list)}
    data = {name: [] for name in names}
    for idx, (task, ag_pars, name) in enumerate(zip(tasks, agent_pars, names)):
        for idx in range(repeats):
            agent = model.LRMeanSD(**ag_pars)
            pandata, pandagent, _ = thh.run(agent, pars=task)
            pandota = thh.join_pandas(pandata, pandagent)
            pandota['part'] = idx
            data[name].append(pandota)
        data[name] = pd.concat(data[name])
        data[name]['pos(t)'] = np.abs(data[name]['pos(t)'])
    data = pd.concat(data, axis=0, names=['Group', 'trial'])
    data.reset_index('Group', inplace=True)
    data.reset_index('trial', inplace=True)
    ymax = data.loc[data['trial'] > ran[0], 'pos(t)'].max()
    ymin = data.loc[data['trial'] > ran[0], 'pos(t)'].min()
    sns.lineplot(data=data, x='trial', y='pos(t)', ax=axes[0],
                 hue='Group', palette=colors, errorbar='se')
    datum_a = data.query('Group == @names[0]')
    datum_b = data.query('Group == @names[1]')
    sns.lineplot(data=datum_a, x='trial', y='con0', ax=axes[1],
                 color='black', label=labels[0][0], errorbar='se')
    sns.lineplot(data=datum_a,
                 x='trial', y='con1', ax=axes[1], label=labels[0][1],
                 color='tab:green', errorbar='se')
    sns.lineplot(data=datum_a,
                 x='trial', y='con2', ax=axes[1], label=labels[0][2],
                 color='tab:blue', errorbar='se')
    sns.lineplot(data=datum_b, x='trial', y='con0', ax=axes[2],
                 color='black', label=labels[1][0], errorbar='se')
    sns.lineplot(data=datum_b,
                 x='trial', y='con1', ax=axes[2], label=labels[1][1],
                 color='tab:green', errorbar='se')
    sns.lineplot(data=datum_b,
                 x='trial', y='con2', ax=axes[2], label=labels[1][2],
                 color='tab:blue', errorbar='se')
    axes[1].legend(ncol=1, fontsize='x-small', handlelength=1)
    axes[2].legend(ncol=1, fontsize='x-small', handlelength=1)

    ticks = np.array(axes[0].get_xticks(), dtype=int) - ran[0] + 1
    for idx in range(len(axes)):
        axes[idx].set_xlim(ran)
        ticks = np.array(axes[idx].get_xticks(), dtype=int) - ran[0] + 1
        axes[idx].set_xticklabels(ticks)
    axes[1].set_ylabel('Grp. -A', labelpad=0)
    axes[2].set_ylabel('Grp. 3A', labelpad=0)
    axes[0].set_ylabel('Error (a.u.)', labelpad=0)
    axes[0].set_yticklabels([])
    axes[1].set_yticks((0, 1))
    axes[2].set_yticks((0, 1))
    axes[0].set_ylim(1.2 * np. array((ymin, ymax)))


def vaswani_2013(fignum=5, show=True, pandota=None, save=False):
    """Reproduces the results from Vaswani_Decay_2013, specifically their
    figures 2a-c.

    """
    reps = 6
    figsize = (6, 6)
    colors = ['blue', 'green', 'red', 'c']
    context_colors = ['black', 'tab:orange', 'tab:purple', 'tab:brown',
                      'tab:olive']
    mags = gs.GridSpec(4, 4, height_ratios=[2.2, 0.6, 1, 1],
                       width_ratios=[1, 1, 0.5, 2.2], wspace=0.05, hspace=0.05)
    fig = plt.figure(num=fignum, clear=True, figsize=figsize)
    axes_con = [fig.add_subplot(mags[2, 0])]
    axes_con.append(fig.add_subplot(mags[2, 1]))
    axes_con.append(fig.add_subplot(mags[3, 0]))
    axes_con.append(fig.add_subplot(mags[3, 1]))
    axes_sum = [fig.add_subplot(mags[0, 0:2]), ]
    axes_sum.append(fig.add_subplot(mags[0, 3], sharex=axes_sum[0]))
    axes_lag = fig.add_subplot(mags[2:4, 3])
    tasks, agents = sims.vaswani_2013(plot=False)

    all_pandas = []
    names = [1.1, 1.2, 1.3, 1.4]
    named_colors = {name: color for name, color in zip(names, colors)}
    for idx, (task, agent_par) in enumerate(zip(tasks, agents)):
        for ix_rep in range(reps):
            agent = model.LRMeanSD(**agent_par)
            pandata, pandagent, _ = thh.run(agent=agent,
                                            pars=task)
            c_pandota = thh.join_pandas(pandata, pandagent)
            c_pandota['group'] = names[idx]
            c_pandota['run'] = idx * reps + ix_rep  # to separate all runs
            c_pandota.reset_index('trial', inplace=True)
            all_pandas.append(c_pandota)
    pandota = pd.concat(all_pandas, axis=0, ignore_index=True)
    adaptation = -pandota['pos(t)'] - pandota['action']
    pandota['adaptation'] = adaptation
    pandota.loc[pandota['group'] == 'Group 1.4', ['adaptation']] *= -1
    # Plot context inference
    for idx, (group, color, ax) in enumerate(zip(names, colors, axes_con)):
        c_panda = pandota.query('group == @group')
        real_con = np.array(c_panda.groupby('trial').mean()['ix_context'])
        con_breaks = np.nonzero(np.diff(real_con))[0] + 1
        con_breaks = np.concatenate([[0], con_breaks, [len(real_con) - 1]])
        cons = np.array([real_con[one_break] for one_break in con_breaks],
                        dtype=int)
        # plot real context
        for c_con, n_con, ix_con in zip(con_breaks, con_breaks[1:], cons):
            if ix_con == pars.CLAMP_INDEX:
                ix_con = -1
            ax.fill_between([c_con, n_con], [0] * 2,
                            [1] * 2,
                            color=context_colors[ix_con],
                            alpha=0.2)
        con_strings = sorted([column for column in c_panda.columns
                              if (column.startswith('con')
                                  and not column.startswith('con_'))])
        for ix_con, con in enumerate(con_strings):
            # Little hack to get the colors to match between real and inferred:
            c_con = ix_con
            if idx == 2 and ix_con == 2:
                c_con = 3
            if idx == 3 and ix_con == 1:
                c_con = 2
            # End little hack.
            try:
                sns.lineplot(data=c_panda, x='trial', y=con,
                             color=context_colors[c_con],
                             ax=ax, errorbar='se')
            except KeyError:
                pass
    for idx in range(4):
        axes_con[idx].text(s='Group {}'.format(names[idx]),
                           x=0.5, y=0.95, horizontalalignment='center',
                           verticalalignment='top',
                           transform=axes_con[idx].transAxes)
        axes_con[idx].set_xlabel('')
        axes_con[idx].set_ylabel('')
        axes_con[idx].set_xticks([])
        axes_con[idx].set_yticks([])
        axes_con[idx].set_ylim([0, 1])
    axes_con[2].set_yticks([0, 1])
    axes_con[2].set_ylabel(r'p(ctx)', labelpad=-0.1)
    axes_con[2].set_xticks([75, 175])
    axes_con[2].set_xticklabels([0, 100])
    axes_con[2].set_xlabel('Trial')
    axes_con[0].text(x=-0.3, y=1.1, s='C',
                     transform=axes_con[0].transAxes,
                     fontdict={'size': 12})

    # Plot summary adaptation:
    condi = 'trial >= 75 and trial <= 175 and group != 1.4'
    pandota_e = pandota.query(condi)
    # pandota_e.reset_index('trial', inplace=True)
    pandota_e.loc[:, 'trial'] -= 100
    sns.lineplot(x='trial', y='adaptation', hue='group',
                 palette=named_colors, data=pandota_e,
                 ax=axes_sum[0], errorbar='se')
    # labels = [name[-3:] for name in names[:-1]]
    # axes_sum[0].legend(ncol=3, labels=labels)
    y_range = axes_sum[0].get_ylim()
    axes_sum[0].plot([0, 0], y_range, linestyle='dashed',
                     color='black', alpha=0.3)
    axes_sum[1].plot([0, 0], y_range, linestyle='dashed',
                     color='black', alpha=0.3)
    axes_sum[0].set_ylim(y_range)
    axes_sum[1].set_ylim(y_range)
    axes_sum[0].set_yticks([0, 1])
    axes_sum[1].set_yticks([])
    axes_sum[0].set_xlabel('Trials since start of error-clamp', labelpad=1)
    axes_sum[1].set_xlabel('Trials since start of error-clamp', labelpad=1)
    axes_sum[0].set_ylabel('Adaptation index')

    axes_sum[0].text(x=-0.1, y=1, s='A',
                     transform=axes_sum[0].transAxes,
                     fontdict={'size': 12})
    axes_sum[1].text(x=-0.1, y=1, s='B',
                     transform=axes_sum[1].transAxes,
                     fontdict={'size': 12})

    # Plot lags
    panda_lag = pandota.query('trial > 90 and trial <= 140')
    # panda_lag.reset_index('trial', inplace=True)
    panda_lag.loc[:, 'trial'] -= 100
    axes_lag.set_ylabel('p(ctx)')
    axes_lag.text(x=-0.1, y=1.033, s='D',
                  transform=axes_lag.transAxes,
                  fontdict={'size': 12})

    sns.lineplot(data=panda_lag, x='trial', y='con1',
                 hue='group', units='run', estimator=None,
                 palette=colors)
    axes_lag.set_xlabel('Trials since start of error-clamp')

    axes_sum[0].set_title('Simulations')
    axes_sum[1].set_title('Exp. data')
    axes_lag.set_title('Simulations')
    if save:
        plt.savefig(FIGURE_FOLDER + 'figure_{}.png'.format(fignum),
                    dpi=600, bbox_inches='tight')
        plt.savefig(FIGURE_FOLDER + 'figure_{}.svg'.format(fignum),
                    format='svg', bbox_inches='tight')

    if show:
        plt.draw()
        plt.show(block=False)


if __name__ == '__main__':
    print('Running all simulations. This might take a while...')
    model_showoff(do_a=False)
    oh_2019_kim_2015()
    davidson_2004()
    vaswani_2013()
