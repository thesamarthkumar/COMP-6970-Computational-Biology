import numpy as np
import pandas as pd
from scipy.stats import norm
from plotnine import *


'''
Homework 4 problem 1 -- Plot data (please save to file, don't just print it)
Plot the timeseries data for simulated nanopore
'''
def plot_timeseries_data(data):
    plot = (
        ggplot(data, aes(x='time', y='level')) +
        geom_line(color='blue', size=0.5) +
        theme(
            plot_background=element_rect(fill='lightgray'),
            panel_background=element_rect(fill='white')
        ) +
        ggtitle("Nanopore data")
    )
    plot.save("timeseries.png", width=10, height=10)
    return plot


'''
Homework 4 problem 2
What is the approximate duration of each "event" in this data given this plot?
'''
approx_duration = 145  # Just eyeballing from the plot


'''
Homework 4 problem 3 -- HMM maximum likelihood state sequence with 4 states
'''
def HMM_MLE(df):
    observations = df['level'].values
    n = len(observations)
    states = ['T', 'A', 'G', 'C']
    num_states = len(states)

    emission_params = {
        'T': (100, 15),
        'A': (150, 25),
        'G': (300, 50),
        'C': (350, 25)
    }

    trans_prob = np.full((num_states, num_states), 1/50)
    np.fill_diagonal(trans_prob, 49/50)

    viterbi = np.zeros((num_states, n))
    backpointer = np.zeros((num_states, n), dtype=int)

    for s in range(num_states):
        mu, sigma = emission_params[states[s]]
        viterbi[s, 0] = np.log(1/num_states) + norm.logpdf(observations[0], mu, sigma)

    for t in range(1, n):
        for s in range(num_states):
            mu, sigma = emission_params[states[s]]
            emission_log_prob = norm.logpdf(observations[t], mu, sigma)
            trans_probs = viterbi[:, t-1] + np.log(trans_prob[:, s])
            best_prev_state = np.argmax(trans_probs)
            viterbi[s, t] = trans_probs[best_prev_state] + emission_log_prob
            backpointer[s, t] = best_prev_state

    best_final_state = np.argmax(viterbi[:, -1])
    best_path = [best_final_state]
    for t in range(n-1, 0, -1):
        best_prev = backpointer[best_path[-1], t]
        best_path.append(best_prev)
    best_path = best_path[::-1]
    decoded_states = [states[i] for i in best_path]

    df = df.copy()
    df['state'] = decoded_states
    return df


'''
Homework 4 problem 4 -- Overlay plot of state sequence
'''
def plot_MLE(state_sequence):
    plot = (
        ggplot(state_sequence, aes(x='time', y='level', color='state')) +
        geom_line(size=0.4) +
        labs(title='Viterbi State Sequence', x='Time', y='Level') +
        theme(
            plot_background=element_rect(fill='lightgray'),
            panel_background=element_rect(fill='white')
        )
    )
    plot.save("viterbi_states_overlay.png", width=10, height=5)
    return plot


'''
Homework 4 problem 5 -- Most likely sequence using approximate event length
'''
def MLE_seq(df, event_length=approx_duration):
    n = len(df)
    sequence = []
    for start in range(0, n, event_length):
        end = min(start + event_length, n)
        chunk = df.iloc[start:end]
        most_common_state = chunk['state'].mode()[0]
        sequence.append(most_common_state)

    sequence_str = ''.join(sequence)
    print("Most likely sequence:", sequence_str)
    return sequence


'''
Homework 4 problem 6 -- Forward/Backward Algorithm for Posterior Probabilities
'''
def HMM_posterior(df):
    observations = df['level'].values
    n = len(observations)
    states = ['T', 'A', 'G', 'C']
    num_states = len(states)

    emission_params = {
        'T': (100, 15),
        'A': (150, 25),
        'G': (300, 50),
        'C': (350, 25)
    }

    trans_prob = np.full((num_states, num_states), 1/50)
    np.fill_diagonal(trans_prob, 49/50)

    log_trans = np.log(trans_prob)
    log_start = np.log(np.full(num_states, 1/num_states))

    log_emission = np.zeros((num_states, n))
    for s in range(num_states):
        mu, sigma = emission_params[states[s]]
        log_emission[s] = norm.logpdf(observations, mu, sigma)

    log_alpha = np.zeros((num_states, n))
    log_alpha[:, 0] = log_start + log_emission[:, 0]
    for t in range(1, n):
        for j in range(num_states):
            log_alpha[j, t] = log_emission[j, t] + np.logaddexp.reduce(log_alpha[:, t-1] + log_trans[:, j])

    log_beta = np.zeros((num_states, n))
    for t in range(n-2, -1, -1):
        for i in range(num_states):
            log_beta[i, t] = np.logaddexp.reduce(log_trans[i, :] + log_emission[:, t+1] + log_beta[:, t+1])

    log_post = log_alpha + log_beta
    log_post -= np.logaddexp.reduce(log_post, axis=0)

    posteriors = np.exp(log_post).T

    posterior_df = df.copy()
    for i, s in enumerate(states):
        posterior_df[f'P({s})'] = posteriors[:, i]

    return posterior_df


'''
Homework 4 problem 7 -- Faceted plot of posterior probabilities
'''
def plot_posterior(posteriors):
    melted = pd.melt(
        posteriors,
        id_vars=['time'],
        value_vars=['P(T)', 'P(A)', 'P(G)', 'P(C)'],
        var_name='State',
        value_name='Probability'
    )

    plot = (
        ggplot(melted, aes(x='time', y='Probability')) +
        geom_line(color='blue') +
        facet_wrap('~State', ncol=1) +
        labs(title='Posterior Probabilities for Each State', x='Time', y='Probability') +
        theme(
            plot_background=element_rect(fill='lightgray'),
            panel_background=element_rect(fill='white')
        )
    )
    plot.save("posterior_facets.png", width=8, height=10)
    return plot


# --- Run all steps ---
df = pd.read_csv("nanopore.csv")

# Problem 1
plot_timeseries_data(df)

# Problem 3
state_sequence = HMM_MLE(df)

# Problem 4
plot_MLE(state_sequence)

# Problem 5
MLE_seq(state_sequence, event_length=approx_duration)  

# Problem 6
posteriors = HMM_posterior(state_sequence)  # you can also pass df if needed

# Problem 7
plot_posterior(posteriors)
