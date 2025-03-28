'''
Homework 3
Samarth Kumar
'''

import numpy as np
import pandas as pd
from scipy.stats import *
from plotnine import *

# Load the data
data = pd.read_csv("nanopore.csv")

'''
Homework 4 problem 1 -- Plot data (please save to file, dont just print it)
plot the timeseries data for simulated nanopore
'''
def plot_timeseries(data):
    plot = (
        ggplot(data, aes(x='time', y='level')) + 
        geom_line(color='blue') +
        labs(
            title='Nanopore Time Series Data',
            x='Time',
            y='Signal Level'
        ) +
        theme_minimal() +
        theme(
            plot_background=element_rect(fill='white'),
            panel_background=element_rect(fill='white')
        )
    )
    plot.save('Q1_Timeseries_Data.png', dpi=300)

# Plot the timeseries data for problem 1, save to file.
plot_timeseries(data)

'''
Homework 4 problem 2
What is the approximate duration of each "event" in this data given this plot?
'''
approx_duration = 50

'''
Homework 4 problem 3 -- HMM maximum likelihood state sequence with 4 states
state 1 - T corresponds to a normal distribution with mean 100 and sd 15
state 2 - A corresponds to a normal dist with mean 150 and sd 25
state 3 - G correcponds to a normal dist with mean 300 and sd 50
state 4 - C corresponds to a normal dist with mean 350 and sd 25
transitions between states are 1/50 and transitions to same state is 49/50
'''
def HMM_MLE(df):
    # State parameters for each nucleotide
    state_params = {'T': {'mean': 100, 'sd': 15},'A': {'mean': 150, 'sd': 25},
                    'G': {'mean': 300, 'sd': 50},'C': {'mean': 350, 'sd': 25}}
    
    # Get emission probabilities
    emissions = pd.DataFrame()
    for state, params in state_params.items():
        emissions[state] = norm.pdf(df['level'], params['mean'], params['sd'])
    
    # Get most likely state at each point
    result = pd.DataFrame(index=df.index)
    result['state'] = emissions.idxmax(axis=1)
    result['level'] = df['level']
    result['time'] = df['time']
    
    return result

# Retrieve the state sequence from problem 3.
states = HMM_MLE(data)

def plot_MLE(state_sequence):
    for state in ['A', 'T', 'G', 'C']:
        state_sequence[f'is_{state}'] = state_sequence['state'] == state
    plot = (
        ggplot(state_sequence, aes(x='time', y='level')) +
        geom_line(aes(color='state')) +
        scale_color_manual(values=['red', 'blue', 'orange', 'green']) +
        labs(
            title='Maximum Likelihood State Sequence',
            x='Time',
            y='Signal Level',
            color='State'
        ) +
        theme_minimal() +
        theme(
            plot_background=element_rect(fill='white'),
            panel_background=element_rect(fill='white')
        )
    )
    plot.save('Q4_State_Sequence.png', dpi=300)

# Plot the state sequence from problem 3, save to file.
plot_MLE(states)


'''
Homework 4 problem 5
Give the most likely sequence this data corresponds to given the likely
event length you found from plotting the data
print this sequence of A/C/G/Ts
'''
def MLE_seq(df, event_length):
    states = HMM_MLE(df)
    n_groups = len(df) // event_length
    sequence = []
    
    for i in range(n_groups):
        start = i * event_length
        end = (i + 1) * event_length
        group = states.iloc[start:end]
        most_common = group['state'].mode().iloc[0]
        sequence.append(most_common)
    
    return ''.join(sequence)

# Obtain the most likely sequence from problem 5.
print(f"Most likely DNA sequence: {MLE_seq(data, approx_duration)}")

'''
Homework 4 problem 6
Forward/backward algorithm giving posterior probabilities for each time point for
each level
'''
def HMM_posterior(df):
    signal = df['level'].values
    num_points = len(signal)
    num_states = 4
    states = {'T': (100, 15),'A': (150, 25),'G': (300, 50),'C': (350, 25)}

    # Initialize transition matrix.
    trans_prob = np.full((num_states, num_states), 1/50)
    np.fill_diagonal(trans_prob, 49/50)

    # log space.
    trans = np.log(trans_prob)
    start = np.log(np.full(num_states, 1/num_states))

    # Calculate probabilities.
    log_emission = np.zeros((num_states, num_points))
    for i, state in enumerate(states):
        mu, sigma = states[state]
        log_emission[i] = norm.logpdf(signal, mu, sigma)

    # Forward pass.
    log_alpha = np.zeros((num_states, num_points))
    log_alpha[:, 0] = start + log_emission[:, 0]
    
    for t in range(1, num_points):
        for j in range(num_states):
            log_alpha[j, t] = log_emission[j, t] 
            + np.logaddexp.reduce(log_alpha[:, t-1] + trans[:, j])

    # Backward pass.
    log_beta = np.zeros((num_states, num_points))
    
    for t in range(num_points-2, -1, -1):
        for i in range(num_states):
            log_beta[i, t] = np.logaddexp.reduce(trans[i, :] 
                                                 + log_emission[:, t+1] + log_beta[:, t+1])

    # Posterior probabilities.
    log_post = log_alpha + log_beta
    log_post -= np.logaddexp.reduce(log_post, axis=0)
    posteriors = np.exp(log_post).T

    # Return the probabilities.
    result = df.copy()
    for i, state in enumerate(states):
        result[f'P({state})'] = posteriors[:, i]

    return result

# Retrieve the posterior probabilities from problem 6.
posteriors = HMM_posterior(data)

'''
Homework 4 problem 7
plot output of problem 5, this time, plot with 4 facets using facet_wrap
'''
def plot_posterior(posteriors):
    plot_data = pd.melt(
        posteriors,
        id_vars=['time', 'level'],
        value_vars=['P(T)', 'P(A)', 'P(G)', 'P(C)'],
        var_name='State',
        value_name='Probability'
    )
    
    plot = (
        ggplot() +
        geom_line(
            data=plot_data,
            mapping=aes(x='time', y='Probability', color='State'),
            size=1
        ) +
        scale_color_manual(values=['red', 'blue', 'orange', 'green']) +
        facet_wrap('~ State', nrow=2) +
        scale_y_continuous(name='Probability', limits=[0, 1]) +
        labs(title='Posterior Probabilities', x='Time') +
        theme_minimal() +
        theme(
            plot_background=element_rect(fill='white'),
            panel_background=element_rect(fill='white')
        )
    )
    plot.save('Q7_HMM_Posterior.png', dpi=300)

# Plot the posterior probabilities from problem 7, save to file.
plot_posterior(posteriors) 
