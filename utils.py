from objects import spinner
import numpy as np

def spinners_to_states(spinners):
    states = []
    for spinner in spinners:
        if not spinner.forced:
            states.append(spinner.thetas[-1])
            states.append(spinner.dthetas[-1])
    return states

def update_spinners(spinners, states):
    for spinner in spinners:
        if not spinner.forced:
            spinner

            
            
