"""File for generating input to and reading from pomdp-solver 
"""
import io
import numpy as np
import os 
import subprocess
import glob

# PATH_TO_POMDP_SOLVER = '/Users/aidapiccato/pomdp-solve-5.4/src/pomdp-solve'
PATH_TO_POMDP_SOLVER = '/om/user/apiccato/lib/pomdp-solve-5.4/src/pomdp-solve'
def run(solver_input, name ):
    """Reads in alpha vectors and finite state controller for POMDP specified in solver-input

    Args:
        solver_input (string): string containing specification of pomdp-problem to be run by solver
    """
    # write input to temporary directory
    temp_directory = 'pomdp' 
    if len(name) > 47:
        # really weird limit on the length of the .pomdp file name (????)
        name = name[-47:]
    output_file_path = os.path.join(os.getcwd(), temp_directory, name)
    input_file_path = os.path.join(os.getcwd(), temp_directory, '{}.pomdp'.format(name))
    pg_path = glob.glob(os.path.join(os.getcwd(), temp_directory, '{}.pg'.format(name)))
    alpha_path = glob.glob(os.path.join(os.getcwd(), temp_directory, '{}.alpha'.format(name)))

    if len(alpha_path) == 0:
        if not os.path.exists(temp_directory):
            os.mkdir(temp_directory)
        with open(input_file_path, 'w') as f:
            f.write(solver_input)
            f.close() 
        # import pdb; pdb.set_trace()
        # run program in temporary directory
        command = "{} -pomdp {} -o {}".format(PATH_TO_POMDP_SOLVER, input_file_path, output_file_path)
        print(command.split(" "))
        subprocess.run(command.split(" "))  
        # read in alpha vectors and policy graph
        pg_path = glob.glob(os.path.join(os.getcwd(), temp_directory, '{}.pg'.format(name)))[0]
        alpha_path = glob.glob(os.path.join(os.getcwd(), temp_directory,  '{}.alpha'.format(name)))[0]
    else:
        pg_path = pg_path[0]
        alpha_path = alpha_path[0]

    pg_f = open(pg_path, 'r')
    pg = pg_f.read()
    pg_f.close()

    alpha_f = open(alpha_path, 'r')
    alpha = alpha_f.read()
    alpha_f.close()

    # reading in list of actions from policy graph
    alpha_buf = io.StringIO(alpha)
    pg_buf = io.StringIO(pg)
    actions = []
    transitions = []
    alphas = []

    l = pg_buf.readline()

    while l != "":
        node_transitions = []
        l_list = l.split(' ')
        actions.append(int(l_list[1]))
        for obs, transition in enumerate(l_list[3:-1]):
            if transition != '-':
                node_transitions.append(int(transition))
            else: 
                node_transitions.append(None)
        transitions.append(node_transitions)
        l = pg_buf.readline()
    
    l = alpha_buf.readline()

    while l != "":
        alpha = alpha_buf.readline()
        alpha = alpha[:-2].split(' ') # removing newline character
        alpha = [float(a) for a in alpha]
        alphas.append(alpha)
        l = alpha_buf.readline() # reading in empty line
        l = alpha_buf.readline()


    alphas = np.vstack(alphas)
    transitions = np.vstack(transitions)

    # deleting temporary directory

    return dict(actions=actions, alphas=alphas, transitions=transitions)

