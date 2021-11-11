import os
from glob import glob
import numpy as np
import pickle as pkl
import pandas as pd
from scipy.stats import wasserstein_distance

def get_trials_features(df):
    """Given a dataframe of trials, returns updated trials with new features extracted from raw data. TASK SPECIFIC"""
    df = df.copy()
    df['agent_dist'] = df.apply(lambda x: len(x.action) - 1, axis=1)
    return df


def get_dataset_paths(job_ids='', base_dir='../logs'):    
    """Returns list of scalar/vector data filenames, config filenames for a given job_id. If job_id is not specified, will return 
    all config/vector/scalar files in the logs directory. These are sorted such that correspondig scalar/vector/config
    filenames have the same index
    """ 
    data_dirs = [os.path.join(base_dir, job_id) for job_id in job_ids]
    fns = []
    for data_dir in data_dirs:
        data_fns = [fname for root, _, _ in os.walk(data_dir) for fname in glob(os.path.join(root, 'data'))]
        for data_fn in data_fns:
            vector_fn = glob(os.path.join(data_fn, 'vector'))[0]
            scalar_fn = glob(os.path.join(data_fn, 'scalar'))[0]
            dataframe_fn = glob(os.path.join(data_fn, 'dataframe'))
            if len(dataframe_fn) == 0:
                dataframe_fn = None
            else:
                dataframe_fn = dataframe_fn[0]
            fns.append(dict(data_fn=data_fn, vector_fn=vector_fn, scalar_fn=scalar_fn, dataframe_fn=dataframe_fn))
    return fns

def _flatten_dict(d):
    """Flattens any nested dictionaries in dictionary d into a single-level dictionary. Only flattens a single level"""
    d_copy = {}
    t = {}
    for k, v in d.items():
        if isinstance(v, dict):
            for nested_k, nested_v in v.items():        
                d_copy.update({nested_k: nested_v})
        else:
            d_copy.update({k: v})      
    return d_copy

def get_trials_dataframe(fns, overwrite=True):
    """Aggregates data for a particular job or set of jobs into a single dataframe, where each row corresponds to a trial. This method is specific to the search-world task

    Args:
        vector_fns (list): List of strings corresponding to paths to vector datasets
        scalar_fns (list): List of strings corresponding to paths to scalar datasets
    """

    vector_dfs = []
    scalar_dicts = []
    for dataset_index, fn_dict in enumerate(fns):
        
        with open(fn_dict['scalar_fn'], 'rb') as f:
            scalar_dict = pkl.load(f)
            scalar_dict.update({'dataset_index': dataset_index})
            scalar_dict.update(_flatten_dict(scalar_dict['env']))
            scalar_dict.update(_flatten_dict(scalar_dict['model']))
            scalar_dicts.append(scalar_dict)

        
        if overwrite == False and fn_dict['dataframe_fn'] is not None:
            with open(fn_dict['dataframe_fn'], 'rb') as f:
                df = pkl.load(f)

        else:
            with open(fn_dict['vector_fn'], 'rb') as f:
                vector_dicts = pkl.load(f)
            for d in vector_dicts:
                d = _flatten_dict(d)

            df = pd.DataFrame(vector_dicts)
            df['dataset_index'] = dataset_index
            df['trial_index'] = df.done.cumsum().shift(fill_value=0)

            constants = ['trial_index', 'dataset_index']            
            changes = df.columns.difference(constants).tolist()

            f = lambda x: x.tolist() if len(x) > 1 else x

            # flattening dataframe into trial rows
            df = df.groupby(constants)[changes].agg(f).reset_index()

            # remove trials that never ended
            df['done'] = df.done.apply(lambda x: [x] if isinstance(x, bool) else x)
            df = df[df.apply(lambda x: True in x.done, axis=1)].reset_index(drop=True)

            with open(os.path.join(fn_dict['data_fn'], 'dataframe'), 'wb') as f:
                pkl.dump(df, f)

        vector_dfs.append(df)

    scalar_df = pd.DataFrame(scalar_dicts)
    
    df = pd.concat(vector_dfs).reset_index(drop=True)
    
    df = df.merge(scalar_df, on='dataset_index')
    
    return df     
    
def _euclid_dist(a, b):
    return np.linalg.norm(a - b)

def _mask(n, p=0.5):
    a = np.zeros(n, dtype=int)
    a[:int(p * n)] = 1
    np.random.shuffle(a)
    a = a.astype(bool)
    return a

def _get_error(condition_df, g):
    opt = condition_df.loc[g.name[:-1] + ('OptimalAgent', )].agent_dist_hist if g.name[:-1] + ('OptimalAgent', ) in condition_df.index else None
    if opt is not None:
        return wasserstein_distance(g.agent_dist_hist, opt)
    return None

def get_condition_features(condition_df, trials_df):
    condition_df = condition_df.copy()

    bins = np.arange(trials_df['agent_dist'].min()-1, trials_df['agent_dist'].max()+1)

    condition_df['agent_dist_hist'] = condition_df.apply(lambda r: list(np.histogram(r.agent_dist, bins=bins, density=True)[0]), axis=1).to_frame()
    condition_df['error'] = condition_df.apply(lambda g: _get_error(condition_df, g), axis=1)
    return condition_df

def get_condition_df(trials_df, condition):
    name_condition = condition.copy()
    name_condition.append('name')
    cond_df = trials_df.groupby(name_condition).agg({'agent_dist': lambda g: list(g)}).reset_index()
    return cond_df.pivot(index=condition, columns='name').stack().reset_index().set_index(condition)

def get_consistency(condition_df, trials_df, condition,):
    """Produces a consistency dataframe and matrix

    Args:
        trials_df ([type]): [description]
        condition ([type]): [description]
        metric ([type]): [description]
    """

    bins = np.arange(trials_df['agent_dist'].min()-1, trials_df['agent_dist'].max()+1)
    
    names = condition_df.index.unique(level=-1)
    names = names[names != 'OptimalAgent']
    consistency_df = np.zeros((len(names), len(names))) 
    for idx_1, agent_1 in enumerate(names):
        for idx_2, agent_2 in enumerate(names): 
            if agent_1 == agent_2: 
                agent_df = condition_df[condition_df.index.get_level_values('name') == agent_1]           
                masks = agent_df.apply(lambda g: _mask(len(g.agent_dist)), axis=1)
                df_1 = agent_df.apply(lambda g: np.asarray(g.agent_dist)[masks.loc[g.name]], axis=1).to_frame('agent_dist')
                df_2 = agent_df.apply(lambda g: np.asarray(g.agent_dist)[~masks.loc[g.name]], axis=1).to_frame('agent_dist')
                df_1['agent_dist_hist'] = df_1.apply(lambda r: list(np.histogram(r.agent_dist, bins=bins, density=True)[0]), axis=1).to_frame()
                df_2['agent_dist_hist'] = df_2.apply(lambda r: list(np.histogram(r.agent_dist, bins=bins, density=True)[0]), axis=1).to_frame()
                
                df_1['error'] = df_1.apply(lambda g: _get_error(condition_df, g), axis=1)
                df_2['error'] = df_2.apply(lambda g: _get_error(condition_df, g), axis=1)
                merged = df_1.merge(df_2, on=condition, suffixes=['_split_1', '_split_2'])
                merged = merged.dropna()
                consistency_df[idx_1, idx_2] = merged.error_split_1.corr(merged.error_split_2)
            else: 
                df_1 = condition_df[condition_df.index.get_level_values('name') == agent_1]['error'].to_frame()
                df_2 = condition_df[condition_df.index.get_level_values('name') == agent_2]['error'].to_frame()
                merged_ = df_1.merge(df_2, on=condition, suffixes=['_'+agent_1, '_'+agent_2])
                merged_ = merged_.dropna()
                consistency_df[idx_1, idx_2] = merged_['error_'+agent_1].corr(merged_['error_'+agent_2])

    consistency_df = pd.DataFrame(consistency_df, columns=names, index=names)
    return consistency_df
