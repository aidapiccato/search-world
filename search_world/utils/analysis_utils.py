import os
from glob import glob
import pickle as pkl
import pandas as pd

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
            dataframe_fn = glob(os.path.join(data_fn, 'dataframe'))[0]
            if len(dataframe_fn) == 0:
                dataframe_fn = None
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
            scalar_dict.update(scalar_dict['env'])
            scalar_dict.update(scalar_dict['model'])
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
    