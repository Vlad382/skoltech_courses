import numpy as np
import pandas as pd

import random
import time
import datetime
from tqdm.notebook import tqdm

from sklearn.model_selection import train_test_split, GridSearchCV, ParameterGrid, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, mutual_info_score, f1_score, accuracy_score
from sklearn.cluster import AgglomerativeClustering
from sklearn.multiclass import OneVsOneClassifier, OneVsRestClassifier

np.random.seed(42)
random.seed(42)

plt.rcParams['figure.figsize'] = (11, 7)
plt.rcParams['font.size'] = 20
plt.rcParams['axes.linewidth'] = 2
plt.rcParams['grid.linewidth'] = 2
plt.rcParams['lines.linewidth'] = 3

plt.rcParams['xtick.major.size'] = 5
plt.rcParams['xtick.major.width'] = 2
plt.rcParams['xtick.minor.size'] = 3
plt.rcParams['xtick.minor.width'] = 2

plt.rcParams['ytick.major.size'] = 5
plt.rcParams['ytick.major.width'] = 2
plt.rcParams['ytick.minor.size'] = 3
plt.rcParams['ytick.minor.width'] = 2

def load_data(path, hidden_test=False):
    data = pd.read_table(path)
    data.rename(columns={data.columns[0]: 'all'}, inplace=True)
    
    data['size_'] = data['all'].apply(lambda x: str(x).split()[0]).astype(float)
    data['mass'] = data['all'].apply(lambda x: str(x).split()[1]).astype(int)
    
    if not hidden_test:
        data['type'] = data['all'].apply(lambda x: str(x).split()[2])

    data.drop(columns='all', inplace=True)
    return data   

def preprocess_data(X, y, normalize=True, test_size=.3, stratify=None):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, stratify=stratify)
    
    if normalize:
        scaler = StandardScaler()
        scaler.fit(X_train)

        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)
    return X_train, X_test, y_train, y_test

def aggregate_form(type_, size_, mass_):
    types_dict = {
        'A_head': 0,
        'A_tail': 1,
        'B_head': 2,
        'B_tail': 3,
        'C_head': 4,
        'C_tail': 5
    }    
    if type_ == 'A':
        point_type = type_ + '_tail' if mass_ > 3.6 else type_ + '_head'
        
    elif type_ == 'B':
        position = position_wrt_line([2.8, 3.5], [3.3, 4.2], [size_, mass_])
        point_type = type_ + '_tail' if position == 'left' else type_ + '_head'
        
    else:
        point_type = type_ + '_tail' if mass_ > 3.85 else type_ + '_head'

    return types_dict[point_type]

def position_wrt_line(p1, p2, p):
    B_x = p2[0] - p1[0]
    B_y = p2[1] - p1[1]
    P_x = p[0] - p1[0]
    P_y = p[1] - p1[1]
    
    cross_prod = B_x * P_y - B_y * P_x
    pos = 'right' if cross_prod > 0 else 'left'
    return pos

def main(config, strategy, X_train, y_train, X_test, y_test):
    assert strategy in ['clustering', 'classification'], 'Wrong ML strategy'
    grid_search_func = grid_search_clustering if strategy == 'clustering' else grid_search_classification
    grid_search_func = exec_time(grid_search_func)
    
    if strategy == 'classification':
        metrics_names = ['f1_micro_train', 'f1_macro_train', 'f1_micro_test', 'f1_macro_test', 'acc_train', 'acc_test']
    else:
        metrics_names = ['silh_train', 'silh_test', 'mi_train', 'mi_test', 'acc_train', 'acc_test']
    metrics_names.append('exec_time')
    metrics_names.append('cv_grid_size')
    
    model_names = config['model_names']
    models = config['models']
    params = config['params']
    hyper_params = config['hyper_params']
    cv = config['cv']
    verbose = config['verbose']
    multiclass_str = config['multiclass_str']
    
    metrics_res = []
    progress_bar = tqdm(zip(model_names, models, params, hyper_params, multiclass_str))
    for n, m, p, hp, multiclass_str in progress_bar:
        progress_bar.set_description(f'Processing {n}')
        
        grid_size = 1
        for v in hp.values():
            grid_size *= len(v)
        
        (_, metrics), d = grid_search_func(n, m, p, hp, X_train, y_train, X_test, y_test, cv, verbose, multiclass_str)
        metrics = np.round(metrics, 4)
        metrics = np.hstack([metrics, d, grid_size])
        metrics_res.append(metrics)
    metrics_res = np.vstack(metrics_res)
    
    results = pd.DataFrame({k: v for k, v in zip(metrics_names, metrics_res.T[:])}, index=model_names)
    return results
    
def grid_search_classification(name, model, params, hyper_params, X_train, y_train, X_test, y_test, cv=5, verbose=True, multiclass_str=False):
    multiclass_str = False if not multiclass_str else OneVsRestClassifier if 'one-vs-rest' else OneVsOneClassifier
    m = model(**params) if not multiclass_str else multiclass_str(model(**params), n_jobs=-1) 
    
    g_search_m = GridSearchCV(m, hyper_params, cv=cv, scoring='f1_micro', n_jobs=-1)
    g_search_m.fit(X_train, y_train)
    
    best_params = g_search_m.best_params_
    if verbose:
        print(f'  Model: {name}, f1_micro_score: {g_search_m.best_score_:.5f}, best_params: {best_params}')
    
    best_params = best_params if not multiclass_str else {k.split('__')[-1]: v for k, v in zip(best_params.keys(), best_params.values())}
    m = model(**params, **best_params) if not multiclass_str else multiclass_str(model(**params, **best_params), n_jobs=-1) 
    m.fit(X_train, y_train)

    y_train_hat = m.predict(X_train)
    y_test_hat = m.predict(X_test)
    
    f1_micro_train, f1_macro_train, f1_micro_test, f1_macro_test, acc_train, acc_test = multiclass_metrics(y_train, y_train_hat, y_test, y_test_hat)
    return m, [f1_micro_train, f1_macro_train, f1_micro_test, f1_macro_test, acc_train, acc_test]

def grid_search_clustering(name, model, params, hyper_params, X_train, y_train, X_test, y_test, cv=5, verbose=True, multiclass_str=None):
    assert not isinstance(model, AgglomerativeClustering), 'Please, not this model!'
    
    best_params, silh_score = custom_grid_search_loop(model, params, hyper_params, X_train, y_train, cv=5)
    if verbose:
        print(f'  Model: {name}, silhouette_score: {silh_score:.5f}, best_params: {best_params}')
    
    m = model(**params, **best_params)
    y_train_hat = m.fit_predict(X_train)
    y_test_hat = m.predict(X_test)
    
    silh_train = silhouette_score(X_train, y_train_hat)
    silh_test = silhouette_score(X_test, y_test_hat)
    
    mi_train = mutual_info_score(y_train, y_train_hat)
    mi_test = mutual_info_score(y_test, y_test_hat)
    
    acc_train = accuracy_score(y_train, y_train_hat)
    acc_test = accuracy_score(y_test, y_test_hat)
    return m, [silh_train, silh_test, mi_train, mi_test, acc_train, acc_test]

def custom_grid_search_loop(model, params, hyper_params, X_train, y_train, cv=5):
    kf = KFold(n_splits=cv)
    hyper_params_grid = ParameterGrid(hyper_params)
    
    list_p_silh = []
    for p in hyper_params_grid:
        list_silh = []
        
        for train_index, test_index in kf.split(X_train):
            X_train_val, X_test_val = X_train[train_index], X_train[test_index]
            y_train_val, y_test_val = y_train[train_index], y_train[test_index]

            m = model(**params, **p)
            m.fit(X_train_val)
            y_test_hat = m.predict(X_test_val)
            
            list_silh.append(silhouette_score(X_test_val, y_test_hat))    
        list_p_silh.append(np.mean(list_silh))
    
    best_par_ind = np.argmax(list_p_silh)
    best_par = hyper_params_grid[best_par_ind]
    return best_par, np.max(list_p_silh)

def multiclass_metrics(y_train_true, y_train_hat, y_test_true, y_test_hat):
    f1_micro_train = f1_score(y_train_true, y_train_hat, average='micro')
    f1_macro_train = f1_score(y_train_true, y_train_hat, average='macro')
    
    f1_micro_test = f1_score(y_test_true, y_test_hat, average='micro')
    f1_macro_test = f1_score(y_test_true, y_test_hat, average='macro')
    
    acc_train = accuracy_score(y_train_true, y_train_hat)
    acc_test = accuracy_score(y_test_true, y_test_hat)
    return f1_micro_train, f1_macro_train, f1_micro_test, f1_macro_test, acc_train, acc_test

def exec_time(func):
    def wrap(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        delta = str(datetime.timedelta(seconds=(end-start))).split('.')[0]
        return result, delta
    return wrap