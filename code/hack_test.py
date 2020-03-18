#!/usr/bin/env python
# coding: utf-8
import nibabel as nib
from nibabel.processing import resample_to_output
import numpy as np
import pandas as pd
from glob import glob
import re
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn import decomposition as dcm
from sklearn import preprocessing as pp
from sklearn import ensemble
import json
from nilearn.input_data import NiftiMasker
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
import pickle

# load the nifti data
masker = NiftiMasker(mask_img='../data/playground/overlap_mask_3mm.nii.gz')
subs_2d = masker.fit_transform('../data/playground/subs_3mm.nii.gz')


for csv in glob('../data/raw/subs-405*.csv'):
    behav_df = pd.read_csv(csv, converters={'id': lambda x: str(x).zfill(4)})
    behav_df.set_index('id', inplace=True)
    behav_df

    
    X_train, X_test, y_train, y_test = train_test_split(subs_2d, behav_df['score'].values, test_size=0.10)
    # preprocessing strategy
    # PCA
    pca = dcm.PCA(svd_solver='full')
    spca = dcm.SparsePCA()
    sparse_alpha_opts = [0.1, 0.5, 1, 2, 5, 10]
    kpca = dcm.KernelPCA()
    kernel_opts = ["linear", "rbf", "sigmoid"]
    n_component_opts = [0.7, 0.8, 0.9, 0.95, 0.99]
    lasso = linear_model.LassoCV(max_iter=100000, n_jobs=28, alphas=np.arange(0.1, 50, 0.1))
    lasso_lars = linear_model.LassoLarsCV(n_jobs=28, max_iter=100000, max_n_alphas=10000)
    eps_opts = [10.0, 5.0, 2.0, 1.5, 0.9, 0.1, 0.01, 0.001, 0.0001]
    elastic = linear_model.ElasticNetCV(alphas=np.arange(0.1, 50, 0.1), max_iter=100000)
    l1_ratio_opts = [0.1, 0.5, 0.9, 0.95, 0.99]
    lasso_lars_bay = linear_model.LassoLarsIC(max_iter=100000)
    cv_opts = [5, 10, 15, 20]

    param_dict = [{
        'clf': (lasso,),
        'pca': (spca,),
        'pca__alpha': sparse_alpha_opts,
        'clf__selection': ['random', 'cyclic'],
        'clf__cv': cv_opts,
    }, {
        'clf': (lasso,),
        'pca': (kpca,),
        'pca__kernel': kernel_opts,
        'clf__selection': ['random', 'cyclic'],
        'clf__cv': cv_opts,
    }, {
        'clf': (lasso_lars,),
        'pca': (spca,),
        'pca__alpha': sparse_alpha_opts,
        'clf__eps': eps_opts,
        'clf__cv': cv_opts,
    }, 
    {
        'clf': (lasso_lars,),
        'pca': (kpca,),
        'pca__kernel': kernel_opts,
        'clf__eps': eps_opts,
        'clf__cv': cv_opts,
    }, {
        'clf': (elastic,),
        'pca': (spca,),
        'pca__alpha': sparse_alpha_opts,
        'clf__l1_ratio': l1_ratio_opts,
        'clf__cv': cv_opts,
    }, 
    {
        'clf': (elastic,),
        'pca': (kpca,),
        'pca__kernel': kernel_opts,
        'clf__l1_ratio': l1_ratio_opts,
        'clf__cv': cv_opts,
    }, 
    ]

    pipeline = Pipeline([("pca", pca), ("clf", lasso)])

    search = GridSearchCV(pipeline, n_jobs=28, param_grid=param_dict, cv=5, scoring='r2')
    
    search.fit(X_train, y_train)

    search.score(X_test, y_test)

    out_file = os.path.basename(csv).replace('csv', 'sav')
    pickle.dump(search, open(out_file, 'wb'))

#{
#        'clf': (lasso_lars_bay,),
#        'pca__n_components': n_component_opts,
#        'clf__eps': eps_opts,
#        'clf__criterion': ['aic', 'bic'],
#    }
# ## Project Goals
# 
# ### Data
# 1. lesion maps
# 2. t1w images N=105 (and lesion masks in native and MNI space)
# 
# ### Previous Work
# - LESYMAP with core 15 tests
# - 435 factor analysis
# - several hundred in the pipeline
# 
# ### Current work
# - variable (how the lesion was mapped (either drawn directly on template or the T1w image))
# - how to automatically draw lesions
# - generate a list of modalities of data collected (CT, MRI, traced or projected...)
# - 435 participants that have been added
# 
# ### Other Work
# - Isles Challenge
# 
# ### Acute versus chronic lesion
# - Acute: diffusion is bright (anisotropic), adc (dark)
# - Chronic: core of lesion (tapering from dark voxels to normal)
# - baseline matching rate (75% agreement between expert raters)
# 
# ### Lesion versus non-lesion
# - reliance on multi-modal structures
# - best modality (FLAIR or diffusion)
#   - typically used (T1w)
# 
#  ### CNNs
#  - CNN
#    - first level usually edges
#    - then orientation
#    - the higher order properties
#    - position?
#    
#    
