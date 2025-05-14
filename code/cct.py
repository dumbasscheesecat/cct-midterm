import pymc as pm
import arviz as az
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

#req1: function to load and return data

ROOT_DIR = Path(__file__).parent.parent
DATA_DIR = ROOT_DIR / "data"

def load_plant_knowledge_data():
    df = pd.read_csv(DATA_DIR / "plant_knowledge.csv")
    data = df.drop(columns=['Informant']).values #full.py for use case
    return data

print(load_plant_knowledge_data()) #checking

#req2/3: define priors and build model

def build_model(data):
    N, M = data.shape
    with pm.Model() as model_cct:
        D = pm.Uniform('D', lower=.5, upper=1, shape=N) #at least by chance performance to expertise
        Z = pm.Bernoulli('Z', p=.5, shape=M) #common practice
        D_reshaped = D[:, None] 
        p = Z * D_reshaped + (1 - Z) * (1 - D_reshaped) 
        X = pm.Bernoulli('X', p=p, observed=data)
    return model_cct

#req4: inference

def sample_posterior(model, draws, tune, chains, target_accept):

#req5: analysis

#req5a: estimate informant competence

#req5b: estimate consensus answers

#req5c: compare naive aggregation