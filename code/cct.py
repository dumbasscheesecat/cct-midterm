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
FIG_DIR = ROOT_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True) #i know it is not part of the folder structure instruction, but plots in terminal is not possible

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
    with model:
        idata = pm.sample(draws=draws, tune=tune, chains=chains, target_accept=target_accept)
    return idata

def get_var_names_from_idata(idata):
    return [v for v in idata.posterior.variables if v not in idata.posterior.coords]

#req5: analysis

def analyze_results(idata, data):
    print('Cultural Consensus Theory Model Summary')

    var_names = get_var_names_from_idata(idata)
    summary = az.summary(idata, var_names=var_names, hdi_prob=0.94)
    print(summary)

    az.plot_pair(idata, var_names=var_names, kind='kde', marginals=True, point_estimate='mean') 
    plt.gcf().suptitle(f"Pair Plot", y=1.02)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "pair_plot.png")
    plt.close()

#req5a: estimate informant competence

    D_means = idata.posterior['D'].mean(dim=('chain', 'draw')).values #gpt suggested
    print('Posterior mean competence D for each informant:', D_means)

    most_competent = np.argmax(D_means)
    least_competent = np.argmin(D_means)
    print(f"\nMost competent informant: P{most_competent+1} (D = {D_means[most_competent]:.2f})") #gpt suggested
    print(f"Least competent informant: P{least_competent+1} (D = {D_means[least_competent]:.2f})")

    az.plot_posterior(idata, var_names=['D'], hdi_prob=0.94)
    plt.tight_layout()
    plt.suptitle("Posterior Distributions of Informant Competence D", y=1.02)
    plt.savefig(FIG_DIR / "competence_posterior.png")
    plt.close()

#req5b: estimate consensus answers

    Z_means = idata.posterior['Z'].mean(dim=('chain', 'draw')).values
    print('Posterior mean consensus answer Z for each question:', Z_means)

    Z_estimated = (Z_means > .5).astype(int) #larger than by chance and rounding up as per instruction
    print('Estimated most likely consensus answer key:', Z_estimated)

    az.plot_posterior(idata, var_names=['Z'], hdi_prob=0.94)
    plt.tight_layout()
    plt.suptitle("Posterior Distributions of Consensus Answers Z", y=1.02)
    plt.savefig(FIG_DIR / "consensus_posterior.png")
    plt.close()

#req5c: compare naive aggregation

    majority_vote = (data.sum(axis=0) >= (data.shape[0] / 2)).astype(int) #gpt suggested
    print('Naive majority vote:', majority_vote)

    comparison = majority_vote == Z_estimated
    agreement = np.mean(comparison) #comparing majority vote answer key to posterior consensus answer key
    print('Agreement between data and inference', agreement)

#the whole thing

def run_analysis():
    data = load_plant_knowledge_data()
    model = build_model(data)
    idata = sample_posterior(model, draws=2000, tune=1000, chains=4, target_accept=0.9) #standard numbers in examples
    analyze_results(idata, data)
    return model, idata

if __name__ == "__main__":
    run_analysis()