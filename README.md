# ATS-NIA
Node Injection Attacks on Graph Neural Networks via Adaptive Target Selection


This repository is our Pytorch implementation of our paper:
Node Injection Attacks on Graph Neural Networks via Adaptive Target Selection

# Abstract
Graph neural networks (GNNs) have demonstrated remarkable performance in various graph-structured data analysis and recognition tasks. However, they remain vulnerable to adversarial attacks. Most existing node injection attack methods adopt static target selection strategies or rely on a single target-selection criterion, which limits their flexibility in identifying attack-susceptible nodes. To overcome these limitations, this paper focuses on the black-box evasion attack scenario and proposes a novel framework, Adaptive Target Selection for Node Injection Attacks (ATS-NIA). ATS-NIA jointly models target selection, feature generation, and edge construction to effectively degrade the predictions of GNNs. Specifically, ATS-NIA selects vulnerable and influential target nodes by jointly considering uncertainty and centrality metrics, which characterize prediction instability and structural influence, respectively. Rather than relying on a fixed target list determined before the attack, ATS-NIA re-evaluates candidate nodes across multiple injection rounds to prioritize targets that are more susceptible to attack. Subsequently, ATS-NIA employs an adaptive feature generator to generate imperceptible and attack-effective node features. To further amplify the attack impact, ATS-NIA constructs perturbation edges between injected nodes and the original graph. The feature generation and edge construction components are jointly optimized within a reinforcement learning framework. Experiments on benchmark datasets show that ATS-NIA outperforms state-of-the-art attack methods.


# Requirements
matplotlib==3.7.1
numpy==1.26.4
pandas==1.5.1
scikit_learn==1.1.2
scipy==1.13.0
torch==1.12.1+cu113
torch_geometric==2.5.3
torch_sparse==0.6.14
tqdm==4.64.1

# RUN CODE
## please simple run the following
python main.py --dataset citeseer --surrogate_model GCN --victim_model GCN --update 3 --alpha 0.6 --ratio 0.03 
