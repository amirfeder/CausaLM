## CausaLM: Causal Model Explanation Through Counterfactual Language Models

### Authors: 
#### [Amir Feder](https://scholar.google.com/citations?user=ERwoPLIAAAAJ&hl=en&oi=ao), [Nadav Oved](https://scholar.google.com/citations?user=9DgSB7sAAAAJ&hl=en), [Uri Shalit](https://shalit.net.technion.ac.il/people/), [Roi Reichart](https://ie.technion.ac.il/~roiri/)

### Abstract:
Understanding predictions made by deep neural networks is notoriously difficult, but also crucial to their dissemination. As all ML-based methods, they are as good as their training data, and can also capture unwanted biases. While there are tools that can help understand whether such biases exist, they do not distinguish between correlation and causation, and might be ill-suited for text-based models and for reasoning about high level language concepts. A key problem of estimating the causal effect of a concept of interest on a given model is that this estimation requires the generation of counterfactual examples, which is challenging with existing generation technology. To bridge that gap, we propose CausaLM, a framework for producing causal model explanations using counterfactual language representation models. Our approach is based on fine-tuning of deep contextualized embedding models with auxiliary adversarial tasks derived from the causal graph of the problem. Concretely, we show that by carefully choosing auxiliary adversarial pre-training tasks, language representation models such as BERT can effectively learn a counterfactual representation for a given concept of interest, and be used to estimate its true causal effect on model performance. A byproduct of our method is a representation that is unaffected by the tested concept, which can be useful in mitigating unwanted bias ingrained in the data.

### Links: (coming soon)

#### [Paper](https://arxiv.org/abs/2005.13407)

#### [Code](https://github.com/amirfeder/CausaLM)

#### [Data](https://www.kaggle.com/amirfeder/causalm)
