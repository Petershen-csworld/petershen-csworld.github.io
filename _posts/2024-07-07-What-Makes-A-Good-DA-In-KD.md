---
layout: post
title: Paper not: What Makes A Good DA In KD
date: 2024-07-07 17:00:00
description: A reading note of the paper.
tags: Papers NIPS2022
categories: sample-posts
---

# Introduction

**Knowledge Distillation** has become a state-of-the-art method for training light-weight models. After it is formally put forward by Hinton[1], lots of work has been discussing designing a better KD loss. However, this paper attempts to understand it from the **Input Side**. Specifically, its interplay with DA(Data Augmentation) is not yet well understood. Based on the motivation, the paper makes the following contributions.
- Prove rigorously that **A Good DA scheme should reduce the covariance of the teacher-student cross-entropy**. 
- Introduce a new metric: T.stddev (stddev of teacher's mean probability).
- Propose a new entropy based DA scheme: CutMixPick to further enhance CutMix. 




# Theoretical Analysis 

## Multi-Class Classification with KD

In a multi-class classification setting, given a training set $ S = \{(x_n, y_n)\}_{n=1}^{N} \sim \mathscr{D}^{N} $ and a predictor $\bold{f}: \chi \rightarrow \mathbb{R}^{C}$, where $ \chi $ is the input space and $ C $ refers to the number of classes.

The predictor $\bold{f}$ is supposed to minimize the $ \textit{true risk} $
$$
R_{\mathscr{D}}(\bold{f}) \overset{def}{=} \mathbb{E}_{(x,y) \sim \mathscr{D}}[L(y, \bold{f}(x))],
$$

where L stands for the loss objective function(e.g., cross-entropy).

For training, the predictor aims to minimize the $ \textit{empirical risk} $ defined on the training sequence :
$$
R_{S}{(\bold{f})} \overset{def}{=} - \frac{1}{N} \sum_{n=1}^{N}\bold{e}_{y_n}^\top log(\bold{f}(x_n)),
$$

where $ \bold{e}_{y_n} \in \{0,1\}^C$ is one-hot vector indicating the label $y \in [C] = \{ 1,2, \cdots C \}$.

In the context of KD, the one-hot hard target vector is replaced with a probability vector $\bold{p}^{(t)}(x) \in \mathbb{R}_{+}^{C}$. giving the $ \textit{empirical distilled risk} $  of $ \bold{f} $:

$$
\hat{R}_{S}^(\bold{f}) \overset{def}{=} - \frac{1}{N} \sum_{n=1}^{N} \bold{p}^{(t)}(x_n)^{\top} log(\bold{f}(x_n))
$$

# References
[1] **Distilling the Knowledge in a Neural Network** Geoffrey Hinton et al.



