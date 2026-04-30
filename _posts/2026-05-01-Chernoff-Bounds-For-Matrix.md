---
layout: post
title: Chernoff bounds for matrix
date: 2026-04-30 00:25:00
description: Matrix Concentration Inequalities
tags: formatting charts
categories: statistics
chart:
plotly: true
thumbnail: assets/img/9.jpg
---


<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/liandengdog.png" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    連登狗
</div>
# Bounding the maximum eigenvalue 
The tail bound of maximum eigenvalue

$$
\begin{aligned}
Pr(\lambda_{\max}(Y) \geq t) &\leq \inf_{\theta>0} \mathbb{E}[\exp(\theta\lambda_{\max}(Y)] \exp{(-\theta t)} \\
& = \inf_{\theta > 0} \mathbb{E}[\lambda_{\max}(\exp(\theta Y))] \exp(-\theta t)\\
& \leq \inf_{\theta>0} \mathbb{E}[\operatorname{tr}(\exp(\theta Y))]\exp(-\theta t) \\
\end{aligned}
$$

For the expectation of maximum eigenvalue.


From Jensen's Inequality, we have

$$
\exp(\theta \mathbb{E}[\lambda_{\max}(Y)]) \leq \mathbb{E}\exp(\theta \lambda_{\max}(Y))
$$

Rearrange the terms and follow similar steps taken in deriving the tail bound of maximum eigenvalue.

$$
\begin{aligned}
\mathbb{E}[\lambda_{\max}(Y)] & \leq \inf_{\theta > 0} \frac{1}{\theta}\log\mathbb{E}\lambda_{\max}(\exp(\theta Y)) \\
\mathbb{E}[\lambda_{\max}(Y)] & \leq \inf_{\theta > 0}\frac{1}{\theta} \log \mathbb{E}\operatorname{tr}(\exp(\theta Y))\\
\end{aligned}
$$


# Lieb’s Theorem  
  
We state the following result without proof; a detailed derivation deserves a separate discussion due to its technical depth.  
  
Let $H$ be a Hermitian matrix. Then the function  

$$ 
A \;\mapsto\; \operatorname{tr}\!\big(\exp(H + \log A)\big)  
$$

is **concave** in the positive definite matrix \( A \).  


## Corollary 1  
  
By applying Jensen’s inequality to the concave function above, we obtain  

$$
\mathbb{E}\!\left[ \operatorname{tr}\!\big(\exp(H + \log A)\big) \right] 
\;\leq\;  
\operatorname{tr}\!\left(\exp\!\big(H + \log \mathbb{E}[A]\big)\right).  
$$
  
Now consider the substitution $A = \exp(Y)$. Then the inequality becomes 

$$
\mathbb{E}\!\left[ \operatorname{tr}\!\big(\exp(H + Y)\big) \right]  
\;\leq\;  
\operatorname{tr}\!\left(\exp\!\big(H + \log \mathbb{E}[\exp(Y)]\big)\right).  
$$

## Corollary 2: Subadditivity of Matrix Cgf

We observe that in Corollary~1, by setting \( H = 0 \) (which is trivially Hermitian), we obtain  

$$
\mathbb{E}\!\left[\operatorname{tr}\!\big(\exp(Y)\big)\right]  
\;\leq\;  
\operatorname{tr}\!\left(\exp\!\big(\log \mathbb{E}[\exp(Y)]\big)\right).  
$$
  
We now extend this inequality to a more general setting with $n$ independent Hermitian matrices $X_k$ of the same dimension. The claim is  

$$
\mathbb{E}\,\operatorname{tr}\!\left(\exp\!\big(\sum_k \theta X_k\big)\right)  
\;\leq\;  
\operatorname{tr}\!\left(\exp\!\big(\sum_k \log \mathbb{E}[\exp(\theta X_k)]\big)\right).  
$$
 
**Proof (by induction).**
  
**Base case** $k = 1$

$$
\mathbb{E}\,\operatorname{tr}\!\big(\exp(\theta X_1)\big)  
\;\leq\;  
\operatorname{tr}\!\left(\exp\!\big(\log \mathbb{E}[\exp(\theta X_1)]\big)\right),  
$$  
which follows directly from Corollary1.  

**Inductive step**

Assume the result holds for $k - 1$ matrices. We prove it for $k$  matrices.  


$$
\begin{aligned}  
\mathbb{E}\,\operatorname{tr}\!\left(\exp\!\big(\sum_{i=1}^k \theta X_i\big)\right)  
&=  
\mathbb{E}\,\operatorname{tr}\!\left(\exp\!\big(\sum_{i=1}^{k-1} \theta X_i + \theta X_k\big)\right) \\  
  
&=  
\mathbb{E}_{X_{1:k-1}}\,\mathbb{E}_{X_k}\,  
\operatorname{tr}\!\left(\exp\!\big(\sum_{i=1}^{k-1} \theta X_i + \theta X_k\big)\right)  
\quad \text{(tower property)}  
\end{aligned}  
$$


Applying Corollary1 (conditioning on $X_{1:k-1}$, we have  


$$
\begin{aligned}  
\mathbb{E}_{X_k}\,  
\operatorname{tr}\!\left(\exp\!\big(\theta X_k + \sum_{i=1}^{k-1} \theta X_i\big)\right)  
&\leq  
\operatorname{tr}\!\left(  
\exp\!\big(  
\sum_{i=1}^{k-1} \theta X_i  
+ \log \mathbb{E}_{X_k}[\exp(\theta X_k)]  
\big)  
\right)  
\quad \text{(Corollary~1)}  
\end{aligned}  
$$ 
  
Therefore,  

$$
\begin{aligned}  
\mathbb{E}_{X_{1:k-1}}\,\mathbb{E}_{X_k}\,  
\operatorname{tr}\!\left(\exp\!\big(\sum_{i=1}^{k-1} \theta X_i + \theta X_k\big)\right)  
&\leq  
\mathbb{E}_{X_{1:k-1}}  
\operatorname{tr}\!\left(  
\exp\!\big(  
\sum_{i=1}^{k-1} \theta X_i  
+ \log \mathbb{E}_{X_k}[\exp(\theta X_k)]  
\big)  
\right).  
\end{aligned}  

$$


Applying the same argument recursively to $X_{1}, \dots, X_{k-1}$, we obtain  

$$  
\mathbb{E}\,\operatorname{tr}\!\left(\exp\!\big(\sum_{i=1}^k \theta X_i\big)\right)  
\;\leq\;  
\operatorname{tr}\!\left(\exp\!\big(\sum_{i=1}^k \log \mathbb{E}[\exp(\theta X_i)]\big)\right).  
$$ 

This completes the proof.



# Master Bounds for Sums of Independent Random Matrices


$$
\mathbb{E} \lambda_{\max}(\sum_k X_k) \leq \inf_{\theta > 0} \frac{1}{\theta} \log \operatorname{tr} \exp(\sum_k \log \mathbb{E} \exp(\theta X_k))
$$

The result could be proved by the chernoff bounds we introduced in the beginning part and Corollary 2 of Lieb's Theorem.




# References

[1] https://arxiv.org/pdf/1501.01571 Chapter 3