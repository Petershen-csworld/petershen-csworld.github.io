---
layout: post
tags:
  - physics
title: EULER-LAGRANGE EQUATIONS
categories: physics,  kinetics
chart:
plotly: true
date: 2026-07-03 14:24:00
description: Derivations of euler-lagrange theroem under classical particle theory and field theory (TBD.)
---


This page is under constructed---

## 1. Particle theory


We define the Lagranian as follows: $L(q_i, \dot q_i) := T(\dot q_i) - V(q_i)$ , here $T$ is the kinetic energy and $V$ is the potential energy. We are interested in the path between $t_1$ and $t_2$, that is we want to know $q(t)$ in this interval, with the constraint that $q_i(t_1)$ and $q_i(t_2)$ are known.


The principle of least action says that action $S$, defined as a function of the path that can be followed, achieves minimum.

$$
S := \int_{t_1}^{t_2} L(q_i,\dot q_i)dt.
$$
The condition that $S$ is minimum requires $\delta S=0$, which means any slight variation of paths would increase $S$. 

Next we calculate $\delta S$, the following derivations are not too rigorous but we could trust the mathematicians:
$$
\begin{aligned}
\delta S &= \delta[\int_{t_1}^{t_2} Ldt] \\
		 &= \int_{t_1}^{t_2}\delta Ldt \\
		 &=\int_{t_1}^{t_2} (\frac{\partial L}{\partial q_i}\delta q_i + \frac{\partial L}{\partial \dot q_i}\delta \dot q_i)dt \\
		 &= \int_{t_1}^{t_2} (\frac{\partial L}{\partial q_i}\delta q_i + \frac{\partial L}{\partial \dot q_i}\frac{d (\delta q_i)}{dt})dt
\end{aligned}
$$
Now we integrate the RHS terms:

$$
\delta S = \int_{t_1}^{t_2}\frac{\partial L}{\partial q_i}\delta q_i dt + \int_{t_1}^{t_2}\frac{\partial L}{\partial \dot q_i}\delta \dot q_idt =  \int_{t_1}^{t_2}\frac{\partial L}{\partial q_i}\delta q_i dt + \frac{\partial L}{\partial \dot q_i}\delta q_i |^{t_2}_{t_1} - \int_{t_1}^{t_2}\frac{d}{dt}(\frac{\partial L}{\partial \dot q_i})\delta q_idt
$$

Note that we have the constraint that $q_i(t_2)$ and $q_i(t_1)$ are fixed. 
$$
\delta S = \int_{t_1}^{t_2}(\frac{\partial L}{\partial q_i}\delta q_i - \frac{d}{dt}(\frac{\partial L}{\partial \dot q_i})\delta q_i)dt
$$
This must be true for all possible $\delta q_i$ so the quantities in brackets must be zero, the equation to be integrated must be zero. Hence we have

$$
\frac{\partial L}{\partial q_i} - \frac{d}{dt} (\frac{\partial L}{\partial \dot q_i}) = 0
$$



