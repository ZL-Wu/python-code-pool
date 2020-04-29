<head>
    <script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>
    <script type="text/x-mathjax-config">
        MathJax.Hub.Config({
            tex2jax: {
            skipTags: ['script', 'noscript', 'style', 'textarea', 'pre'],
            inlineMath: [['$','$']]
            }
        });
    </script>
</head>
---
title: "Logistic Regression (sklearn instruction)"
date: 2020-04-22T00:19:25-04:00
tags: ["logistic regression"]
series: ["Machine Learning"]
categories: ["Machine Learning"]
img: ""
toc: true
summary: "This is a summary of using logistic regression libraries in sklearn. Focus on the matters that should be paid attention to in the tuning."
draft: true
---

Reference: [sklearn's linear model](https://scikit-learn.org/stable/modules/linear_model.html#)

[Sklearn-LR code example]()

In the previous article [Logistic Regression Principle](./MLA_Logistic_Regression.md), it summarized the principle of logistic regression. Here is a summary of using logistic regression libraries in sklearn. Focus on the matters that should be paid attention to in the tuning.

## 1. Overview
There are mainly 3 Class of logistic regression:
- LogisticRegression
- LogisticRegressionCV
- logistic_regression_path

The main difference between LogisticRegression and LogisticRegressionCV is that LogisticRegressionCV uses cross-validation to select the regularization coefficient C. LogisticRegression needs to specify a regularization coefficient each time. Except this coefficient C, LogisticRegression and LogisticRegressionCV are basically used in the same way.

The logistic_regression_path class is special. After fitting the data, it cannot directly make predictions, and can only select the appropriate logistic regression coefficient and regularization coefficient for the fitted data. It is mainly used in model selection. This class is generally not used.

## 2. Regularization Parameter: Penalty
LogisticRegression and LogisticRegressionCV have regularization terms by default. Parameter penalty can be 'l1','l2','elasticnet','none'. Default value is 'l2'.

- **$l_1$**:
$$ \min_{w,c}\|w\|_1  + C\sum_{i=1}^mlog(exp(-y_i(X_i^Tw+c))+1) $$

- **$l_2$**:
$$ \min_{w,c}\frac{1}{2}w^Tw + C\sum_{i=1}^mlog(exp(-y_i(X_i^Tw+c))+1) $$

- **elasticnet**: a combination of L1 and L2, and minimizes the following cost function:
$$ \min_{w,c} \frac{1-\rho}{2}w^Tw + \rho\|w\|_1  + C\sum_{i=1}^mlog(exp(-y_i(X_i^Tw+c))+1) $$

>When tuning the parameters, if the main purpose is to solve the overfitting, it is generally enough to choose L2 regularization for the penalty. However, if L2 regularization is still overfitting, that is, if the prediction performance is poor with L2, then L1 regularization can be considered.

>In addition, if the model has lots of features, we hope that some unimportant features' coefficients be zeroed, so that the model coefficients can be sparse, L1 regularization can also be used in this case.


The choice of penalty parameter will affect the choice of loss function optimization algorithm (solver parameter):
- If penalty="$l_2$" (Ridge Regression), there are 4 optional algorithms (solver):
    - "newton-cg"
    - "lbfgs"
    - "liblinear"
    - "sag"

- If penalty="$l_1$" (Lasso Regression), there are only 1 optional algorithms (solver): **solver="liblinear"**. Because the loss function with L1 regularization is not derivable, only "liblinear" can solve it. the other three methods ["newton-cg", "lbfgs", "sag"] requires the first or second continuous derivative of the loss function.

## 3. Optimization Algorithm Parameter: Solver
There are 5 algorithms can be choosed to optimize Logistic Regression's Loss Function:

- **"newton-cg"**: A type of Newton's method family, iteratively optimizes the loss function by using the second derivative matrix of the loss function, the Hessian matrix.
- **"lbfgs"**: A kind of quasi-Newton method, iteratively optimizes the loss function by using the second derivative matrix of the loss function, the Hessian matrix.
- **"liblinear"**:  Coordinate Axis Descent method
- **"sag"**:  the random average gradient descent, a variant of the gradient descent method. The difference from the ordinary gradient descent method is that only a part of the samples are used to calculate the gradient in each iteration, which is suitable when there are many sample data. (like Mini-Batch Gradient Descent)
- **"saga"**: The “sag” solver uses Stochastic Average Gradient descent 6. It is faster than other solvers for large datasets, when both the number of samples and the number of features are large. The “saga” solver 7 is a variant of “sag” that also supports the non-smooth penalty="l1". This is therefore the solver of choice for sparse multinomial logistic regression. It is also the only solver that supports penalty="elasticnet".


"sag" uses only a part of the samples, so don't choose it when the sample size is small. And if the sample size is very large, such as greater than 100,000, "sag" is the first choice. 

But sag cannot be used for L1 regularization, so when you have a large number of samples and need L1 regularization, you have to make a trade-off. Either reduce the sample size by sampling the samples, or return to L2 regularization.

>Note that ‘sag’ and ‘saga’ fast convergence is only guaranteed on features with approximately the same scale. You can preprocess the data with a scaler from sklearn.preprocessing.

>Algorithm to use in the optimization problem.
> - For small datasets, ‘liblinear’ is a good choice, whereas ‘sag’ and ‘saga’ are faster for large ones.
> - For multiclass problems, only ‘newton-cg’, ‘sag’, ‘saga’ and ‘lbfgs’ handle multinomial loss; ‘liblinear’ is limited to one-versus-rest schemes.
> - ‘newton-cg’, ‘lbfgs’, ‘sag’ and ‘saga’ handle L2 or no penalty
> - ‘liblinear’ and ‘saga’ also handle L1 penalty
> - ‘saga’ also supports ‘elasticnet’ penalty
> - ‘liblinear’ does not support setting penalty='none'

**References**

L-BFGS-B – Software for Large-scale Bound-constrained Optimization
Ciyou Zhu, Richard Byrd, Jorge Nocedal and Jose Luis Morales. 
<http://users.iems.northwestern.edu/~nocedal/lbfgsb.html>

LIBLINEAR – A Library for Large Linear Classification
<https://www.csie.ntu.edu.tw/~cjlin/liblinear/>

SAG – Mark Schmidt, Nicolas Le Roux, and Francis Bach
Minimizing Finite Sums with the Stochastic Average Gradient 
<https://hal.inria.fr/hal-00860051/document>

SAGA – Defazio, A., Bach F. & Lacoste-Julien S. (2014).
SAGA: A Fast Incremental Gradient Method With Support for Non-Strongly Convex Composite Objectives 
<https://arxiv.org/abs/1407.0202>

Hsiang-Fu Yu, Fang-Lan Huang, Chih-Jen Lin (2011). Dual coordinate descent
methods for logistic regression and maximum entropy models. Machine Learning 85(1-2):41-75. 
<https://www.csie.ntu.edu.tw/~cjlin/papers/maxent_dual.pdf>



## 4. Classification Type Parameter: Multi_class
There are three values can be choose of the parameter multi_class:
- **"ovr"**: one-vs-rest(OvR)
- **"multinomial"**: many-vs-many(MvM)
- **"auto"**

> If the option chosen is ‘ovr’, then a binary problem is fit for each label. For ‘multinomial’ the loss minimised is the multinomial loss fit across the entire probability distribution, even when the data is binary. ‘multinomial’ is unavailable when solver=’liblinear’. ‘auto’ selects ‘ovr’ if the data is binary, or if solver=’liblinear’, and otherwise selects ‘multinomial’.
The idea of OvR is very simple. No matter how many catogeries in logistic regression classification model, we can treat them as many binary logistic regression. For example, if there are K categories, we view each one of K categories as postive and the rest are negative. Finally we get K-1 classifiers. (like cross validation)

The idea of MvM is relatively complicated. For example, if there are T categories, we pick two categories's data from the samples each time, and noted them as T1 and T2 classes (there are $C_T^2$ ways). Use T1 as a positive class, and T2 as a negative class to perform binary logistic regression to obtain model parameters. Finally, We need $T(T-1) / 2 $  classifications in total.

As we can see, OvR is relatively simple, but the classification performance is relatively poor.(here refers to the distribution of most samples, OvR may be better under certain sample distributions). The MvM classification is relatively accurate, but the classification speed is not as fast as OvR.

- If choose "ovr", all 5 solvers can be chose: liblinear, newton-cg, lbfgs, sag, saga.
- If choose "multinomial", except "liblinear", we can use all other 4 solvers.

## 5. Type Weight Parameter: class_weight
class_weight: dict or ‘balanced’, default=None

The class_weight parameter is used to indicate the various categories' weights in the classification model. If None, all categories' weights are the same. Or we can input the weights of each category manually. For a binary model of 0 and 1 as an example, we can define the class_weight = {0: 0.9, 1: 0.1}, so that the weight of type 0 is 90%, and the weight of type 1 is 10%.

If **class_weight="balanced"**, then the library will calculate the weight based on the training sample size. The larger the sample size of a certain type, the lower the weight, and the smaller the sample size, the higher the weight.

So what does class_weight do? **In the classification model, we often encounter two types of problems**:

1. cost of misclassification. For example, it is very expensive to classify legal users and illegal users and classify illegal users as legal users. We prefer to classify legal users as illegal users. At this time, we can manually re-screen, but we do not want to classify illegal users as legal users . At this time, we can appropriately increase the weight of illegal users.

2. the sample is highly unbalanced. For example, we have 10,000 binary sample data of legal users and illegal users. There are 9995 legal users and only 5 illegal users. If we do n’t consider the weight, we can divide all Of the test sets are predicted as legitimate users, so the prediction accuracy rate is 99.95% in theory, but it does not make any sense. At this time, we can choose balanced to let the class library automatically increase the weight of illegal user samples.


And for the second type of sample imbalance, we can also consider using the sample weight parameter mentioned in the next section: sample_weight instead of class_weight.

## 6. sample_weight
Due to the sample imbalance, the sample is not an unbiased estimate of the overall sample, which may lead to a decline in our model's predictive ability. In this case, we can try to solve this problem by adjusting the sample weight. There are two ways to adjust the sample weight. **The first is to use balanced in class_weight. The second is to adjust the weight of each sample by sample_weight when calling the fit function.**

When doing logistic regression in scikit-learn, if the above two methods are used, then the real weight of the sample is **class_weight * sample_weight**.