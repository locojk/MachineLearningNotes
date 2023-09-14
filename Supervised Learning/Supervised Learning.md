# Machine learning



## Definition

- Field of study that gives computers the ability to learn without being explicity programmed



### Machine learning algorithms

- Supervised learning

  Used mostly in real-world application

  

- Unsupervised learning

- Recommender systems

- Reinforcement learning

  

## Supervised learning



**Input -> output label** mapping

Learns from being given "**Right answers**"



Ex:

![image info](./image/Screenshot%202023-04-16%20at%203.56.45%20PM.png)



**Regression**: particular type of supervised learning

- Predict a number 
- Infinitely many possible outputs

![image info](./image/Screenshot%202023-04-16%20at%204.04.52%20PM.png)



**Classification**: 

- predic categories (can be non-number) with small number of possible outputs

![image info](./image/Screenshot%202023-04-16%20at 4.13.20%20PM.png)



![image info](./image/Screenshot%202023-04-16%20at%204.17.05%20PM.png)



![image info](./image/Screenshot%202023-04-16%20at%204.21.50%20PM.png)

The learning algorithm has to decide **how to fit a boundary line** through this data



## Unsupervised learning



- Find something interesting in **unlabeled data**
- Data only comes with input x, but not output labels y
- Algorithm has to find structure in the data



### Clustering

- Grouping data into clusters
- Algorithm figure out how to group data without supervision



### Anomaly detection

- Find unusual data points



### Dimensionality reduction

- Compress data using fewer numbers

  

# Linear Regression



![image info](./image/KuO0hQDNR4OjtIUAzUeDRA_e53cd42b5a7c4dc28ee33f6a2519d4a1_w1l3ivq_1.png)



![image info](./image/Screenshot%202023-04-27%20at%203.14.57%20PM.png)

**w,b also called parameters, coefficients, weight**



### Cost function

- Takes the prediction **y hat** and compares it to the target y by taking y hat minus y
- Tell how well the model is working, measure the difference between the model prediction(y hat) and the actual true value(y).



![image info](./image/Screenshot%202023-04-28%20at%205.46.43%20PM.png)

- **Division 2** in **1/2m** just to make following calculation neater
- Differenr people may use different cost function for different application



### Cost function intution

#### - Find minimize of cost function

![image info](./image/Screenshot%202023-04-28%20at%205.53.20%20PM.png)



**When w =1:**



![image info](./image/Screenshot%202023-04-28%20at%206.00.05%20PM.png)



**When w = 0.5:**

![image info](./image/Screenshot%202023-04-28%20at%206.03.05%20PM.png)



**When w = 0:**

![image info](./image/Screenshot%202023-04-28%20at%206.04.39%20PM.png)



**The final plot is:** 

![image info](./image/Screenshot%202023-04-28%20at%206.05.58 PM.png)

**When w = 1, J(w) is minimum**



#### **J(w,b)**

![image info](./image/Screenshot%202023-04-28%20at%206.11.34%20PM.png)

![image info](./image/Screenshot%202023-04-28%20at%206.18.17%20PM.png)



### Gradient Descent

- Have **J(w1, w2, .... , wn, b)**, want to **min J(w1, w2, .... , wn, b)**

- Gradient descent is **an algorithm for finding values of parameters w and b** that minimize the cost function J

  ![image info](./image/Screenshot%202023-04-30%20at%203.09.21%20PM.png)

  

- How gradient descent work?

  ![image info](./image/Screenshot%202023-04-30%20at%203.18.00%20PM.png)



#### Gradient descent algorithm

- Alpha is **learning rate**（Control how big step to take)

- Using **derivative**

- **Repeat until convergence**(收敛)

- **Simultaneously**(同时) update **w** and **b**

  ![image info](./image/Screenshot%202023-04-30%20at%203.45.13%20PM.png)



#### Derivative review

- d/dw J(w) is the **slope**

  ![image info](./image/Screenshot%202023-04-30%20at%203.52.58%20PM.png)
  
  ![image info](./image/Screenshot%202023-04-30%20at%203.52.58%20PM.png)



#### Learning rate

- Find learning rate

  ![image info](./image/Screenshot%202023-04-30%20at%203.57.57%20PM.png)

  

  ![image info](./image/Screenshot%202023-04-30%20at%204.00.17%20PM.png)

  

  - Detivative 会先大后小

    ![image info](./image/Screenshot%202023-04-30%20at%204.03.52%20PM.png)

  

  #### Gradient descent for linear regression

  ![image info](./image/Screenshot%202023-04-30%20at%204.06.12%20PM.png)

  

  - **Global minmum** is what we want finally (**local mimimum** not good enough)

    ![image info](./image/Screenshot%202023-04-30%20at%204.11.46%20PM.png)

  

  - **Batch gradient descent**
  
    ![image info](./image/Screenshot%202023-04-30%20at%204.13.16%20PM.png)



# Multiple features in Linear Regression



#### Example:

![image info](./image/Screenshot%202023-05-05%20at%206.41.17%20PM.png)

![image info](./image/Screenshot%202023-05-05%20at%206.47.11%20PM.png)



![image info](./image/Screenshot%202023-05-05%20at%206.50.27%20PM.png)



### Vectorization



- Make code shorter and make it run much more efficiently
- Take advantage of modern **numerical linear algebra libraries (NumPy)**

![image info](./image/Screenshot%202023-05-08%20at%207.36.38%20PM.png)



- For loop run one by one, one step one time
- Get value together and calculate all entries in parallel, much faster



![image info](./image/Screenshot%202023-05-08%20at%207.41.24%20PM.png)



![image info](./image/Screenshot%202023-05-08%20at%207.49.40%20PM.png)



### Gradient descent in multiple linear regression

![image info](./image/Screenshot%202023-05-08%20at%208.10.19%20PM.png)



![image info](./image/Screenshot%202023-05-08%20at%208.15.46%20PM.png)

- **Normal equation** mwthod may be used in machibe learning libraries that implement linear regression
- **But Gradient descent is ts the recommended method!**



## Feature scaling

- Very small changes to w1 can have large impact, so w1 should be small
- It takes a much larger change in w2 in order to change the presictions much, so w2 should be small

![image info](./image/Screenshot%202023-05-08%20at%209.45.41 PM.png)



### Scale features: Performing some transformation of your training data (x1 and x2 is scaled both 0-1)

![image info](./image/Screenshot%202023-05-08%20at%209.48.12%20PM.png)

- Can speed gradient descent significantly



### How to scaling feature?



#### Feature scaling

![image info](./image/Screenshot%202023-05-08%20at%209.53.04%20PM.png)



#### Mean normalization

- Have both negtive and positive number

![image info](./image/Screenshot%202023-05-08%20at%2010.00.06%20PM.png)



#### Z-score normalization

![image info](./image/Screenshot%202023-05-08%20at%2010.02.43%20PM.png)





### No harm to carry feature scaling

![image info](./image/Screenshot%202023-05-08%20at%2010.05.22%20PM.png)



## Checking gradient descent for convergence

![image info](./image/Screenshot%202023-05-08%20at%2010.11.48%20PM.png)



## Choosing the learning rate

![image info](./image/Screenshot%202023-05-08%20at%2010.15.24%20PM.png)

- **One important trade-off is that if learning rate is too small, take a lot of iterations to converge**



### Trying the value for learning rate

![image info](./image/Screenshot%202023-05-08%20at%2010.20.50%20PM.png)



## Feature engineering

- The **choice of feature**s can have a huge impact on your learning algorithm's performance

![image info](./image/Screenshot%202023-05-08%20at%2010.31.31%20PM.png)

## Polynomial regression

- **Curve regression**
- **Feature scaling is very important**

![image info](./image/Screenshot%202023-05-08%20at%2010.34.35%20PM.png)



![image info](./image/Screenshot%202023-05-08%20at%2010.36.08 PM.png)





# Classification



![image info](./image/Screenshot%202023-05-01%20at%201.46.23%20AM.png)



## Logistic regression (Curve)

![image info](./image/Screenshot%202023-05-01%20at%201.50.08%20AM.png)



![image info](./image/Screenshot%202023-05-01%20at%201.51.48%20AM.png)



#### What logistic regression output mean?



![image info](./image/Screenshot%202023-05-01%20at%201.55.29%20AM.png)





## Decision Boundary



![image info](./image/Screenshot%202023-05-11%20at%2011.53.01%20PM.png)



**Set threshold(临界) above which you predict y is one**

![image info](./image/Screenshot%202023-05-05%20at%203.13.08%20PM.png)



- **Non-linear decision boundaries**

  ![image info](./image/Screenshot%202023-05-05%20at%206.03.41%20PM.png)

  
  
  ![image info](./image/Screenshot%202023-05-05%20at%206.07.38%20PM.png)





## Cost function for logistic regression



- Target lable y is only 0 or 1
- How can we chose **w** and b



- Squared error cost function for logistic is non-convex
- Lots of local minmum
- Need another cost function to make it convex to use gradient dscent



### Loss Fucntion

- x-axis is **model fucntion**
- y-axis is **loss**
- Loss apply yo single training example but cost apply to entire training example



#### y is equal to 1

![image info](./image/LF.png)



#### y is equal to 0

![image info](./image/Lf2.png)



#### The cost function:

- Cost function is a function of the entire training set, the average of the loss function on the individual training examples

![image info](./image/cost.png)



### Simplified loss function

![image info](./image/SLF.png)

![image info](./image/SLF2.png)

- This cost fucntion is derived from statistics using a statistical principle
- Maximum likehood estimation



## Gradient Descent Implementation

![image info](./image/LGD.png)

- Their deriviavte is same but the defination of f is diff



## Overfitting

- **Bias**: if the algrothm is underfit the data meaning that it's just not even able to fit the training set well

  

- **Overfit**: if the data over fit the data too "Well", model may generalize new examples poorly, if your tarining set were just evenr a little bit different, the function that the algorithm fits could end up being totally different (High variance)
- Ex: large size lead to low price like the third picture

![image info](./image/OF.png)

![image info](./image/OF2.png)



### Regularization to Reduce Overfitting

- Collect more tarining examples may works
- Use less polynomial features may works (**Feature selection**), but useful features may loss



#### **Regularization**

- More gently reduce the impacts of some of reatures, reduce size of parameters

![image info](./image/REa.png)



![image info](./image/AF.png)



### Cost function with regularization

- If have a lot of features, don't know which are the most improtant fdeatures and which ones to penalize
- Penalize all of the features(penalize all the parameters)

- lambda is regularization parameter

![image info](./image/Rg.png)



- This new cost function trades off two goals:
- Trying to minimize this first term encourages the algorithm to fit the training data well by minimizing the squared differences of the predictions and actual value
- Trying to minimize the second term, the algorithm also tries to keep the parameters wj small which tend to reduce overfitting
- If lambda is very large, to minimize the cost function wj must be very small(close to 0)

![image info](./image/Lam.png)



### Derivative

![image info](./image/rd.png)

![image info](./image/im.png)



- Shrink wj, mutiplying some percent to wj (like 0.998)

![image info](./image/shrink.png)



### Regularized logistic regression

![image info](./image/LR.png)



![image info](./image/)
