# Machine learning



## 1. Definition

- Field of study that gives computers the ability to learn without being explicity programmed



### Machine learning algorithms

- Supervised learning

  Used mostly in real-world application

  

- Unsupervised learning

- Recommender systems

- Reinforcement learning

  

## 2. Supervised learning



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



## 2.Unsupervised learning



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



### Logistic regression (Curve)

![image info](./image/Screenshot%202023-05-01%20at%201.50.08%20AM.png)



![image info](./image/Screenshot%202023-05-01%20at%201.51.48%20AM.png)



#### What logistic regression output mean?



![image info](./image/Screenshot%202023-05-01%20at%201.55.29%20AM.png)



#### How to do that in Python

```python
import numpy as np
%matplotlib widget
import matplotlib.pyplot as plt
from plt_one_addpt_onclick import plt_one_addpt_onclick
from lab_utils_common import draw_vthresh
plt.style.use('./deeplearning.mplstyle')

# Input is an array. 
input_array = np.array([1,2,3])
exp_array = np.exp(input_array)

# NumPy has a function called exp(), which offers a convenient way to calculate the exponential (𝑒^𝑧) of all elements in the input array (z).
print("Input to exp:", input_array)
print("Output of exp:", exp_array)

# Input is a single number
input_val = 1  
exp_val = np.exp(input_val)

print("Input to exp:", input_val)
print("Output of exp:", exp_val)

def sigmoid(z):
    """
    Compute the sigmoid of z

    Args:
        z (ndarray): A scalar, numpy array of any size.

    Returns:
        g (ndarray): sigmoid(z), with the same shape as z
         
    """

    g = 1/(1+np.exp(-z))
   
    return g
  
  
# Generate an array of evenly spaced values between -10 and 10
z_tmp = np.arange(-10,11)

# Use the function implemented above to get the sigmoid values
y = sigmoid(z_tmp)

# Code for pretty printing the two arrays next to each other
np.set_printoptions(precision=3) 
print("Input (z), Output (sigmoid(z))")
print(np.c_[z_tmp, y])

# Plot z vs sigmoid(z)
fig,ax = plt.subplots(1,1,figsize=(5,3))
ax.plot(z_tmp, y, c="b")

ax.set_title("Sigmoid function")
ax.set_ylabel('sigmoid(z)')
ax.set_xlabel('z')
draw_vthresh(ax,0)

# Logistic Regression
x_train = np.array([0., 1, 2, 3, 4, 5])
y_train = np.array([0,  0, 0, 1, 1, 1])

w_in = np.zeros((1))
b_in = 0

plt.close('all') 
addpt = plt_one_addpt_onclick( x_train,y_train, w_in, b_in, logistic=True)
```



#### Decision Boundary



![image info](./image/Screenshot%202023-05-11%20at%2011.53.01%20PM.png)



**Set threshold(临界) above which you predict y is one**

![image info](./image/Screenshot%202023-05-05%20at%203.13.08%20PM.png)



- **Non-linear decision boundaries**

  ![image info](./image/Screenshot%202023-05-05%20at%206.03.41%20PM.png)

  
  
  ![image info](./image/Screenshot%202023-05-05%20at%206.07.38%20PM.png)

### How to make Decsion Boundary in Python

```python
import numpy as np
%matplotlib widget
import matplotlib.pyplot as plt
from lab_utils_common import plot_data, sigmoid, draw_vthresh
plt.style.use('./deeplearning.mplstyle')

# Data set
X = np.array([[0.5, 1.5], [1,1], [1.5, 0.5], [3, 0.5], [2, 2], [1, 2.5]])
y = np.array([0, 0, 0, 1, 1, 1]).reshape(-1,1) 

# Plot Data
fig,ax = plt.subplots(1,1,figsize=(4,4))
plot_data(X, y, ax)

ax.axis([0, 4, 0, 3.5])
ax.set_ylabel('$x_1$')
ax.set_xlabel('$x_0$')
plt.show()

# Plot sigmoid(z) over a range of values from -10 to 10
z = np.arange(-10,11)

fig,ax = plt.subplots(1,1,figsize=(5,3))
# Plot z vs sigmoid(z)
ax.plot(z, sigmoid(z), c="b")

ax.set_title("Sigmoid function")
ax.set_ylabel('sigmoid(z)')
ax.set_xlabel('z')
draw_vthresh(ax,0)

# Choose values between 0 and 6
x0 = np.arange(0,6)

x1 = 3 - x0
fig,ax = plt.subplots(1,1,figsize=(5,4))

# Plot the decision boundary
ax.plot(x0,x1, c="b")
ax.axis([0, 4, 0, 3.5])

# Fill the region below the line
ax.fill_between(x0,x1, alpha=0.2)

# Plot the original data
plot_data(X,y,ax)
ax.set_ylabel(r'$x_1$')
ax.set_xlabel(r'$x_0$')
plt.show()
```



![image info](./image/)



# Recommendation System



### Using per-item features

#### Example: Predicting moview ratings

- Some number of users and items

![image info](./image/image-20230430214739922.png)



- **Add feature for each movie**

  Then have linear regression modle of vectors

  ![image info](./image/Screenshot%202023-04-30%20at%209.54.11%20PM.png)

  

  - **Cost function for each user j to learn w, b**

  ![image info](./image/Screenshot%202023-04-30%20at%2010.58.52%20PM.png)

  ![image info](./image/Screenshot%202023-04-30%20at%2011.01.04%20PM.png)

  

- **PS: 后半段橙色的是 Regularization term**



#### What if there is no feature for items (Collaborative  filtering)

- **w(j) * x(i)** should approximately equal to actual rate

- **Can take reasonable guess** at waht lists a feature

  ![image info](./image/Screenshot%202023-04-30%20at%2011.35.25%20PM.png)

  

  - **Cost function for item i to learn x(i)** 

    ![image info](./image/Screenshot%202023-04-30%20at%2011.40.47%20PM.png)

  

  - **Collaborative filtering algroithm**

    - Because multiple users have rated the same movie collaboratively, give you a sense of what this movie maybe like, allow you to guess what are approprite features for that movie, then allow you to predict how other users that havn't rated that movie ay decide to rate it in the future
    - The underlying idea behind collaborative filtering is that people with similar tastes in the past are likely to have similar tastes in the future.
    - It does not require information about the items themselves or their characteristics, and it can handle new items that have not been rated before. 

    ![image info](./image/Screenshot%202023-04-30%20at%2011.47.53%20PM.png)

    ![image info](./image/Screenshot%202023-04-30%20at%2011.54.40%20PM.png)



#### Collaborative filtering with binary labels

- **1** mean like **0** mean don't like

- Many way to find 1 or 0

  ![image info](./image/Screenshot%202023-05-01%20at%2012.54.42%20AM.png)

  ![image info](./image/Screenshot%202023-05-01%20at%2012.54.42%20AM.png)

  

  ![image info](./image/Screenshot%202023-05-01%20at%2012.57.31%20AM.png)

  

- **Using logistic regression**

  ![image info](./image/Screenshot%202023-05-01%20at%201.04.11%20AM.png)

  

  - **Binary cross entropy cost function**

    ![image info](./image/Screenshot%202023-05-01%20at%201.09.01%20AM.png)



![image info](./image/)
