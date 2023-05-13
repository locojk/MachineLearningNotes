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





![image info](./image/...)
