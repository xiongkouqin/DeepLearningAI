# Supervised Learning: Regression and Classification

## 1. Machine Learning Overview

## 1.1 Supervised VS Unsupervised

- Supervised: algorthms learn the `mapping` from input and output, learn from being given right `answers`
	- First, train model using x and y
	- After that, the model can predict y by given unknown x
	- Types:
		- Regression: predict a number infinitely many possible outputs
		- Classification: predict categories small number of possible outputs
- Unsupervised: training data includes no output label y, we are going to find structure, pattern or something interesting
	- Clustering
		- For exmaple, Google News group news with same topic together
	- Anomaly detection
		- find unusual events/ data points
	- Dimensionally reduction
		- compress data

## 2. Linear Regression

### 2.1 Example: House size and price



### 2.2 Terminology

```markdown
x = input variable / feature
y = output variable / target
m = number of training examples
(x , y) = single training example
(x^i , y^i) = ith single training example
```



### 2.3 Univariate Linear Regression

一元线性回归，也就是输入只有一个feature



### 2.4 Cost Function

> Why cost function is needed, to find the best fit line, we need to know how to measure the error at least. So we have cost function. 
>
> why 1/2m, just to make the cost not larger because m is larger

![image-20231130224359034](assets/image-20231130224359034.png)



## 3. Gradient Descent

a systemmatic way to find best parameters that minimize J (cost)

> Outline:
>
> start with some w, b
>
> keep changing w, b to reduce J(w, b)
>
> Until we settle at or near a minimum
>
> `直观理解`
>
> 你站在一座山上，你想要最快下山，你环顾四周，找最陡（下降最快）的地方走一步
>
> 但是其实是有可能走到local minima的！（看你的起点位置）

### 3.1 Implement Gradient Descent

![image-20231202161203166](assets/image-20231202161203166.png)

### 3.2 Learning Rate

1) Too small?
	- Take very long to find minima
2) Too larg?
	- overshoot or not converge



![image-20231202162456056](assets/image-20231202162456056.png)

> ⚠️ 值得关注的点：
>
> 1. Can reacg local minimum with fixed learning rate
>
> 	Because:
>
> 	- Near a local minimum, derivative becomes smaller, update steps become smaller
> 	- Can reach minimum without decreasing learning rate 
>
> 2. Also, we know that if we're at the local minimum, derivative is 0, w will not be updated

### 3.3 "Batch" gradient descent

"Batch": each step of gradient descent uses all the training examples



## 4. Mutiple Features

### 4.1 Terminologies

$$
x^j = j^{th} \;feature \\
n = number \;of \;features \\
\vec{x} ^ {(i)} = features \;of \;i^{th} \;training\; example \\
\vec{x_j} ^ {(i)} = value \;of \;features \;j \;of \;i^{th} \;training \;example \\
f_{w,b}(x) = wx + b \\
f_{\vec{w},b}(\vec{x}) = \vec{w}\cdot\vec{x} + b \\
$$

Note: in f, w and x can be both row vector, it is dot product, we can think of this just to make we can write the formula in a more simple and clean way



### 4.2 Vectorization

- make your formula and code more easier
- use GPU, run in parallel

#### 4.2.1 Parameters and features

![image-20231203221027193](assets/image-20231203221027193.png)

![image-20231203221441963](assets/image-20231203221441963.png)

![image-20231203221254019](assets/image-20231203221254019.png)

#### 4.2.2 Gradient Descent for Multiple Features

![image-20231203222155236](assets/image-20231203222155236.png)

![image-20231203222201465](assets/image-20231203222201465.png)

![image-20231203222419840](assets/image-20231203222419840.png)



## 5. Practical Tips for Linear Regression

### 5.1 Feature Scaling

#### 5.1.1 Why we need it 

比如房价预测的例子，size value range 很大，w1一般取很大的值，number of bedrooms value range很小，w2一般要很大的值，这会带来什么问题吗？看一看cost function的contour plot: gradient descent 会bounce around比较难收敛。

![image-20231203223217115](assets/image-20231203223217115.png)

#### 5.1.2 Why we need it 

有好几种：

1. divde by maximum value

	![image-20231203223550094](assets/image-20231203223550094.png)

2. mean normalization

	![image-20231203223556749](assets/image-20231203223556749.png)

3. z score

	![image-20231203223622589](assets/image-20231203223622589.png)

### 5.2 Check Gradient Descent for Convergence

![image-20231203224332667](assets/image-20231203224332667.png)

> 如果J在某一次迭代变大：比较可能的是代码写错或者学利率选择太大了。
>
> 选择epsilon很难，最好还是看左边的图。

### 5.3 Chooisng Learning Rate

![image-20231203224946497](assets/image-20231203224946497.png)

![image-20231203225013157](assets/image-20231203225013157.png)

### 5.4 Feature Engineering

> Using intuition to design new features, by transforming or combining original features

自己去组合然后选择feature

比如原本给的是x1 = frontage, x2 = depth, 我们可能直接用一个feature x3 = area = frontage * depth

### 5.5 Polynomial Regression

> use both linear regression and feature engineering

![image-20231203230026843](assets/image-20231203230026843.png)

> 到底要用什么feature？ 后面会讲的。

## 6. Classification

### 6.1 Motivation

虽然可以仍然用regression + threshold来做分类，但效果一般都不好。

### 6.2 Logistic Regression

![image-20231204001037749](assets/image-20231204001037749.png)

![image-20231204001025132](assets/image-20231204001025132.png)

![image-20231204001031345](assets/image-20231204001031345.png)

#### 6.2.1 Decision Boundary

> 简单来说，设置threshold,然后可以解出来什么条件下(X取什么值)，达到这个threshold

![image-20231204001723360](assets/image-20231204001723360.png)

![image-20231204001728817](assets/image-20231204001728817.png)

#### 6.2.2 Cost Function

> 如果继续用squared error,画一下图像可以知道函数不再是convex，非常大可能性你会到达一个local minima，所以找一个别的cost function让我们得到一个convex，比较容易收敛到global minima
>
> cost function:是对整个training set的sum
>
> Loss function:是一个sample的误差

![image-20231204003740113](assets/image-20231204003740113.png)

![image-20231204003756592](assets/image-20231204003756592.png)

![image-20231204004037670](assets/image-20231204004037670.png)

![image-20231204004120701](assets/image-20231204004120701.png)

#### 6.2.3 Simplified Cost Function

![image-20231204004533012](assets/image-20231204004533012.png)

![image-20231204004601563](assets/image-20231204004601563.png)

#### 6.2.4 Gradient Descent Implementation

![image-20231204005151235](assets/image-20231204005151235.png)

![image-20231204005208553](assets/image-20231204005208553.png)

## 7. Regularization to Reduce Overfitting

### 7.1 The problem of overfitting

![image-20231204005603703](assets/image-20231204005603703.png)

![image-20231204005610018](assets/image-20231204005610018.png)

### 7.2 Addressing Overfitting

1. Collect more training data

	![image-20231204010731760](assets/image-20231204010731760.png)

2. Select features to include/exclude (后面的课会讲，有些算法可以自动选)

	![image-20231204010704577](assets/image-20231204010704577.png)

3. Regularization - reduce the size of parameters (不一定是0，可能是一个很小的数字)

	usually focus on w, but not b

	![image-20231204010803858](assets/image-20231204010803858.png)

### 7.3 Regularization

> 总的来说，有一些high order的term，他们是有用的，但是可能没有那么重要，去除他们error可能会比较大，但include进来如果不做正则化，会导致overfit。
>
> 因此，我们penalize，w，也就是w不要选太大的值，太大就会overfit。
>
> 但是我们又不确定哪些要penalize，所以直接对全部都进行penalize。

![image-20231204012338179](assets/image-20231204012338179.png)

![image-20231204012344143](assets/image-20231204012344143.png)

![image-20231204012349612](assets/image-20231204012349612.png)

### 7.4 Regularized Linear Regression

![image-20231204013112167](assets/image-20231204013112167.png)

![image-20231204013120731](assets/image-20231204013120731.png)

![image-20231204013127874](assets/image-20231204013127874.png)

![image-20231204013136948](assets/image-20231204013136948.png)

### 7.5 Regularized Logistic Regression

![image-20231204013321866](assets/image-20231204013321866.png)

![image-20231204013328253](assets/image-20231204013328253.png)



## 8. Nearal Network

### 8.1 Intuition

#### 8.1.1 Brain and Neuron

> acutually we still do not know how brain works, so it's a 不是那么严谨的比喻

![image-20231205002315507](assets/image-20231205002315507.png)

![image-20231205002748128](assets/image-20231205002748128.png)

#### 8.1.2 Demand Prediction

比如你要判断a T-shirt is top seller or not, 你用logistic regression构建的可以看做一个简单的neuron，neuron network就是你用更多neuron。

> Another way to interpret neural network:
>
> Look at the output layer, it's just a logistic regression, but it does not use original features, so we can think of the last layer is automatically derive better feature for us to train the logistic regression (Feature Engineering)

![image-20231205003401836](assets/image-20231205003401836.png)

#### 8.1.3 Example: Face Recognition

![image-20231205004841568](assets/image-20231205004841568.png)

![image-20231205004846604](assets/image-20231205004846604.png)

> Here, but not exactly, just MIGHT
>
> first layer looks for small lines
>
> second layer looks for small part of face
>
> Third layer looks for larger part of face

### 8.2 Neural Network Model

#### 8.2.1 Layer

![image-20231205005920197](assets/image-20231205005920197.png)

![image-20231205010246750](assets/image-20231205010246750.png)

![image-20231205010306074](assets/image-20231205010306074.png)

#### 8.2.2 More Complex Neural Network

![image-20231205011030074](assets/image-20231205011030074.png)

#### 8.2.3 Inference: Make Predictions

> Forward propagation for prediction!
>
> Back propagation for learning!

![image-20231205011755322](assets/image-20231205011755322.png)

![image-20231205011759842](assets/image-20231205011759842.png)

![image-20231205011803469](assets/image-20231205011803469.png)

### 8.3 Build using Tensorflow

#### 8.3.1 Tensorflow Implementation

![image-20231205185741204](assets/image-20231205185741204.png)

#### 8.3.2 Data in Tensorflow

> Due to historical reason, there are some conflicts between numpy and tensor flow:
>
> - Tensor is the "matrix" in tensorflow, they can be converted into/from np array
> - [1, 2] is just a 1d array, not matrix, no row, no column
>
> Btw, Dense means 全连接

![image-20231205191142769](assets/image-20231205191142769.png)

![image-20231205191155564](assets/image-20231205191155564.png)

#### 8.3.3 True Implementation

> 简单来说，不需要自己去写layer_1是什么，a1怎么计算，而是直接declarative地写出来model是什么样的，就可以了。

![image-20231205191742462](assets/image-20231205191742462.png)

![image-20231205191749061](assets/image-20231205191749061.png)

![image-20231205191755808](assets/image-20231205191755808.png)

### 8.4 Implement from Scratch

#### 8.4.1 Straight forward way

![image-20231205193443076](assets/image-20231205193443076.png)

#### 8.4.2 General way

![image-20231205193418507](assets/image-20231205193418507.png)

### 8.5 Vectorization

![image-20231205205655705](assets/image-20231205205655705.png)
