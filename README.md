## The Iris Dataset
### In this notebook, many Machine Learning Algorithms were implemented on the Iris Dataset.
### The Iris Dataset is one of the "Hello World" problems in Machine Learning.

The Machine Learning **(ML)** Algorithms used are:
* __[Supervised Learning:](https://www.ibm.com/cloud/blog/supervised-vs-unsupervised-learning#:~:text=for%20your%20situation.-,What%20is%20supervised%20learning%3F,-Supervised%20learning%20is)__
    * __[Logistic Regression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)__
    * __[Decision Tree](https://scikit-learn.org/stable/modules/tree.html)__
    * __[Support Vector Machine](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html)__
    * __[XGBoost](https://xgboost.readthedocs.io/en/stable/install.html)__
<br><br>
* __[Unsupervised Learning:](https://www.ibm.com/cloud/blog/supervised-vs-unsupervised-learning#:~:text=and%20polynomial%20regression.-,What%20is%20unsupervised%20learning%3F,-Unsupervised%20learning%20uses)__
    * __[K-means](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html)__<br>
<br>

### Below are some basic explanations, from the *[Machine Learning Specialization - DeepLearning.AI](https://www.deeplearning.ai/courses/machine-learning-specialization/)* courses.
#### Supervised Learning 
It is the use of labeled datasets to train ML algorithms.
During training, the input $x$ is mapped to the output $y$. If trained well, the algorithm will be able to predict unseen data.
Many factors tell if the ML model is trained well, such as the Gradient Descent where the input trainng data is utilized to fit the parameters w, b (or the weights) by minimizing the measure of the error between the model's predictions $f_{w,b}(x^{(i)})$, and the actual data $y^{(i)}$.
The measure is called the cost $J(w,b)$ where:
$$ J(\mathbf{w},b) = \frac{1}{m} \sum_{i=0}^{m-1} \left[ loss(f_{\mathbf{w},b}(\mathbf{x}^{(i)}), y^{(i)}) \right] \tag{1}$$
In this notebook, we are using Classification. Therefore, the classification loss function loss(f_{\mathbf{w},b}(\mathbf{x}^{(i)}), y^{(i)}) \right is discribed as follow:
$$loss(f_{\mathbf{w},b}(\mathbf{x}^{(i)}), y^{(i)}) = -y^{(i)} \log\left(f_{\mathbf{w},b}\left( \mathbf{x}^{(i)} \right) \right) - \left( 1 - y^{(i)}\right) \log \left( 1 - f_{\mathbf{w},b}\left( \mathbf{x}^{(i)} \right) \right) \tag{2}$$
And the Gradient Descent is a loop where we substract the partial derivative from the weights simultaneously:
$$\begin{align*} \text{repeat}&\text{ until convergence:} \; \lbrace \newline
\;  w &= w -  \alpha \frac{\partial J(w,b)}{\partial w} \tag{3}  \; \newline 
 b &= b -  \alpha \frac{\partial J(w,b)}{\partial b}  \newline \rbrace
\end{align*}$$
where, parameters $w$, $b$ are updated simultaneously *(calculating the partial derivatives for all the parameters before updating any of the parameters)*.  
The gradient is defined as:
$$\begin{align}
\frac{\partial J(w,b)}{\partial w}  &= \frac{1}{m} \sum\limits_{i = 0}^{m-1} (f_{w,b}(x^{(i)}) - y^{(i)})x^{(i)} \tag{4}  \; \newline 
\frac{\partial J(w,b)}{\partial b}  &= \frac{1}{m} \sum\limits_{i = 0}^{m-1} (f_{w,b}(x^{(i)}) - y^{(i)})  \newline 
\end{align}$$
Where $m$ is the number of training examples.
<br><br>
In this notebook, the Supervised Learning Algorithms are implemented with the __[$sklearn$](https://scikit-learn.org/stable/)__ library.
### Unsupervised Learning
It is the use of ML algorithms to identify patterns in data sets containing data points thar are neither classified nor labaled.
<br>In this notebook the $K$-means is used as an Unsupervised Learning Algorithm.
<br>The $K$-means algorithm is a method to automatically cluster similar
data points together. 

* It is the practive of grouping a training set $\{x^{(1)}, ..., x^{(m)}\}$,  into a few cohesive “clusters”. 
<br><br>
* $K$-means is an iterative procedure that
    * Starts by guessing the initial centroids, and then 
    * Refines this guess by 
        * Repeatedly assigning examples to their closest centroids, and then 
        * Recomputing the centroids based on the assignments.       
<br><br>
* The inner-loop of the algorithm repeatedly carries out two steps: 
    * (i) Assigning each training example $x^{(i)}$ to its closest centroid, and
    * (ii) Recomputing the mean of each centroid using the points assigned to it.    
<br><br>
* The $K$-means algorithm will always converge to some final set of means for the centroids. 
<br><br>
* However, that the converged solution may not always be ideal and depends on the initial setting of the centroids.
    * Therefore, in practice the $K$-means algorithm is usually run a few times with different random initializations. 
    * One way to choose between these different solutions from different random initializations is to choose the one with the lowest cost function value (distortion).
<br><br>
* In this notebook the $K$-means is implemented in two ways:
    * Implementing the algorithms with $numpy$
    * Using the $sklearn$ Library