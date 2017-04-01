#*Writeup CarND-Traffic-Sign-Recognition** 

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)

* Explore, summarize and visualize the data set

* Design, train and test a model architecture

* Use the model to make predictions on new images

* Analyze the softmax probabilities of the new images

* Summarize the results with a written report

  ​


## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

Please find the link to my code: [project code](https://github.com/anujpotnis/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

###Data Set Summary & Exploration

####1. Provide a basic summary of the data set and identify where in your code the summary was done. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

Numpy was used to calculate the summary statistics of the traffic signs data set.

```python
# Number of training examples
n_train = y_train.shape[0] # 34799

# Number of testing examples.
n_test = y_test.shape[0] # 12630

# What's the shape of an traffic sign image?
image_shape = X_train.shape[1:3] # 32 by 32

# How many unique classes/labels there are in the dataset.
n_classes = np.unique(y_train).shape[0] # 43
```



####2. Include an exploratory visualization of the dataset and identify where the code is in your code file.

```python
# Data exploration visualization code
np.random.seed(0)
fig, axes = plt.subplots(4, 4, figsize=(12, 6),
                         subplot_kw={'xticks': [], 'yticks': []})
fig.subplots_adjust(hspace=0.3, wspace=0.05)
axes = axes.ravel()

for idx in range(16):
    sign = np.random.randint(0, n_train)
    axes[idx].imshow(X_train[sign])
    axes[idx].set_title(y_train[sign])
```

<p align="center">
  <img src="writeup_data/sign_visualize.png" alt="Signboard Visualization"/>
</p>

```python
# Histogram (using Matplotlib)
plt.hist(y_train, bins=n_classes)
plt.ylabel("Counts per Traffic Sign Labels")
plt.xlabel("Traffic Sign Label Number")
plt.title("Traffic Sign")
plt.show()
```

<p align="center">
  <img src="writeup_data/histogram.png" alt="Histogram", height="400"/>
</p>

###Design and Test a Model Architecture

####1. Describe how, and identify where in your code, you preprocessed the image data. What tecniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc.

The data was normalized between -1 and +1. This significantly improved the accuracy. The weights in the neural network are typically between -1 and +1. Therefore normalizing the input data to that range allowed the network to converge with better accuracy.

An attempt was made to subtract the mean from every image. The idea was to center the data around 0. However this reduced the accuracy and hence was not used in the submitted code.

Grayscale was not used since color information plays a vital role in the recognition of signboards.

```python
# Preprocess the data

def normalize(img):
    img = img/127.5-1.
    print(np.min(img))
    return img

# def mean_substract(img):
#     img = img - np.mean(img)
#     return img
```

```python
# X_train = mean_substract(X_train)
# X_valid = mean_substract(X_valid)
# X_test = mean_substract(X_test)

X_train = normalize(X_train)
X_valid = normalize(X_valid)
X_test = normalize(X_test)
```
### Normalized Signboard Images

<p align="center">
  <img src="writeup_data/sign_visualize_norm.png" alt="Normalized", height="400"/>
</p>

### Mean Subtracted Signboard Images

<p align="center">
  <img src="writeup_data/sign_visualize_mean_sub.png" alt="Mean Subtract", height="400"/>
</p>

####2. Describe how, and identify where in your code, you set up training, validation and testing data. How much data was in each set? Explain what techniques were used to split the data into these sets. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, identify where in your code, and provide example images of the additional data)

The code for splitting the data into training and validation sets is contained in the fifth code cell of the IPython notebook.  

To cross validate my model, I randomly split the training data into a training set and validation set. I did this by ...

My final training set had X number of images. My validation set and test set had Y and Z number of images.

The sixth code cell of the IPython notebook contains the code for augmenting the data set. I decided to generate additional data because ... To add more data to the the data set, I used the following techniques because ... 

Here is an example of an original image and an augmented image:

![alt text][image3]

The difference between the original data set and the augmented data set is the following ... 


####3. Describe, and identify where in your code, what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

```python
def TrafficSignClassifier_LeNet(x):    
    # Hyperparameters
    mu = 0
    sigma = 0.1

    # Layer 1: Convolutional. Input = 32x32x3. Output = 28x28x6.
    conv1_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 3, 6), mean = mu, stddev = sigma))
    conv1_b = tf.Variable(tf.zeros(6))
    conv1   = tf.nn.conv2d(x, conv1_W, strides=[1, 1, 1, 1], padding='VALID') + conv1_b

    # Activation.
    conv1 = tf.nn.relu(conv1)

    # Pooling. Input = 28x28x6. Output = 14x14x6.
    conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
.
.
.
  # Layer 5: Fully Connected. Input = 84. Output = 43.
      fc3_W  = tf.Variable(tf.truncated_normal(shape=(84, n_classes), mean = mu, stddev = sigma))
      fc3_b  = tf.Variable(tf.zeros(43))
      logits = tf.matmul(fc2, fc3_W) + fc3_b

      return logits
```



The architecture summary is as follows:

|      Layer      |               Description                |
| :-------------: | :--------------------------------------: |
|      Input      |            32x32x3 RGB image             |
| Convolution 5x5 | 1x1 stride, valid padding, outputs 28x28x6 |
|      RELU       |                                          |
|   Max pooling   |       2x2 stride,  outputs 14x14x6       |
| Convolution 5x5 | 1x1 stride, valid padding, outputs 10x10x16 |
|      RELU       |                                          |
|   Max Pooling   |       2x2 stride,  outputs 5x5x16        |
|     Flatten     |                output 400                |
| Fully Connected |                output 120                |
|      RELU       |                                          |
| Fully Connected |                output 84                 |
|      RELU       |                                          |
| Fully Connected |                output 43                 |



####4. Describe how, and identify where in your code, you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

```python
# Configure training parameters

rate = 0.001
EPOCHS = 10
BATCH_SIZE = 128

logits = TrafficSignClassifier_LeNet(x)
probability = tf.nn.softmax(logits)
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits, one_hot_y)
loss_operation = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate = rate)
training_operation = optimizer.minimize(loss_operation)
```

The model was trained at an initial low rate of 0.001. This ensured a slow but stable learning. Also, the Adam Optimizer controls the rate.10 epochs were sufficient as the accuracy did not improved after that. A batch size of 128 was optimal considering the memory size of the GPU used.

An Adam optimizer was used since it adaptively computes the learning rate. Adam works well in practive.



####5. Describe the approach taken for finding a solution. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

The code for calculating the accuracy of the model is located in the ninth cell of the Ipython notebook.

My final model results were:
* training set accuracy of ?
* validation set accuracy of ? 
* test set accuracy of ?

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
* What were some problems with the initial architecture?
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to over fitting or under fitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
* Which parameters were tuned? How were they adjusted and why?
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

If a well known architecture was chosen:
* What architecture was chosen?
* Why did you believe it would be relevant to the traffic sign application?
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?


###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

The first image might be difficult to classify because ...

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. Identify where in your code predictions were made. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

The code for making predictions on my final model is located in the tenth cell of the Ipython notebook.

Here are the results of the prediction:

|     Image     |  Prediction   |
| :-----------: | :-----------: |
|   Stop Sign   |   Stop sign   |
|    U-turn     |    U-turn     |
|     Yield     |     Yield     |
|   100 km/h    |  Bumpy Road   |
| Slippery Road | Slippery Road |


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of ...

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction and identify where in your code softmax probabilities were outputted. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were

| Probability |  Prediction   |
| :---------: | :-----------: |
|     .60     |   Stop sign   |
|     .20     |    U-turn     |
|     .05     |     Yield     |
|     .04     |  Bumpy Road   |
|     .01     | Slippery Road |


For the second image ... 