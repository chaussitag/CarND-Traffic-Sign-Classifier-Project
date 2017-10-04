#**Traffic Sign Recognition** 

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/training_data_distribution.png "Visualization"
[image2]: ./examples/grayscale.jpg "Grayscaling"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./dataset/test-images/1_1.png "Traffic Sign 1"
[image5]: ./dataset/test-images/8_1.png "Traffic Sign 2"
[image6]: ./dataset/test-images/14_2.png "Traffic Sign 3"
[image7]: ./dataset/test-images/30_2.png"  Traffic Sign 4"
[image8]: ./dataset/test-images/7_1.png "Traffic Sign 5"
[image9]:./examples/resampled_data_distribution.png "resampled"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/chaussitag/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

###Data Set Summary & Exploration

####1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the **numpy** library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is 32x32x3
* The number of unique classes/labels in the data set is 43

####2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the data distributed, the x-axis represents the classes of traffic sign, the y-axis represents counts of each traffics sign in the training set.

![alt text][image1]

###Design and Test a Model Architecture

####1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to convert the images to grayscale because the traffic sign is independent of pixel color.

Here is an example of a traffic sign image before and after grayscaling.

![alt text][image2]

As a last step, I normalized the image data so they has a zero mean, because a normalized dataset has benifitial for training a better classifier.

I decided to generate additional data because it is easy to find out that the orignal training dataset is unevenlly distributed, some type of trafic signs have much more samples than others, resulting the last model pays more attention to them. I use the SMOTE algorithm provides by package named  *imblearn* to over-sampling the minority class, here is the distribution of each class before and after over-sampling:

![alt text][image9]

####2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

I extend the LeNet5 model, and my final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 gray image   							| 
| Convolution 5x5    	| 1x1 stride, VALID padding, outputs 28x28x6 	|
| RELU					|												|
| Convolution 5x5    	| 1x1 stride, VALID padding, outputs 24x24x12 	|
| RELU					|												|
| Max pooling 2x2      	| 2x2 stride,  outputs 12x12x12			|
| Convolution 5x5	    | 1x1 stride, VALID padding, outputs 8x8x16           |
| RELU					|												|
| Max pooling 2x2      	| 2x2 stride,  outputs 4x4x16		|
| Flatten                           | outputs 256 dimentional vector |
| Fully connected		| outputs 120 dimentional vector     |
| RELU					|												|
| dropout	                        |  keep_prob=0.8								|
| Fully connected		| outputs 84 dimentional vector     |
| RELU					|												|
| dropout	                        |  keep_prob=0.8								|
| Fully connected		| outputs 43 dimentional vector, the logits     |
| Softmax				| the softmax probobility of each class|
 


####3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used an AdamOptimizer, the batch size is 512, and i trained the model 120 epochs, i used 0.8 as the keep probablity for the dropout operation. I used 0.01 as the start learning rate, and use an exponential schedule to decay the learning rate every two epochs, here is the code:
<pre><code># set up a variable that's incremented once per batch and controls the learning rate decay.
global_step = tf.Variable(0.0, dtype=tf.float32)
\# Decay the learning rate once per two epochs, using an exponential schedule starting at 0.01.
learning_rate = tf.train.exponential_decay(BASE_LEARNING_RATE,
                                           global_step,
                                           2 * num_train_images,
                                           LEARNING_RATE_DECAY_BASE, staircase=True)
training_op = tf.train.AdamOptimizer(learning_rate=learning_rate)
                .minimize(loss_op, global_step=global_step)
</code></pre>

####4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of  1.0
* validation set accuracy of **0.965533**
* test set accuracy of 0.941013

If an iterative approach was chosen:
####(1)What was the first architecture that was tried and why was it chosen?####
At first i choose LeNet5 as the initial model.
####(2)What were some problems with the initial architecture?####
after several tries of the hyper-parameters, the last validation accuracy of the LeNet5 is about 0.93 to 0.94, so i dicide to make some changes to the model.
####(3)How was the architecture adjusted and why was it adjusted?####
According my experments, the training accuracy is about 4 to 6 percent higher than the validation accuracy, so it seems that there is some overfitting,  i add drop out operations to the last two fully connected layers, and add L2 regularization to all trainable parameters. From the visualization of the training dataset, it's easy to find that the distribution of the training dataset is not even, i try to use the SMOTE algorithm to over-sampling the minority classes. I also add one more convolutional layer to the original LeNet5 model, make it more powerful to classify the traffic signs. Combing all the changes listed, the final model has an validation accuracy above 0.965.


###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

I downloades 14 test images with German traffic signs from the internet, here are five of them[refer to the ipython notebook for all the 14 test images]:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

The third image might be difficult to classify because it is blurred and a little bit rotated.
The fourth and fifth images might be difficult to classify because they are occluded by some other things.

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction for the 5 images listed above(results for other test images are in the ipython notebook)

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Speed limit(30km/h)     |  Speed limit(30km/h) 		| 
| Speed limit(120km/h)     |  Speed limit(120km/h) 		| 
| Stop					| Stop											|
| Beaware of ice/snow	      		| Beaware of ice/snow 				|
| Speed limit(100km/h)     |  Speed limit(80km/h) 		| 


The model was able to correctly guess 4 of the above listed 5 traffic signs, which gives an accuracy of 80%(for the whole 14 test images i downloaded from the internet, the model has an accuracy of 0.714286), This compares favorably to the accuracy on the test set of 94.10%.

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a "Speed limit(30km/h)" sign (probability of 0.999059), and the image does contain a "Speed limit(30km/h)" sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 0.999059       			| Speed limit (30km/h) | 
| 0.000375    				| Speed limit (50kmh) |
| 0.000344					| Stop |
| 0.000097	      			| Speed limit (60km/h) |
| 0.000095				    | Speed limit (80km/h)    |


For the rest images, please refer to the output of the 11th cell of the ipython notebook.

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
####1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


