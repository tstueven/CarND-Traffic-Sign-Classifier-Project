# **Traffic Sign Recognition**

## Writeup

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:

* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report

[//]: # (Image References)

[sample_hist]: ./plots/sign_hist.png "Visualization"

[image2]: ./examples/grayscale.jpg "Grayscaling"

[image3]: ./examples/random_noise.jpg "Random Noise"

[image4]: ./traffic_sign_screenshots_google_maps/Vorfahrt.png "Traffic Sign 1"

[image5]: ./traffic_sign_screenshots_google_maps/Einbahnstrasse.png "Traffic Sign 2"

[image6]: ./traffic_sign_screenshots_google_maps/Halteverbot.png "Traffic Sign 3"

[image7]: ./traffic_sign_screenshots_google_maps/Busspur.png "Traffic Sign 4"

[image8]: ./traffic_sign_screenshots_google_maps/Vorfahrstrasse.png "Traffic Sign 5"

## Rubric Points

### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.

---

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to
my [project code](https://github.com/tstueven/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic signs
data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is 32x32
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing
how the data is distributed over the different classes. One can see that some
signs appear a lot more often than others. That might be on purpose because
maybe some signs are more difficult to classify than others, but I think this is
not in general a good dataset. Also, a lot off common traffic signs are
completely missing which leads to large problems later.

![alt text][sample_hist]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to convert the images to grayscale because results
were a lot better than when I kept all color channels. I found this
counter-intuitive since many signs are distinctively red or blue which should
help with classification. I probably just gets too complex when keeping all
these information and this makes it a lot more difficult for the model to
converge.

Here is an example of a traffic sign image before and after grayscaling.

![alt text][image2]

As a last step, I normalized the image data because that is beneficial for
learning.

I decided against the effort of generating additional data because I didn't
expect it to help with the biggest problem in the end for which I'd have had to
generate whole new classes.

#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer                |     Description                                | 
|:---------------------:|:---------------------------------------------:| 
| Input                | 32x32x1 grayscale image                            | 
| Convolution 5x5        | 1x1 stride, valid padding, outputs 28x28x16    |
| RELU                    |												|
| Max pooling            | 2x2 stride, outputs 14x14x16              |
| Dropout                | 0.5                      |
| Convolution 5x5        | 1x1 stride, valid padding, outputs 10x10x32    |
| RELU                    |												|
| Max pooling            | 2x2 stride, outputs 5x5x32          |
| Dropout                | 0.5                      |
| Flattening              | outputs 800    |
| Fully connected        | ouputs 280         |
| RELU                    |												|
| Dropout                | 0.5                      |
| Fully connected        | ouputs 160         |
| RELU                    |												|
| Dropout                | 0.5                      |
| Fully connected        | ouputs 43         |
| Softmax                |                                            |
|						|												|
| Cross Entropy Error    |    For Learning            |

#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used the Adam optimizer with a batch size of 128 and a
learning rate of 0.001. These were defaults taken from LeNet and modifications
did not seem to lead to significantly better results. After about 50 epochs the
model did not seem to improve on validation accuracy, so I let the training stop
after 80 epochs.

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

I started with the LeNet because the problem did not seem to different from
identifying handwritten digits (low number of classes, low resolution, little
complexity). First I only adapted the input size. Then I hoped to get better
results with dropout and probably replacing the max pooling which seems to be
out of fashion. To let the network have a chance to evaluate still well with
dropout I upped the number of convolutional filters. Max pooling turned out to
be important for performance after all and was reintroduced. In the end the only
real significant performance gain came from using grayscale instead of the color
channels as input.

There did not seem to be a problem with overfitting in the sense that validation
results get worse again after a number of epochs. The problem might rather be
called "saturation fitting" because the training accuracy was so close to 1 that
there was not much room left for improvement from those data.

My final model results were:

* training set accuracy of 0.998
* validation set accuracy of 0.961
* test set accuracy of 0.957

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I took as screenshots from Google Street
View in Hamburg.

![alt text][image4] ![alt text][image5] ![alt text][image6]
![alt text][image7] ![alt text][image8]

The first and second image might be difficult to classify because the view is
tilted. The sign in the secon image is also relatively small.

The third and fourth image are impossible to classify because they were not
represented in the original data set. I only realized this after I had already
processed them and was disappointed about the result. I then kept then because I
found the behaviour interesting.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image                    |     Prediction                                | 
|:---------------------:|:---------------------------------------------:| 
| Right-of-way at the next intersection| Right-of-way at the next intersection | 
| No entry                | Turn right ahead       |
| No stopping                | Right-of-way at the next intersection                 |
| Bus lane               | No entry                                    |
| Priority road           | Priority road                               |

The model was able to correctly guess 2 of the 5 traffic signs, which gives an
accuracy of 40%. This is a lot worse than in the test set but to be expected in
retrospect since two images were impossible to classify. I kept them though
because the results in the next section I found quite interesting.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th code
cell of the Ipython notebook.

For the first image, the model is extremely sure that this is a Right-of-way at
the next intersection sign (
probability of over .9997), and the image does indeed contain a Right-of-way at
the next intersection sign.

The top five soft max probabilities were

| Probability            |     Prediction                                | 
|:---------------------:|:---------------------------------------------:| 
| .9997                 | Right-of-way at the next intersection                                    | 
| .0001                    | General caution                                        |
| <.0001                    | Pedestrians                                      |
| <.0001                    | Double curve  |
| <.0001                    | Beware of ice/snow          |

For the second image is not sure at all and does not have to correct result in
its top five. This is really disappointing since the No entry sign was 990 times
in the training data set. I feel like there must be an error but was not able to
find it. Probably in the end the problem lies in the small size of the sign
within the image. At least the model is really unsure about its predictions.

The top five soft max probabilities were

| Probability            |     Prediction                                | 
|:---------------------:|:---------------------------------------------:| 
| .35                 | Turn right ahead                                    | 
| .27                    | Yield                                        |
| .09                    | Priority road                                      |
| .05                   | No passing for vehicles over 3.5 metric tons  |
| .04                    | Road work                              |

For the third image there is no good prediction which is not surprising as the
No stopping sign was not part of the data. With a threshold one could easily
make sure the network returns at least no result instead of a wrong result.

The top five soft max probabilities were

| Probability            |     Prediction                                | 
|:---------------------:|:---------------------------------------------:| 
| .50                 |Right-of-way at the next intersection    | 
| .12                    | Priority road                                        |
| .06                    | Beware of ice/snow                                      |
| .05                   | Roundabout mandatory  |
| .05                    | Keep right                              |

The fourth image was predicted wrongly. The model is quite certain that a
sign it has never seen before is the No entry sign which is was not able to
recognize in the second image. Here I'd probably help to somehow make better use
of the colors than I was able to.

The top five soft max probabilities were

| Probability            |     Prediction                                | 
|:---------------------:|:---------------------------------------------:| 
| .93                 | No entry                                    | 
| .02                    | Keep right                                        |
| .01                    | Roundabout mandatory                                      |
| .01                   | Turn left ahead  |
| .01                    | Stop                              |

The fifths image was classified perfectly! :)

The top five soft max probabilities were

| Probability            |     Prediction                                | 
|:---------------------:|:---------------------------------------------:| 
| ~1                 | Priority road                      | 
| ~0                    | Roundabout mandatory     |
| ~0                    | Ahead only                |
| ~0                   | Yield |
| ~0                    | Road work           |

