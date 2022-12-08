SOCIAL MEDIA SHARES
by Nota Gaia (264341, captain), Dospinescu Sara (26851), Giannandrea Giulia (269321)

INTRODUCTION

What is the best artificial intelligence model, in terms of accuracy and efficiency, to predict the number of social media shares of a company based on contents and supposed publication time? 
This is the question we tried to answer to,  for our project. 
We found ourselves hired by a company to help the social media department analyse its communications’ success, having as knowledge and skills the ones learnt during AI and Machine Learning lectures and having as dataset the one provided by the company itself.
Our dataset stores data about features(such as words, links, videos, images); keywords; topics; some variables; also stores information about the contents' sentiment analysis (for instance about subjectivity, polarity, rates of positive and negative words in the content); popularity of the content; time of pubblication (day of the week in particular).

METHODS

After importing the needed libraries and the dataset, we isolated the shares' variable column, since it is the the dependent variable, so the value we want to predict. 
We procedeed with an explanatory data analysis throu visualizations (one by one independent variable on x axis and always shares' variable on y axis). Since we noticed that some of these graphics we plotted were very similar, we decided to start the data cleaning part finding and deleting all the variables with a correlation higher than a certain threshold. Our first attempt was to set the theshold at 0.85, but soon we realized that it was definitely too low. Our second and final attempt was 0.95, and from it two columns (non_stop_words, non_stop_unique_tokens) were deleted, so we had 56 variables in total. Thanks to these plots we could also see that there is not linearity between the regressor and the response variable.
Right after we divided the dataset in train and test sets, choosing as split size respectively 0.73/0.27.
The next step was scaling data: we tried different ways to scale our data. 
Initially we tried robust scalar; then our manual implementation for the outliers' removal with standard scalar; we also implemented from scratch a piece of code to remove outliers with robust scalar; in the end we tried both robust scalar and standard scalar together. Only after all these experiments, we understood that robust scalar was the most efficient way to scale our data, having as advantage the automatic outliers' deletion.

EXPERIMENTAL DESIGN

Initially, we took in consideration all the regression models we knew and we analyzed theorically all of them, one by one, to choose which ones to implement.
We immediately discarded K-NN because it works better on small datasets and this was not our case; we doubted CART TREES because we thought that, in order to effectively and accurately analyse our dataset, we would need a more complex structure than a simple tree. So we preferred to implement at the end Random Forest, SVR and Polynomial Regression.
Even though we had already realised that our data were not linearly dependent, we still implemented Linear Regression with very poor results, as expected. 

To evaluate our models' performance, we chose to calculate the mean squared error, which measures the average of the squares of the errors, that is the average squared difference between the estimated values and the actual value; and the mean absolute error, that is a measure of errors between paired observations expressing the same phenomenon, is calculated as the sum of absolute errors divided by the sample size.

Addionally, to make sure that every model is evaluated at its best, so at its lowest error score, we used CrossValidation to calculate the best combination of hyperparameters of each of the models we implemented. 
For example, the best hyperparameter for Polynomial Regression is degree 2.

RESULTS

We computed mean squared error and mean absolute error for the three models we chose. Here there are the results:

SVR -> mean square error is: 11013.363209211355; mean absolute error is: 2427.9992616525187
Random Forest -> mean square error is: 10739.47526604121; mean absolute error is: 3053.8303148987284
Polynomial Regression -> mean square error is: 20942.150106090856; mean absolute error is: 4576.989176063572

Polynomial Regression is visibly the worst model in terms of accuracy, infact both the MSE and the MAE are so much higher with respect to the other two models.
Between SVR and Random Forest, for us, the best one is SVR. Even though the mean squared error is higher (+273,89), the MAE is definitely lower (-625,84) plus the running time, which is less in the SVR case.

Once we found out which is the best model, we validate it. We restrict the set of independent variables at just the most relevant ones and on these we run again SVR. We evaluate the output and we see that the mean square error is  10978.618879950773, the mean absolute error is  2409.3500586094356.

CONCLUSIONS
