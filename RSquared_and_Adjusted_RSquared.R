# These are some notes on the intution behind R-Squared (for SimpleLinearRegression)
# and  Adjusted R^2 (for Multiple Linear Regression, same concepts apply)

------------------------------------ R^2----------------------------------------

"Sum of Squares of Residuals = Sum( y_i - y_ihat)^2

Total Sum of Squares  = Sum(y_i - y_avg)^2

Let SS denote "Sum of Squares"

R^2 = 1 - (SS_residuals / SS_total)

What you're trying to do with your regression is you're trying to fit a line to minimize the 
SS_residuals.

So we're trying to fit the best line.

R^2 is telling us, "How good is your line compared to the average line?: ______"

In order to fit the best fitting line we need to run a Regression.

So, ideally, if your SS_residuals goes to zero, then your R^2 goes to one, IDEALLY. Normally never happens.

So the closer R^2 is to one, the better; the further from 1, the worse. 

Question: Can R^2 be negative? 

Answer: Yes, R^2 can be negative if your SS_residual actually fits your data WORSE than the average
line. It also probably implies that the model is broken."


-------------------------------------Adjusted R^2---------------------------------
  
" SS_residuals ---> Min

So we want to use R^2 as a goodness of fit parameter (the greater the better)
  
The problem arises when we introduce more variables.

Lemma: R^2 will never decrease. 

     Proof: R^2 = 1 - (SS_residuals / SS_total). 

            Once you add a new variable to your model. Its going to somehow affect what the model looks
            like. 

            The fact that we are trying to minimize the sum of squares of residuals, means that, either
            this new variable will help minimize the sum of squares of residuals, i.e, somehow 
            the regression process will find a way to give it a coefficient that will help minimize
            the sum of square residuals and in that case, R^2 will be 
                R^2 = 1 - (SS_residuals / SS_total). "1 MINUS SOMETHING LESS THAN WHAT IT USED TO BE"
                divided by the same value.

Because by adding a new variable we are not affecting the observations and we're not
affecting the averages of the observations. So by adding a new variable,
  the regression process, through the condition; SS_residuals ---> Min will definitely 
try to minimize this value; make it even smaller than it currently is.

          If in the SPECIAL case where you DO add an independent variable, and whatever coefficent
          you give it, you cannot decrease SS_residual, then that coefficient becomes zero. (Very Rare)


So pretty much only two options: R^2 will either increase or not change at all.

Coefficients are very rarely zero, because there will always be a SLIGHT random correlation between
the independent variable and the dependent variable."


----------------> R^2 is BIASED <-------------- it increases no matter if is improvement or non-improvement

But,  -----------------------------> Adjusted R^2 <--------------------- 
  
"So we use Adjusted R^2 = 1 - [ ( 1 - R^2) * [(n - 1)/(n-p-1)] ]

Where p is the number of regressors and n is the sample size
  
Adjusted R^2 has a "penalization factor"

"As you add more regressors, the adjusted R^2 decreases."
  
  
  
  