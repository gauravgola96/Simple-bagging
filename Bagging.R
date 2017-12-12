## R Package for Bagging
install.packages("ipred")
library(ipred)

##  Classification: german credit

names(GermanCredit)

class(GermanCredit$Creditability)
# Convert target variable to factor
GermanCredit$Creditability <- factor(GermanCredit$Creditability)

German.bagging <- bagging(Creditability ~.,
                           data=GermanCredit,
                           mfinal=15, 
                           control=rpart.control(maxdepth=5, minsplit=15))