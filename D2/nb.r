# install.packages(c("tidyverse", "ggplot2", "caret", "caretEnsemble", "psych", "Amelia", "mice", "GGally", "rpart"))
# library(tidyverse)
library(ggplot2)
library(caret)
library(caretEnsemble)
library(psych)
library(Amelia)
library(mice)
library(GGally)
library(rpart)

data <- read.csv("diabetes.csv")

data$Outcome <- factor(data$Outcome, levels = c(0, 1), labels = c("False", "True"))
