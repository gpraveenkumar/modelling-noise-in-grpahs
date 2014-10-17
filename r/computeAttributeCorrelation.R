setwd("N:/jen/noise/r") 

data <- read.table(file = "../data/attributeCorrelationCheck.txt",header = T)

cor(data$A,data$B)
