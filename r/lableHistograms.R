setwd("N:/jen/noise/r") 
library(ggplot2)

fileName = "p74-feature1"

data <- read.table(file = paste('../data/', fileName , '.txt' , sep = ""), header = T)
ggplot(data, aes(x=ratings)) + geom_histogram(binwidth=0.5) + theme_bw()
ggsave(file=paste('./plots/', fileName , '.png' , sep = ""), width=5, height=5)


table(data$rating)
val <- 1
sum(data$rating < val)
sum(data$rating >= val)

d1 <- read.table(file = paste('../data/', "34-feature1" , '.txt' , sep = ""), header = T)
d2 <- read.table(file = paste('../data/', "34-feature2" , '.txt' , sep = ""), header = T)
d3 <- read.table(file = paste('../data/', "34-feature3" , '.txt' , sep = ""), header = T)

fileName = "34-feature-all"

dat <- read.table(file = paste('../data/', fileName , '.txt' , sep = ""), header = T)
cor(dat)
