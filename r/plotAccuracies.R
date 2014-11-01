setwd("N:/jen/noise/r") 
library(ggplot2)

path = "../results/"
fileName = "original_polBlog_flipResultsBaselines"

data <- read.table(file = paste(path, fileName , '.txt' , sep = ""), header = T)
  
pd <- position_dodge(.025)

ggplot(data, aes(x=trainingSize, y=Accuracy_Mean, colour=Label)) + 
  geom_errorbar(aes(ymin=Accuracy_Mean-Accuracy_SD, ymax=Accuracy_Mean+Accuracy_SD), width=.02,position = pd) +
  geom_line(position = pd) +
  geom_point(position = pd) 

ggsave(file=paste('./plots/', fileName , '-errorbars.png' , sep = ""))

pd <- position_dodge(0)
ggplot(data, aes(x=trainingSize, y=Accuracy_Mean, colour=Label)) + 
  geom_line(position = pd) +
  geom_point(position = pd) 

ggsave(file=paste('./plots/', fileName , '.png' , sep = ""))
