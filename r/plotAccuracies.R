setwd("N:/jen/noise/r") 
library(ggplot2)

path = "../results/"
fileName = "flipResults"

data <- read.table(file = paste(path, fileName , '.txt' , sep = ""), header = T)
  
pd <- position_dodge(.015)
ggplot(data, aes(x=trainingSize, y=Accuracy_Mean, colour=Label)) + 
  geom_errorbar(aes(ymin=Accuracy_Mean-Accuracy_SD, ymax=Accuracy_Mean+Accuracy_SD), width=.02,position = pd) +
  geom_line(position = pd) +
  geom_point(position = pd) 
ggsave(file=paste('./plots/', fileName , '.png' , sep = ""))