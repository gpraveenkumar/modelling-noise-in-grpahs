setwd("N:/jen/noise/r") 
library(ggplot2)

pd <- position_dodge(.025)

ggplot(data, aes(x=trainingSize, y=Accuracy_Mean, colour=Label)) + 
  geom_errorbar(aes(ymin=Accuracy_Mean-Accuracy_SE, ymax=Accuracy_Mean+Accuracy_SE), width=.02,position = pd) +
  geom_line(position = pd) +
  geom_point(position = pd)  +
  ggtitle( paste(fileName ,suffix) )+
  scale_colour_manual(values = rhg_cols)

ggsave(file=paste('./plots/', fileName , '-errorbars.png' , sep = ""))




rhg_cols <- c("maroon", "dodgerblue1", "skyblue2", 
              "chocolate1", "bisque3",  "lightgreen", "mediumpurple3",
              "palevioletred1", "lightsalmon4", "black", "darkgoldenrod1")

rhg_cols <- c("maroon", "dodgerblue1", "skyblue2", 
              "chocolate1", "bisque3",  "lightgreen",  "black","mediumpurple3",
              "palevioletred1", "lightsalmon4", "darkgoldenrod1")

rhg_cols <- c("maroon", "dodgerblue1",  
              "chocolate1", "black", "skyblue2", "bisque3",  "lightgreen",  "mediumpurple3",
              "palevioletred1", "lightsalmon4", "darkgoldenrod1")



path = "../results/"
#fileName = "polBlog_flipResults"
#fileName = "school_flipLabelResults"
#fileName = "school_dropLabelResults"
fileName = "school_dropEdgesResults"
#fileName = "school_rewireEdgesResults"
suffix = ""
data <- read.table(file = paste(path, fileName , '.txt' , sep = ""), header = T)
#data <- subset(data,trainingSize <= 0.10)
data <- subset(data,Label != "10perc_2repeat")
data <- subset(data,Label != "10perc_5repeat")
data <- subset(data,Label != "10perc_10repeat")
data <- subset(data,Label != "20perc_2repeat")
data <- subset(data,Label != "20perc_5repeat")
data <- subset(data,Label != "20perc_10repeat")
data <- subset(data,Label != "25perc_2repeat")
data <- subset(data,Label != "25perc_5repeat")
data <- subset(data,Label != "25perc_10repeat")
#data <- subset(data,Label != "15perc_2repeat")
#data <- subset(data,Label != "15perc_5repeat")
#data <- subset(data,Label != "15perc_10repeat")
#data <- subset(data,Label != "30perc_2repeat")
#data <- subset(data,Label != "30perc_5repeat")
#data <- subset(data,Label != "30perc_10repeat")
#data <- subset(data,Label != "5perc_2repeat")
#data <- subset(data,Label != "5perc_5repeat")
#data <- subset(data,Label != "5perc_10repeat")


pd <- position_dodge(0)
ggplot(data, aes(x=trainingSize, y=Accuracy_Mean, colour=Label)) + 
  geom_line(position = pd) +
  geom_point(position = pd) +
  ggtitle( paste(fileName ,suffix) )+
  scale_colour_manual(values = rhg_cols)

ggsave(file=paste('./plots/', fileName ,suffix, '.png' , sep = ""))




