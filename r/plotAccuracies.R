setwd("N:/jen/noise/r") 
library(ggplot2)

pd <- position_dodge(.025)

ggplot(data, aes(x=trainingSize, y=Accuracy_Mean, colour=Label)) + 
  geom_errorbar(aes(ymin=Accuracy_Mean-Accuracy_SE, ymax=Accuracy_Mean+Accuracy_SE), width=.02,position = pd) +
  geom_line(position = pd) +
  geom_point(position = pd)  +
  ggtitle( paste(fileName ) )+
  scale_colour_manual(values = rhg_cols)

ggsave(file=paste('./plots/', fileName , 'PriorClass0-errorbars.png' , sep = ""))




rhg_cols <- c("maroon", "dodgerblue1", "skyblue2", 
              "chocolate1", "bisque3",  "lightgreen", "mediumpurple3",
              "palevioletred1", "lightsalmon4", "black", "darkgoldenrod1")

rhg_cols <- c("maroon", "dodgerblue1", "skyblue2", 
              "chocolate1", "bisque3",  "lightgreen", "mediumpurple3",
              "palevioletred1", "lightsalmon4", "darkgoldenrod1", "black")

rhg_cols <- c("maroon", "dodgerblue1", "skyblue2", 
              "chocolate1", "bisque3",  "lightgreen",  "black","mediumpurple3",
              "palevioletred1", "lightsalmon4", "darkgoldenrod1")

rhg_cols <- c("maroon", "dodgerblue1",  
              "chocolate1", "bisque3","black","lightgreen","skyblue2","mediumpurple3",
              "palevioletred1", "lightsalmon4", "darkgoldenrod1")

rhg_cols <- c("maroon", "dodgerblue1",  
              "chocolate1", "black", "skyblue2", "bisque3",  "lightgreen",  "mediumpurple3",
              "palevioletred1", "lightsalmon4", "darkgoldenrod1")




path = "../results/"
titleName = "rewireEdges"
fileName = paste("school074-label0_run2_", titleName ,"Results",sep="")
yLabel = "squaredLoss"

data <- read.table(file = paste(path, fileName , '.txt' , sep = ""), header = T)



pd <- position_dodge(0)
ggplot(data, aes(x=trainingSize, y=SquaredLoss_Mean, colour=Label)) + 
  geom_line(position = pd) +
  geom_point(position = pd) +
  ggtitle( fileName ) +
  ylab(yLabel) 
#+ scale_colour_manual(values = rhg_cols) 
+  ylim(0.20,0.45)  

suffix = paste(yLabel,"")
ggsave(file=paste('./plots/', fileName, '_', suffix, '.png' , sep = ""))




pd <- position_dodge(.025)
ggplot(data, aes(x=trainingSize, y=PriorClass0_Mean, colour=Label)) + 
  geom_errorbar(aes(ymin=PriorClass0_Mean-PriorClass0_SE, ymax=PriorClass0_Mean+PriorClass0_SE), width=.02,position = pd) +
  geom_line(position = pd) +
  geom_point(position = pd)  +
  ggtitle( paste(fileName) ) +
  scale_colour_manual(values = rhg_cols)



data <- subset(data,Label == "original" | 
                 Label ==  "05perc_2repeat" |
                 Label ==  "05perc_5repeat" |
                 Label ==  "05perc_10repeat" |
                 Label ==  "70perc_2repeat" |
                 Label ==  "70perc_5repeat" |
                 Label ==  "70perc_10repeat" |
                 Label ==  "60perc_2repeat" |
                 Label ==  "60perc_5repeat" |
                 Label ==  "60perc_10repeat" |
                 Label ==  "100perc_2repeat" |
                 Label ==  "100perc_5repeat" |
                 Label ==  "100perc_10repeat"
)

data <- subset(data,Label == "original" | 
                 Label ==  "05perc_10repeat" |
                 Label ==  "15perc_10repeat" |
                 Label ==  "30perc_10repeat" |
                 Label ==  "40perc_10repeat" |
                 Label ==  "50perc_10repeat" |
                 Label ==  "60perc_10repeat" |
                 Label ==  "70perc_10repeat" |
                 Label ==  "80perc_10repeat" |
                 Label ==  "90perc_10repeat" |
                 Label ==  "100perc_10repeat"
)

data <- subset(data,Label == "original" | 
                 Label ==  "05perc_5repeat" |
                 Label ==  "15perc_5repeat" |
                 Label ==  "30perc_5repeat" |
                 Label ==  "40perc_5repeat" |
                 Label ==  "50perc_5repeat" |
                 Label ==  "60perc_5repeat" |
                 Label ==  "70perc_5repeat" |
                 Label ==  "80perc_5repeat" |
                 Label ==  "90perc_5repeat" |
                 Label ==  "100perc_5repeat"
)

data <- subset(data,Label == "original" | 
                 Label ==  "05perc_2repeat" |
                 Label ==  "15perc_2repeat" |
                 Label ==  "30perc_2repeat" |
                 Label ==  "40perc_2repeat" |
                 Label ==  "50perc_2repeat" |
                 Label ==  "60perc_2repeat" |
                 Label ==  "70perc_2repeat" |
                 Label ==  "80perc_2repeat" |
                 Label ==  "90perc_2repeat" |
                 Label ==  "100perc_2repeat"
)


################ Depreicated

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
