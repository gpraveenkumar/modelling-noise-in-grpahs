setwd("N:/jen/noise/r") 
library(ggplot2)

fileName = "Rongjing-school074-label0-algo1"

data <- read.table(file = paste('../results/', fileName , '.txt' , sep = ""), header = T, sep="\t")
#ggplot(res, aes(x,fill=name)) + geom_histogram(position = 'dodge') + theme_bw()
#ggsave(file=paste('./plots/', fileName , '.png' , sep = ""), width=5, height=5)

d0 <- data.frame(x=data$trainingSize)
d1 <- data.frame(x=data$meanAccuracy)
d2 <- data.frame(x=data$mle_meanAccuracy)
d3 <- data.frame(x=data$mple_meanAccuracy)

d1$name <- 'Joint'
d2$name <- 'MLE'
d3$name <- 'MPLE'

res <- rbind(d1,d2,d3)
yLabel = "meanAccuracy"


x <- 6
d1 <- data[,c(1,3,9,12)]
d2 <- reshape(d1,direction="long",idvar="trainingSize",varying=list(2:4),v.names = "accuracy")
d2[,2] <- factor(d2[,2])
#levels(d2[,2]) <- c("NodeFeatures","NodeFatures + UserPosting","NodeFeatures + WallPosted","NodeFeatures + UserPosting + WallPosted")
levels(d2[,2]) <- c("Joint","MLE","MPLE")
d2[,3] <- 1 - d2[,3]

pd <- position_dodge(0)
ggplot(d2, aes(x=trainingSize, y=accuracy, color=time)) + 
  geom_line(position = pd) +
  geom_point(position = pd) +
  ggtitle( fileName ) +
  ylab(yLabel) 



pd <- position_dodge(0)
ggplot(d2, aes(x=trainingSize, y=accuracy, color=time)) + 
  geom_line(position = pd) +
  geom_point(position = pd) +
  ggtitle( fileName ) +
  ylab(yLabel) 

suffix = paste(yLabel,"")
ggsave(file=paste('./plots/', fileName, '_', suffix, '.png' , sep = ""),width=9.69,height=7.79)





