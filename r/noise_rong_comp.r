setwd("N:/jen/noise/r") 
library(ggplot2)

#fileName = "Rongjing-mple_noise-school074-label0-flipLabel"
fileName = "Rongjing-mple_noise-facebook-label0-flipLabel"

data <- read.table(file = paste('../results/', fileName , '.txt' , sep = ""), header = T, sep="\t")

yLabel = "Squared Loss"


#d2 <- reshape(data,direction="long",idvar="trainingSize",varying=list(2:3),v.names = "squaredLoss")
#d2[,2] <- factor(d2[,2])
#levels(d2[,2]) <- c("originalPMLE","rongjingAlgo")


pd <- position_dodge(0)
ggplot(data, aes(x=trainingSize, y=meanSquaredLoss, color=Label) ) + 
  geom_line(position = pd) +
  geom_point(position = pd) +
  ggtitle( fileName ) +
  ylab(yLabel) + ylim(0.10,0.45)  
#+
# scale_colour_manual(values = rhg_cols) +
#  

suffix = paste(yLabel,"")
ggsave(file=paste('./plots/', fileName, '_', suffix, '.png' , sep = ""),width=9.69,height=7.79)



