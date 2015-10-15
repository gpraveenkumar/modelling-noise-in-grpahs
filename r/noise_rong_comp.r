setwd("N:/jen/noise/r") 
library(ggplot2)

fileName = "Rongjing-mple_noise-school074-label0-flipLabel"
fileName = "Rongjing-noisefacebook-label0-algo_cleanedRepeats"
#fileName = "Rongjing-mple_noise-facebook-label0-flipLabel"

data <- read.table(file = paste('../results/', fileName , '.txt' , sep = ""), header = T, sep="\t")

yLabel = "PMLE Squared Loss"


#d2 <- reshape(data,direction="long",idvar="trainingSize",varying=list(2:3),v.names = "squaredLoss")
#d2[,2] <- factor(d2[,2])
#levels(d2[,2]) <- c("originalPMLE","rongjingAlgo")


pd <- position_dodge(0)
ggplot(data, aes(x=trainingSize, y=i2_meanSquaredLoss, color=Label) ) + 
  geom_line(position = pd) +
  geom_point(position = pd) +
  ggtitle( fileName ) +
  ylab(yLabel) + ylim(0.10,0.45)  
#+
# scale_colour_manual(values = rhg_cols) +
#  

suffix="_PMLEModel"
suffix = paste(yLabel,suffix,"")
ggsave(file=paste('./plots/', fileName, '_', suffix, '.png' , sep = ""),width=9.69,height=7.79)


data <- subset(data,Label == "originalPMLE" |  Label == "rongjingAlgo" | Label == "rongjingPMLE" | 
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

data <- subset(data,Label == "originalPMLE" | Label == "rongjingAlgo" | Label == "rongjingPMLE" | 
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

data <- subset(data,Label == "originalPMLE" |  Label == "rongjingAlgo" | Label == "rongjingPMLE" | 
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



