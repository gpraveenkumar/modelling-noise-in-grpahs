setwd("N:/jen/noise/r") 
library(ggplot2)
library(gridExtra)


path = "../results/"
fileName = "distributionTest_school_dropLabelResults"

data <- read.table(file = paste(path, fileName , '.txt' , sep = ""), header = T)

plot_main <- ggplot(data, aes(x = 1:100, y = Accuracy_Mean)) + geom_point() + geom_line() +
  geom_hline(aes(yintercept=mean(Accuracy_Mean)), color="red", linetype="solid", size=1) +
  geom_hline(aes(yintercept=mean(Accuracy_Mean) - sd(Accuracy_Mean)), color="blue", linetype="dashed", size=1) +
  geom_hline(aes(yintercept=mean(Accuracy_Mean) + sd(Accuracy_Mean)), color="blue", linetype="dashed", size=1) +
  geom_hline(aes(yintercept=mean(Accuracy_Mean) - 2*sd(Accuracy_Mean)), color="green", linetype="dashed", size=1) +
  geom_hline(aes(yintercept=mean(Accuracy_Mean) + 2*sd(Accuracy_Mean)), color="green", linetype="dashed", size=1) +
  geom_hline(aes(yintercept=mean(Accuracy_Mean) - 3*sd(Accuracy_Mean)), color="orange", linetype="dashed", size=1) +
  geom_hline(aes(yintercept=mean(Accuracy_Mean) + 3*sd(Accuracy_Mean)), color="orange", linetype="dashed", size=1) +
  ylim(0.52,0.59) +
  geom_hline(aes(yintercept=median(Accuracy_Mean)), color="yellow", linetype="dashed", size=1)

plot_right <- ggplot(data, aes(x = Accuracy_Mean)) + 
  geom_density(alpha=.5) +
  geom_vline(aes(xintercept=mean(Accuracy_Mean)), color="red", linetype="solid", size=1) +
  geom_vline(aes(xintercept=mean(Accuracy_Mean) - sd(Accuracy_Mean)), color="blue", linetype="dashed", size=1) +
  geom_vline(aes(xintercept=mean(Accuracy_Mean) + sd(Accuracy_Mean)), color="blue", linetype="dashed", size=1) +
  geom_vline(aes(xintercept=mean(Accuracy_Mean) - 2*sd(Accuracy_Mean)), color="green", linetype="dashed", size=1) +
  geom_vline(aes(xintercept=mean(Accuracy_Mean) + 2*sd(Accuracy_Mean)), color="green", linetype="dashed", size=1) +
  geom_vline(aes(xintercept=mean(Accuracy_Mean) - 3*sd(Accuracy_Mean)), color="orange", linetype="dashed", size=1) +
  geom_vline(aes(xintercept=mean(Accuracy_Mean) + 3*sd(Accuracy_Mean)), color="orange", linetype="dashed", size=1) +
  geom_vline(aes(xintercept=median(Accuracy_Mean)), color="yellow", linetype="dashed", size=1) +
  xlim(0.52,0.59) + 
  coord_flip() 

#arrange the plots together, with appropriate height and width for each row and column
grid.arrange(plot_main, plot_right, ncol=2, nrow=1, widths=c(4, 1), heights=c(1, 4))

g <- arrangeGrob(plot_main, plot_right, ncol=2, nrow=1, widths=c(4, 1), heights=c(1, 4))

ggsave(file=paste('./plots/', fileName , '.png' , sep = ""),g)



