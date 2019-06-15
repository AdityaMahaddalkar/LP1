
# install.packages(c("ggplot2", "readr", "gridExtra", "grid", "plyr"))
library(ggplot2)
library(readr)
library(gridExtra)
library(grid)
library(plyr)

iris = read.csv("iris.csv")

summary(iris)

# Sepal length
HisSl <- ggplot(data=iris, aes(x=SepalLengthCm)) +
	geom_histogram(binwidth=0.2, color="black", aes(fill=Species))+
	xlab("Sepal length") +
	ylab("Frequency") +
	theme(legend.position="none") +
	ggtitle("Histogram of Sepal Length") +
	geom_vline(data=iris, aes(xintercept = mean(SepalLengthCm)), linetype="dashed", color="grey")

# Sepal width
HistSw <- ggplot(data=iris, aes(x=SepalWidthCm)) +
  geom_histogram(binwidth=0.2, color="black", aes(fill=Species)) + 
  xlab("Sepal Width (cm)") +  
  ylab("Frequency") + 
  theme(legend.position="none")+
  ggtitle("Histogram of Sepal Width")+
  geom_vline(data=iris, aes(xintercept = mean(SepalWidthCm)),linetype="dashed",color="grey")


# Petal length
HistPl <- ggplot(data=iris, aes(x=PetalLengthCm))+
  geom_histogram(binwidth=0.2, color="black", aes(fill=Species)) + 
  xlab("Petal Length (cm)") +  
  ylab("Frequency") + 
  theme(legend.position="none")+
  ggtitle("Histogram of Petal Length")+
  geom_vline(data=iris, aes(xintercept = mean(PetalLengthCm)),
             linetype="dashed",color="grey")



# Petal width
HistPw <- ggplot(data=iris, aes(x=PetalWidthCm))+
  geom_histogram(binwidth=0.2, color="black", aes(fill=Species)) + 
  xlab("Petal Width (cm)") +  
  ylab("Frequency") + 
  theme(legend.position="right" )+
  ggtitle("Histogram of Petal Width")+
  geom_vline(data=iris, aes(xintercept = mean(PetalWidthCm)),linetype="dashed",color="grey")


# Plot all visualizations
grid.arrange(HisSl + ggtitle(""),
             HistSw + ggtitle(""),
             HistPl + ggtitle(""),
             HistPw  + ggtitle(""),
             nrow = 2,
             top = textGrob("Iris Frequency Histogram", 
                            gp=gpar(fontsize=15))
)

#boxplots
Box1 <- boxplot(iris$SepalLengthCm ~ iris$Species)
Box2 <- boxplot(iris$SepalWidthCm ~ iris$Species)
Box3 <- boxplot(iris$PetalLengthCm ~ iris$Species)
Box4 <- boxplot(iris$PetalWidthCm ~ iris$Species)


