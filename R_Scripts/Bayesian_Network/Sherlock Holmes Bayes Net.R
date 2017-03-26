#Sherlock Holmes Murder Case using Bayes Net
library(gRain)
library(gRbase)
library(ggm)


#Building the DAG

#Visualization Plot
g <- list(~criminal, ~scene | criminal , ~knife | criminal , ~expert | knife, ~height | scene)

crimedag <- dagList(g)

plot(crimedag)


#Assigning Probabilities

##Levels
suspect <- c("one", "two", "three")

##Conditional Probabilities Tables (CPTs)
c <- cptable(~criminal, values = c(1/3, 1/3, 1/3), levels = suspect)

s.c <- cptable(~scene | criminal, values = c(0.8, 0.1, 0.1, 0.1, 0.8, 0.1, 0.1, 0.1, 0.8), levels = suspect)

h.s <- cptable(~height | scene, values = c(0.6, 0.3, 0.1, 0.1, 0.6, 0.3, 0.1, 0.4, 0.5), levels = suspect)

k.c <- cptable(~knife | criminal, values = c(0.9, 0.05, 0.05, 0.05, 0.9, 0.05, 0.05, 0.05, 0.9), levels = suspect)

e.k <- cptable(~expert | knife, values = c(0.8, 0.1, 0.1, 0.1, 0.8, 0.1, 0.1, 0.1, 0.8), levels = suspect)

##Compiling
plist <- compileCPT(list(c, s.c, h.s, k.c, e.k))
grn1 <- grain(plist)
plot(grn1)

#Set Evidence and run query
find1 <- setFinding(grn1, nodes = c("height", "expert"), states = c("one", "three"))
querygrain(find1, nodes = c("criminal"), type = "marginal")

#Other Queries
querygrain(find1, nodes = c("scene", "knife"), type = "joint")

querygrain(grn1, nodes = c("criminal", "scene"), type = "conditional")
