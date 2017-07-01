#load libraries
library(arules)
library(arulesViz)

# Transactional data is stored in a slightly different format than that we used
# previously.

# In comparison, transactional data is a more free form. As usual, each row in the data
# specifies a single example—in this case, a transaction. However, rather than having a
# set number of features, each record comprises a comma-separated list of any number
# of items, from one to many.

# Creating a sparse matrix for transactional data. Each row in sparse matrix
# indicates a transaction
# Since there are 169 diff items, matrix will have 169 columns

# Creating a sparse matrix with read_transactions()
groceries <- read.transactions("groceries.csv", sep = ",")

# Summary of transactions
summary(groceries)

# 9835 transactions and 169 different items throughout
# 1 where item was purchased and 0 if not 
# Density of 0.02609146 refers to non-zero proportion of cells
# This implies 9835 * 169 * 0.02609146 = 43,367 items were purchased
# 4.409 distinct grocery items per transaction

# Sizes imply that 2159 transactions had only 1 item in it and one transaction
# had 32 items 

# Can't view the sparse matrix as wont be stored in a df. Use inspect()
# Inspect first five transacitons 
inspect(groceries[1:5])

# itemFrequency() lets us see the proportion of transactions that contain the item
# remember that rows are transactions and columns are items

# proportion of first 3 items (sorted by alphabetical order). The support, 
# proportion of transactions that contains the item 
itemFrequency(groceries[, 1:3])

# support of specific item 
itemFrequency(groceries[ , "baby cosmetics"])

# visualizing item support based on support (min 0.1)
itemFrequencyPlot(groceries, support = 0.1)

# Visualizing item support based on topN (top 20)
itemFrequencyPlot(groceries, topN = 20)

# Visualizing transaction data - plotting the sparse matrix
# Useful for identifying potential data issues
# Using the image() function
# Sparse matrix of first five transactions
image(groceries[1:5])

# if sort transactions by data, obvious patterns could reveal seasonality
# sample 100 transactions from our sparse matrix and visualize it
image(sample(groceries, 100))

# Applying the apriori algorithm
# there can sometimes be a fair amount of trial and error needed to 
# find the support and confidence parameters that produce a reasonable number of association rules.
# If you set these levels too high, you might find no rules or rules that are too generic to be very useful.
# On the other hand, a threshold too low might result in an unwieldy number of rules,
# or worse, the operation might take a very long time or run out of memory during the
# learning phase.

# setting support = 0.1 and confidence = 0.8 gives 0 associations
# this means that we need an item to appear in 983 transactions to even be considered
# think about the smallest number of transactions you would need before you would
# consider a pattern interesting

# confidence too low, large amount of unreliable rules
# confidence too high, obvious rules, eggs and flour

# helps to set minlen = 2 to eliminate rules that contains fewer than 2 items
# {} --> milk meets the confidence of 0.25, not actionable insight

# Applying the Apriori algorithm
groceryrules <- apriori(groceries, parameter = list(support = 0.006, confidence = 0.25, minlen = 2))

groceryrules

# Dig deeper to evaluate these rules
summary(groceryrules)

# In our rule set, 150 rules have only two items, while 297 have three, and 16
# have four.

# size of rule or length is LHS + RHS. {bread} -> {butter} has length of 2
# {pb, jelly} -> {bread} is three

# We might be alarmed if most or all of the rules had support and confidence very near the
# minimum thresholds, as this would mean that we may have set the bar too high.
# This is not the case here, as there are many rules with much higher values of each.

# Lift
# lift(X -> Y) = confidence(X -> Y) / support(Y)
# how much more likely one item/itemset is purchased relative to its typical rate
# lift(X,Y) > 1 implies {X,Y} is purchased more as a set than {Y} alone
# lift(X,Y) = lift(Y,X)
# A large lift value is therefore a strong indicator that a rule is important, and reflects a true connection
# between the items.

#inspecting specific rules generated from the algorithm
inspect(groceryrules[1:3])
inspect(groceryrules)

# improving model performance by sorting
inspect(sort(groceryrules, by = "lift")[1:5])

# with a lift of about 3.96, implies that people who buy herbs are nearly
# four times more likely to buy root vegetables than the typical customer

#Subsets of association rules
# find all rules that include berries
# using subset()

berryrules <- subset(groceryrules, items %in% "berries")
inspect(berryrules)

#saving the rules
write(groceryrules, file = "groceryrules.csv", sep = ",", quote = TRUE, row.names = FALSE)

groceryrules_df <- as(groceryrules, "data.frame")

str(groceryrules_df)

# Visualizing rules in graphs 
library(plotly)
data(Groceries)
rules <- apriori(Groceries, parameter=list(support=0.001, confidence=0.8))
rules

# interactive scatter plot visualization
plotly_arules(rules)
plotly_arules(rules, measure = c("support", "lift"), shading = "confidence")
plotly_arules(rules, method = "two-key plot")

# add jitter, change color and markers and add a title
plotly_arules(rules, jitter = 10, opacity = .7, size = 10, symbol = 1,
              colors = c("blue", "green"))

#graph based
subrules2 <- head(sort(rules, by="lift"), 10)
plot(subrules2, method="graph")
plot(subrules2, method="graph", control=list(type="itemsets"))
plot(subrules2, method="paracoord", control=list(reorder=TRUE))
