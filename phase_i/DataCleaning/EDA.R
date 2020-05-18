
#### EDA
### 1. Univariable study
summary(myData$zestimate)
ggplot(myData, aes(x=zestimate)) + 
  geom_histogram(colour="darkblue", fill="lightblue") +
  scale_x_continuous(limits = c(0, 1e7))
labs(y = "Count", x = "Zestimate ($)") +
  theme_bw()
print(skewness(myData$zestimate))
print(kurtosis(myData$zestimate))
# Comment: postive skewed, transformed to log scale

print(skewness(log(myData$zestimate)))
ggplot(myData, aes(x=zestimate)) + 
  geom_histogram(colour="darkblue", fill="lightblue") +
  scale_x_log10(breaks = trans_breaks("log10", function(x) 10^x),
                labels = trans_format("log10", math_format(10^.x))) +
  labs(y = "Count", x = "Zestimate ($)") +
  theme_bw()
ggsave(paste(output_dir, "zestimateHistogram.png", sep = "/"))


### 2. Bivariable study
# boxplot of school ratings
tmp <- gather(myData, school_type, rating, elementrary_rating:high_school_rating, factor_key=TRUE)
ggplot(tmp, aes(x=school_type, y=rating, color = school_type)) + 
  geom_boxplot() +
  theme_bw() +
  scale_x_discrete(name = "School type", labels=c("elementrary_rating" = "Elementary", "middle_school_rating" = "Middle school", 
                                                  "high_school_rating" = "High school")) +
  scale_y_continuous(name = "Rating", breaks=seq(0,10,2)) +
  theme(legend.position = "none") + 
  coord_fixed(ratio=0.15)
ggsave(paste(output_dir, "schoolRatings.png", sep = "/"))

# Convert the variable school rating from a numeric to a factor variable
myData$elementrary_rating <- as.factor(myData$elementrary_rating)
myData$middle_school_rating <- as.factor(myData$middle_school_rating)
myData$high_school_rating <- as.factor(myData$high_school_rating)
ggplot(myData[!is.na(myData$elementrary_rating),], aes(x=elementrary_rating, y=zestimate, color=elementrary_rating))+
  geom_boxplot()+
  theme_bw()
ggplot(myData[!is.na(myData$middle_school_rating),], aes(x=middle_school_rating, y=zestimate, color=middle_school_rating))+
  geom_boxplot()+
  theme_bw()
ggplot(myData[!is.na(myData$high_school_rating),], aes(x=high_school_rating, y=zestimate, , color=high_school_rating))+
  geom_boxplot()+
  theme_bw()

# boxplot of type
ggplot(myData[!is.na(myData$type),], aes(x=type, y=zestimate, color = type)) + 
  geom_boxplot() +
  theme_bw() +
  scale_x_discrete(name = "House type") +
  scale_y_continuous(name = "Zestimate ($)", limits = c(0, 3e6)) +
  theme(legend.position = "none") +
  coord_fixed(ratio=0.0000007)
ggsave(paste(output_dir, "type.png", sep = "/"))

# boxplot of heating
ggplot(myData[!is.na(myData$heating),], aes(x=heating, y=zestimate, color = heating)) + 
  geom_boxplot() +
  theme_bw() +
  scale_x_discrete(name = "Heating type") +
  scale_y_continuous(name = "Zestimate ($)", limits = c(0, 3e6)) +
  theme(legend.position = "none") +
  coord_fixed(ratio=0.0000007)
ggsave(paste(output_dir, "heating.png", sep = "/"))

# boxplot of cooling
ggplot(myData[!is.na(myData$cooling),], aes(x=cooling, y=zestimate, color = cooling)) + 
  geom_boxplot() +
  theme_bw() +
  scale_x_discrete(name = "Cooling type") +
  scale_y_continuous(name = "Zestimate ($)", limits = c(0, 3e6)) +
  theme(legend.position = "none") +
  coord_fixed(ratio=0.0000003)
ggsave(paste(output_dir, "cooling.png", sep = "/"))


# boxplot of parking
ggplot(myData[!is.na(myData$parking),], aes(x=parking, y=zestimate, color = parking)) + 
  geom_boxplot() +
  theme_bw() +
  scale_x_discrete(name = "Parking") +
  scale_y_continuous(name = "Zestimate ($)", limits = c(0, 3e6)) +
  theme(legend.position = "none") +
  coord_fixed(ratio=0.0000009)
ggsave(paste(output_dir, "parking.png", sep = "/"))


# zestimate scatter plot along w/ year_built
myData %>% 
  group_by(year_built) %>% 
  summarize(mean_zestimate = mean(zestimate),n()) %>% 
  ggplot(aes(x=year_built,y=mean_zestimate)) +
  geom_smooth(color="grey40") +
  geom_point(color="red") +
  coord_cartesian(ylim=c(0,5e6)) +
  theme_bw() + 
  scale_x_continuous(name = "Year built", breaks=seq(1865,2020,15)) +
  scale_y_continuous(name = "Zestimate distribution ($)", limits = c(0, 5e6)) +
  theme(axis.text.x = element_text(angle=45, vjust = 0.25)) +
  coord_fixed(ratio=0.00001)
ggsave(paste(output_dir, "zestimateChange.png", sep = "/"))

# zestimate scatter plot along w/ area_sqft
myData %>% ggplot(aes(x=area_sqft, y=zestimate)) +
  geom_smooth(color="red") +
  theme_bw() +
  scale_x_log10(name = "Total area (sqft)") +
  scale_y_log10(name = "Zestimate ($)") + 
  coord_fixed(ratio=0.3)
ggsave(paste(output_dir, "zestimateChange2.png", sep = "/"))  


# zestimate scatter plot along w/ annual_tax_amount
myData %>% ggplot(aes(x=annual_tax_amount, y=zestimate)) +
  geom_smooth(color="red") +
  theme_bw() +
  scale_x_log10(name = "Annual tax amount ($)") +
  scale_y_log10(name = "Zestimate ($)") + 
  coord_fixed(ratio=0.45)
ggsave(paste(output_dir, "zestimateChange3.png", sep = "/"))  

# zestimate scatter plot along w/ bedroom
myData %>% ggplot(aes(x=bedroom, y=zestimate)) +
  geom_smooth(color="red") +
  theme_bw() +
  scale_x_log10(name = "Number of bedrooms") +
  scale_y_log10(name = "Zestimate ($)") + 
  coord_fixed(ratio=0.4)
ggsave(paste(output_dir, "zestimateChange4.png", sep = "/"))  


# zestimate scatter plot along w/ bathroom
myData %>% ggplot(aes(x=bathroom, y=zestimate)) +
  geom_smooth(color="red") +
  theme_bw() +
  scale_x_log10(name = "Number of bathrooms") +
  scale_y_log10(name = "Zestimate ($)") + 
  coord_fixed(ratio=0.3)
ggsave(paste(output_dir, "zestimateChange5.png", sep = "/")) 

### Conclusion:
#1. there is correlation between middle_school_rating with zestimate and high_school_rating with zestimate
#2. year buit around 1950-1990 is more centered as before and after that time period is much more spread out. 
#3. the bigger the size, the more expensive the price
#4. central cooling correlated with higher price
#5. zestimate increases with bedroom mumber 
#6. zestimate increases with bathroom mumber 

# I wil only focus on the continuous variables, but there are many other that we should analyze.
# The trick here seems to be the choice of the right features (feature selection) and not the definition of complex relationships between them (feature engineering).


## Correlation matrix
numeric = na.omit(select_if(myData, is.numeric))
numeric.cor = cor(numeric)
numeric.rcorr = rcorr(as.matrix(numeric))
numeric.rcorr$P
corrplot(numeric.cor)
# Comment: none of the correlation was significant