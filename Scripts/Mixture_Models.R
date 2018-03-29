
library(mixtools)
df=read.csv("E:\\Dhaval\\Data Science\\Kaggle\\Zillow\\Input\\train_2016_v2.csv")
head(df,5)
log_error_df=df$logerror
mixmdl=normalmixEM(log_error_df)
plot(mixmdl,which=2)
