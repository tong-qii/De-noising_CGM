# Wavelet transform
install.packages(c("tidyverse", "dplyr", "refund", "devtools", "gridExtra")) 
devtools::install_github("tidyfun/tidyfun") 
devtools::install_github("tidyfun/refundr")
devtools::install_github("irinagain/iglu", build_vignettes = TRUE)

library(iglu)
library(ggplot2)
library(dplyr)
library(refund)
library(gridExtra)




dataset <- "Hall2018"
filename <- "./data/pbio.2005143.s010"
curr = read.table(filename, header = TRUE, sep = "\t")

# Reorder and trim the columns to follow format
curr = curr[c(3,1,2)] 

# Renaming the columns the standard format names
colnames(curr) = c("id","time","gl")
# Ensure glucose values are recorded as numeric
curr$gl = as.numeric(as.character(curr$gl))

# Reformat the time to standard
curr$"time" = as.POSIXct(curr$time, format="%Y-%m-%d %H:%M:%S") 

#write.csv(curr, file = paste("./data/",dataset, "_processed.csv", sep = ""), 
#            row.names=F, col.names = !file.exists(paste("./data/",dataset, "_processed.csv", sep = "")),
#            append = T, sep = ",")
curr %>% ggplot(aes(x=time, y=gl, group=id, color=id)) +
  geom_point()


ids <- unique(curr$id)
ids
times <- unique(curr$time)

test = curr[which(curr$id=='2133-003'),]
test %>% ggplot(aes(x=time, y =gl)) +
  geom_point()


example_data_hall <- curr




####---- Tsalikian2005 dataset----####
# Here we have named the created folder by first author last name and date of the original paper
dataset <- "Tsalikian2005"

# We will set the working directory here in the created folder
#setwd(dataset)


file.path <- "./data/tblDDataCGMS.csv"
# Alternatively, if the file structure has been changed, simply place the tblDDataCGMS.csv file into the created folder
# Then run the file path as follows:
# file.path <- "tblDDataCGMS.csv"

# Read the table in
curr = read.csv(file.path)

# Standardize date and time
# Convert the 12-hour time to 24
curr$ReadingTm = strftime(strptime(curr$ReadingTm, "%I:%M %p"), format = "%H:%M:%S", tz="") 
# Remove the time information from the date
curr$ReadingDt = strftime(curr$ReadingDt, format = "%Y-%m-%d", tz="") 
# Combine the two into a standard formatted time object
curr$time = as.POSIXct(paste(curr$ReadingDt, curr$ReadingTm), format="%Y-%m-%d %H:%M:%S") #combine the date and time into one column

# Reorder and keep only the columns we want
curr = curr[c(2,7,6)]

# Renaming the columns the standard format names
colnames(curr) = c("id","time","gl")

#Ensure glucose values are recorded as numeric
curr$gl = as.numeric(curr$gl)

# Save the cleaned data to the created dataset folder
# The cleaned file will be named "dataset"_processed.csv
# write.table(curr, file = paste("./data/",dataset, "_processed.csv", sep = ""),
#             row.names=F, col.names = !file.exists(paste("./data/",dataset, "_processed.csv", sep = "")),
#             append = T, sep = ",")



####----Andersin2016----####
dataset <- "Anderson2016"
file.path <- "./data/CGM.txt"
# Read the raw data in 
curr = read.table(file.path, header = TRUE, sep = "|")

# Reorder and keep only the columns we want
# See below for an important note on how which time to keep was chosen
curr = curr[c(1,5,4)]

# Renaming the columns with the standard format names
colnames(curr) = c("id","time","gl")

# Ensure glucose values are recorded as numeric
curr$gl = as.numeric(curr$gl)

# Standardize date and time
curr$"time" = as.POSIXct(curr$"time", format="%Y-%m-%d %H:%M:%S")

# Save the cleaned data to the created dataset folder
# The cleaned file will be named "dataset"_processed.csv
# write.table(curr, file = paste("./data/",dataset, "_processed.csv", sep = ""), 
#             row.names=F, col.names = !file.exists(paste("./data/",dataset, "_processed.csv", sep = "")),
#             append = T, sep = ",")


#### ---- Buckingham2007----####
dataset <- "Buckingham2007"
file.path <- "./data/tblFNavGlucose.csv"
curr = read.csv(file.path, header = TRUE, stringsAsFactors = FALSE)

# combine date and time into standard format (POSIX1t format)
curr$time = strptime(paste(as.Date(curr$NavReadDt), curr$NavReadTm),
                     format = "%Y-%m-%d %H:%M:%S")

# reorder and select only id, time, gl columns
curr = curr[, c(2,6,5)]

# Renaming the columns with the standard format names
colnames(curr) = c("id","time","gl")

#Ensure glucose values are recorded as numeric
curr$gl = as.numeric(curr$gl)

# Change all values less than 32 mg/dL and above 450 mg/dL to NA
curr$gl[curr$gl <= 32] <- NA
curr$gl[curr$gl > 450] <- NA

# This dataset has some NA values for glucose readings, 
# If you would like to filter them out, simply uncomment this code:
# curr = na.omit(curr)

# The following function is used to remove regions of zero variability
# These regions likely arise from cgm sensor errors.

zero.remove = function(tab){
  tab %>%
    mutate(diff = gl - lag(gl)) %>% 
    filter(diff != 0, diff != lag(diff), diff != lag(diff, 2)) %>%
    select(-diff)
}

curr = curr %>% group_split(id) %>% map_dfr(zero.remove)

# Save the cleaned data to the created dataset folder
# The cleaned file will be named "dataset"_processed.csv
write.table(curr, file = paste('./data/',dataset, "_processed.csv", sep = ""), row.names = F, 
            col.names = !file.exists(paste('./data/',dataset, "_processed.csv", sep = "")), 
            sep = ",")







                         