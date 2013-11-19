################################################################################
################       STA  250  HW2  CODE (Olivia Lee)      ###################
################################################################################

################################################################################
##############################   Problem  1  ###################################
##############################    USE R      ###################################
################################################################################

mini <- FALSE

#============================== Setup for running on Gauss... ==============================#

args <- commandArgs(TRUE)

cat("Command-line arguments:\n")
print(args)

####
# sim_start ==> Lowest possible dataset number
###

###################
sim_start <- 1000
###################

if (length(args)==0){
  sim_num <- sim_start + 1
  set.seed(121231)
} else {
  # SLURM can use either 0- or 1-indexing...
  # Lets use 1-indexing here...
  sim_num <- sim_start + as.numeric(args[1])
  sim_seed <- (762*(sim_num-1) + 121231)
}

cat(paste("\nAnalyzing dataset number ",sim_num,"...\n\n",sep=""))

# Find r and s indices:
s_index=ceiling((sim_num-1000)/50)    
		#sim_num = 1001:1250 --> sim_num-1000 = 1:250
r_index=sim_num%%50            
		# returns a result of of remainder of sim_num/50
		# for 1:250%%50 returns 1,2,...,49,0,1,2,...,49,0,....
		# what I exptec for r_index is 1,2,...,49,50,1,2,....,49,50,1,2,.....
r_index[which(r_index==0)]=50   
		# Thus I change those value of 0 into 50
#============================== Run the simulation study ==============================#

# Load packages:
library(BH)
library(bigmemory.sri)
library(bigmemory)
library(biganalytics)

# I/O specifications:
datapath <- "/home/pdbaines/data"
outpath <- "/home/oylee/Stuff/HW2/BLB/output/"

# mini or full?
if (mini){
	rootfilename <- "blb_lin_reg_mini"
} else {
	rootfilename <- "blb_lin_reg_data"
}

# Filenames:
file = paste(rootfilename,".desc",sep="")    

# Set up I/O stuff:
filename = file.path(datapath,file)
# Attach big.matrix :
temp=attach.big.matrix(filename)
# Remaining BLB specs:
n=length(temp[,1])
b=floor(n^0.7)
d=length(temp[1,])
# Extract the subset:
set.seed(s_index)
newtemp=temp[sample(1:n,b,replace=FALSE),]
# Reset simulation seed:
set.seed(sim_seed)
# Bootstrap dataset:
test=rmultinom(1, size = n, prob=rep(1,b)/b)
# Fit lm:
model=lm(newtemp[,d]~newtemp[,1:(d-1)]-1,data=data.frame(newtemp),weights=test)
beta=model$coefficients
# Output file:
outfile = file.path(outpath,paste0("coef_",sprintf("%02d",s_index),"_",sprintf("%02d",r_index),".txt"))
# Save estimates to file:
write.table(beta,file=outfile,row.name=FALSE)

##########################################################################################
###################################    Problem  2     ####################################
###################################    USE python     ####################################
##########################################################################################

######################
#### Map Function ####
######################

#!/usr/bin/env python
# From: http://www.michael-noll.com/tutorials/writing-an-hadoop-mapreduce-program-in-python/

#import math
import sys
from math import floor, ceil
for line in sys.stdin:
    numbers = map(float, line.split())       #split each element by " ", and change into float
    x = numbers[0] * 10    # since ceil is integer-base, so I multiply x-cor by 10.
    y = numbers[1] * 10    # do the same thing to y-cor
    word = str(ceil(x) / 10) + '_' + str(ceil(y) / 10)   #divided the ceiling result by 10 and use
    							 # "_" to connect x_upper bound and y_upper bound
    print '%s\t%s' % (word, 1)

###########################
##### Reduce Function #####
###########################

#!/usr/bin/env python

from operator import itemgetter
import sys

current_word = None
current_count = 0
word = None

for line in sys.stdin:
    line = line.strip()       #remove leadibg abd trailing whitespace
    chopped = line.split()    #separate the key and the value
    word = chopped[0]         #key
    count = chopped[1]        #value
    try:
        count = int(count)
    except ValueError:
                # count was not a number, so silently
                # ignore/discard this line
        continue
    if current_word == word:
        current_count += count
    else:
        if current_word:
            bound = str.partition(current_word, "_")     ## String split by "_"
            x_up = str(bound[0])
            y_up = str(bound[2])
            x_low = str(float(x_up) - 0.1)       #find the x_lower bound
            y_low = str(float(y_up) - 0.1)       #find the y_lower bound
            current_count = str(current_count)
            print '%s\t%s\t%s\t%s\t%s' % (x_low, x_up, y_low, y_up, current_count)   #paste the result
        current_count = count
        current_word = word
if current_word == word:
    bound = str.partition(current_word, "_")
    x_up = str(bound[0])
    y_up = str(bound[2])
    x_low = str(float(x_up) - 0.1)
    y_low = str(float(y_up) - 0.1)
    current_count = str(current_count)
    print '%s\t%s\t%s\t%s\t%s' % (x_low, x_up, y_low, y_up, current_count)


##########################################################################################
###################################    Problem  3     ####################################
###################################     ON HIVE       ####################################
##########################################################################################

# Create the table 
CREATE EXTERNAL TABLE new_table(group INT, value FLOAT) ROW FORMAT DELIMITED FIELDS TERMINATED BY '\t STORED AS TEXTFILE;

# Load in Data
LOAD DATA INPATH '/user/hadoop/data/groups.txt' OVERWRITE INTO TABLE new_table;

#Write out the within-group mean results
insert overwrite local directory '/home/hadoop/test/mean' SELECT AVG(value) FROM new_table GROUP BY group;

#Write out the within-group variance results
insert overwrite local directory '/home/hadoop/test/var' SELECT VARIANCE(value) FROM new_table GROUP BY group;
