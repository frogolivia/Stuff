###################################################
##############      Problem 1         #############
###################################################

##############      CODE for kernel     #############

#include <stdio.h>
#include <stdlib.h>

#include <cuda.h>
#include <curand_kernel.h>
#include <math_constants.h>

extern "C"
{

__global__ void
rtruncnorm_kernel(
                  float *x,
	      int n,
                  float *mu, float *sigma,
                  int maxtries,
	      float *a,
	      float *b,
	      int rng_a,
	      int rng_b,
	      int rng_c)
{  
    int accept, tries;
    float temp_x,
	bound, alpha, z, prob, u, trunc, righttail;  //variable for rejection sampling
	
    // Usual block/thread indexing...
    int myblock = blockIdx.x + blockIdx.y * gridDim.x;
    int blocksize = blockDim.x * blockDim.y * blockDim.z;
    int subthread = threadIdx.z*(blockDim.x * blockDim.y) + threadIdx.y*blockDim.x + threadIdx.x;
    int idx = myblock * blocksize + subthread;

    // Setup the RNG:
    curandState rng;
    curand_init(rng_a + idx* rng_b, rng_c, 0, &rng);
    if (idx < n ){
	tries = 0;
	accept = 0;
	temp_x= 0;
	while (accept == 0 && tries<maxtries){
	 temp_x= mu[idx] + sigma[idx] * curand_normal(&rng);
	 tries = tries + 1;
	 if (temp_x>a[idx] && temp_x<b[idx]){
	   accept = 1;
	   x[idx]=temp_x;
	   }//end of if	
	}//end of while
	while (accept ==0){ // not updated at all--> run rejection sampling
   if (abs(*a-*mu)>abs(*b-*mu)){
	trunc = *b;
     } //end of if
   else { trunc = *a;
      }//end of else
   if (trunc>*mu){  //right tail
	bound = (trunc-*mu)/ *sigma;
	righttail=1;
     } // end of if
    else{//left tail
	bound = abs((*mu-trunc) / *sigma);
	righttail=0;
    } //end of else
   alpha = (bound+sqrt(pow(bound,2)+4))/2;
   z = bound + (-log(curand_uniform(&rng))/alpha);
	if (bound<alpha){
		prob = exp(-pow((alpha-z),2)/2) ;
	 } //end of if
	else {
		prob = exp(pow((bound-alpha),2)/2-pow((alpha-z),2)/2);
	 }//end of else
	u = curand_uniform(&rng);
	if (u<prob){
	    if (righttail ==1){  //right tail
		accept = 1;
	   	x[idx] = *mu+*sigma * z;
	     } //end of if
	    else{   //left tail
	   	accept = 1;
	   	x[idx] = *mu+ *sigma * (-z);
	    }//end of else
	 } //end of if

}//end of rejection sampling
}//end of if
return;
}//end of function
} // END extern "C"

##############  code ( run  in R )  #############
library(RCUDA)
cuGetContext(TRUE)

"compute_grid" <- function(N,sqrt_threads_per_block=16L,grid_nd=1)
{
    # if...
    # N = 1,000,000
    # => 1954 blocks of 512 threads will suffice
    # => (62 x 32) grid, (512 x 1 x 1) blocks
    # Fix block dims:
    block_dims <- c(as.integer(sqrt_threads_per_block), as.integer(sqrt_threads_per_block), 1L)
    threads_per_block <- prod(block_dims)
    if (grid_nd==1){
      grid_d1 <- as.integer(max(1L,ceiling(N/threads_per_block)))
      grid_d2 <- 1L
    } else {
      grid_d1 <- as.integer(max(1L, floor(sqrt(N/threads_per_block))))
      grid_d2 <- as.integer(ceiling(N/(grid_d1*threads_per_block)))
    }
    grid_dims <- c(grid_d1, grid_d2, 1L)
    return(list("grid_dims"=grid_dims,"block_dims"=block_dims))
}

n = 10000L

grid = compute_grid(n,sqrt_threads_per_block=16L,grid_nd=1)
m = loadModule("rtruncnorm.ptx")
my_kernel <- m$rtruncnorm_kernel
grid_dims <- grid$grid_dims
block_dims <- grid$block_dims
x = matrix(0,n,1)
mu = matrix(2,n,1)
sigma = matrix(1,n,1)
a=matrix(0,n,1)
b=matrix(1.5,n,1)
rng_a = 33L
rng_b=45L
rng_c=12L
maxtries = 2000L

sample=.cuda(my_kernel,"X"=x,n,mu,sigma,maxtries,a,b,rng_a,rng_b,rng_c,gridDim=grid_dims,blockDim=block_dims,outputs="X")

############  Time code   ###########
timetable=matrix(0,2,8)
for (i in 1:8){
   n=10^i
   grid = compute_grid(n,sqrt_threads_per_block=16L,grid_nd=1)
   grid_dims <- grid$grid_dims
   block_dims <- grid$block_dims

CPU= system.time(rtnorm(n,mean=2, sd=1, 0,1.5))
GPU = system.time({.cuda(my_kernel,"X"=x,n,mu,sigma,maxtries,a,b,rng_a,rng_b,rng_c,gridDim=grid_dims,blockDim=block_dims,outputs="X")})
timetable[1,i] = CPU[3]
timetable[2,i] = GPU[3]
}




####  f.  #### 
#set a = -1000 ~ -Inf
a=matrix(-1000,n,1)
b=matrix(-200,n,1)
sample1=.cuda(my_kernel,"X"=x,n,mu,sigma,maxtries,a,b,rng_a,rng_b,rng_c,gridDim=grid_dims,blockDim=block_dims,outputs="X")

#set b= 1000 ~ Inf   (right tail)
a=matrix(900,n,1)
b=matrix(1000,n,1)
sample2=.cuda(my_kernel,"X"=x,n,mu,sigma,maxtries,a,b,rng_a,rng_b,rng_c,gridDim=grid_dims,blockDim=block_dims,outputs="X")

####  g. #####
a=matrix(-2000,n,1)
b=matrix(-3,n,1)
mu = matrix(0,n,1)
sigma = matrix(1,n,1)
sample_f=.cuda(my_kernel,"X"=x,n,mu,sigma,maxtries,a,b,rng_a,rng_b,rng_c,gridDim=grid_dims,blockDim=block_dims,outputs="X")
### mean(sample_f) = -3.281287

###################################################
##############      Problem 2         #############
###################################################

library(RCUDA)
#install.packages("msm")
library(msm)
library(MASS)
library(compiler)


cuGetContext(TRUE)

"compute_grid" <- function(N,sqrt_threads_per_block=16L,grid_nd=1)
{
    # if...
    # N = 1,000,000
    # => 1954 blocks of 512 threads will suffice
    # => (62 x 32) grid, (512 x 1 x 1) blocks
    # Fix block dims:
    block_dims <- c(as.integer(sqrt_threads_per_block), as.integer(sqrt_threads_per_block), 1L)
    threads_per_block <- prod(block_dims)
    if (grid_nd==1){
      grid_d1 <- as.integer(max(1L,ceiling(N/threads_per_block)))
      grid_d2 <- 1L
    } else {
      grid_d1 <- as.integer(max(1L, floor(sqrt(N/threads_per_block))))
      grid_d2 <- as.integer(ceiling(N/(grid_d1*threads_per_block)))
    }
    grid_dims <- c(grid_d1, grid_d2, 1L)
    return(list("grid_dims"=grid_dims,"block_dims"=block_dims))
}

gen_z=function(beta,y,x){
	z=matrix(0,length(y),1)
	x=as.matrix(x)
	beta=as.matrix(beta)
	bound=matrix(0,length(y),2)
	bound[which(y==0),1]=-Inf
	bound[which(y==1),2]= Inf
      z = rtnorm(matrix(1,length(y),1),mean=x%*%beta, sd=1, lower=bound[,1], upper=bound[,2])
      #z[which(y==1)] = rtnorm(1,mean=x[i,]%*%beta, sd=1, lower=0, upper=Inf)
	return(z)
} #end of gen_z function
gen_z = cmpfun(gen_z)


probit_mcmc=function(y, X, beta_0, sigma_0_int, niter, burnin, GPU=FALSE){
	X= as.matrix(X)
	y= as.matrix(y)
	beta_sample=matrix(0,burnin+niter+1,length(beta_0))
	beta_sample[1,] = beta_0
	B = solve(sigma_0_int+t(X)%*%X)
	for (i in 2:(burnin+niter+1)){
		if (GPU){   # use GPU
			n = as.integer(dim(y)[1])
			a = matrix( 0, length(y), 1)
			b = matrix( 0, length(y), 1)
			a[which(y==0)] = -Inf  
			b[which(y==1)] = Inf
			maxtries = as.integer(2000)
			grid = compute_grid(n,sqrt_threads_per_block=16L,grid_nd=1) 
			m = loadModule("rtruncnorm.ptx")
			my_kernel <- m$rtruncnorm_kernel
			grid_dims <- grid$grid_dims
			block_dims <- grid$block_dims
			x = matrix(0,n,1)
			mu = X%*%beta_sample[(i-1),]
			sigma = matrix(1,n,1)
			rng_a = 33L
			rng_b= 45L
			rng_c=as.integer(i)
			z_sample=.cuda(my_kernel,"X"=x,n,mu,sigma,maxtries,a,b,rng_a,rng_b,rng_c,gridDim=grid_dims,blockDim=block_dims,outputs="X")
		}
		else {   # use CPU
			z_sample=gen_z(beta_sample[(i-1),],y,X)
		}
		beta_sample[i,]=mvrnorm(n=1,B%*%(sigma_0_int%*%beta_0+t(X)%*%z_sample),B)
	}
	return(beta_sample[((burnin+1):(niter+burnin+1)),])
}#end of mcmc function
probit_mcmc=cmpfun(probit_mcmc)

pri_beta=matrix(0,8,1)
pri_sig=matrix(0,8,8)

### c. varify code by mini_data also check the performance of probit_mcmc()  #####
library(MCMCpack)
mini = read.table("mini_data.txt", header=TRUE)
cpu = probit_mcmc(mini[,1], mini[,2:9],pri_beta, pri_sig, niter=2000, burnin=500,GPU=FALSE)
plot(mcmc(cpu))
gpu = probit_mcmc(mini[,1], mini[,2:9],pri_beta, pri_sig, niter=2000, burnin=500,GPU=TRUE)
plot(mcmc(Gpu))
## run GLM
model = glm(y~.-1,data=mini,family=binomial(link=probit))
est= model$coeffients     # grab the estimators of GLM

####### d. analyze data_01-data_05

Data1=read.table("data_01.txt", header=TRUE)
Data2=read.table("data_02.txt", header=TRUE)
Data3=read.table("data_03.txt", header=TRUE)
Data4=read.table("data_04.txt", header=TRUE)
Data5=read.table("data_05.txt", header=TRUE)

cputable=matrix(0,1,5)
for (i in 1:5){
	t = eval(parse(text=paste("Data",i,sep="")))
	cputable[1,i] = system.time(probit_mcmc(t[,1], t[,2:9],pri_beta, pri_sig, niter=2000, burnin=500,GPU=FALSE))[3]
	}
write.table(cputable,file="cpu_result.txt")

gputable=matrix(0,1,5)
for (i in 1:5){
	t = eval(parse(text=paste("Data",i,sep="")))
	gputable[1,i] = system.time(probit_mcmc(t[,1], t[,2:9],pri_beta, pri_sig, niter=2000, burnin=500,GPU=TRUE))[3]
	}
write.table(gputable,file="gpu_result.txt")
