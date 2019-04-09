library(diptest)

#' Bandwidth calculation for density estimation
#'
#' Calculates the smallest value so that the gaussian kernel density estimate of the given data \code{x} has \code{k} modes.
#' The smaller you choose the bandwidth for a kernel density estimate, the larger the number of modes becomes. This function calculates the smallest value leading to a kernel density estimate with \code{k} number of modes.
#'
#' @param x vector of data
#' @param k number of modes
#' @param prec number of digits for precision of calculation
#' @param density.fun A function that returns a vector of density estimates
#'
#' @return the smallest value so that the gaussian kernel density estimate of the given data \code{x} has \code{k} modes.
#'
#' @export
h.crit <- function(x, k, prec=6, density.fun=NULL){
  if(is.null(density.fun)) {
    density.fun <- function(x,h){density(x,bw=h,kernel ="gaussian")$y}
  }

  digits=prec
  prec=10^(-prec)
  x <- sort(x)
  minh <- min(diff(x))      #minimal possible h
  maxh <- diff(range(x))/2  #maximal possible h
  a <- maxh
  b <- minh
  zaehler=0

  while (abs(b-a)>prec){
    m <- nr.modes(density.fun(x,a))

    b <- a
    if (m > k){
      minh <- a
      a <- (a + maxh)/2
    }
    else {
      maxh <- a
      a <- (a - minh)/2
    }
  }
  a=round(a,digits)

  if(nr.modes( density.fun(x,a) ) <= k){
    #subtract until more than k modes
    while(nr.modes( density.fun(x,a) ) <= k){
      a = a - prec
    }
    a=a+prec
  }

  if(nr.modes( density.fun(x,a) ) > k){
    #add until nr. of moodes correct
    while(nr.modes( density.fun(x,a) ) > k){
      a = a + prec
    }
  }

  a
}

#' Number of modes
#'
#' Calculates the number of modes for given y-values of a density function.
#'
#' @param y vector of y-values of a density function
#'
#' @export
nr.modes <- function(y) {
  d1 <- diff(y)
    signs <- diff(d1/abs(d1))
    length(signs[signs==-2])
}

#' Silvermantest
#'
#' The silvermantest tests the null hypothesis that an underlying density has at most \code{k} modes.
#'
#' @param x vector of data
#' @param k number of modes for the null hypothesis
#' @param R number of bootstrap replications
#' @param adjust boolean to activate the adjusting of the p-value (valid if k=1) (see Hall and York)
#' @param digits number of digits of the p-value
#' @param density.fun A function that returns a vector of density estimates
#'
#' @return An object of the class Silvermantest (see: \code{\link{Silvermantest-class}}).
#' @export
silverman.test <- function(x, k, R=999, adjust=FALSE, digits=6, density.fun=NULL){
  # x: data
  # k: number of modes to be tested
  # M: number of bootstrap replications

  #check if seed is available (as done in boot package)
  #if so save it
  seedAvailable = exists(x=".Random.seed",envir=.GlobalEnv,inherits=FALSE)
  if(seedAvailable)
    saved_seed = .Random.seed
  else{
    rnorm(1)
    saved_seed = .Random.seed
  }

  #temp function for bootstrapping
  y.obs <- function(x,h,sig=sd(x)){
    mean(x) + (x-mean(x)+h*rnorm(length(x),0,1))/((1+h^2/sig^2)^(1/2))
    #(x+h*rnorm(length(x),0,1))/((1+h^2/sig^2)^(1/2))
  }

  #temp function for density calculation
  if(is.null(density.fun)) {
    density.fun <- function(x,h){density(x,bw=h,kernel ="gaussian")$y}
  }

  #start of the test
  h0 <- h.crit(x, k, density.fun=density.fun)

  # statistic function
  mode.fun <- function(d, i, h0) {
    x.boot <- sort(y.obs(d[i], h0))
    nr.modes(density.fun(x.boot, h0))
  }
  mod.boot <- boot::boot(x, statistic = mode.fun, R = R, h0 = h0)

  n <- sum(as.vector(mod.boot$t) > k)
  p <- n/R
  
  if (adjust) {
    if (k==1) {
      #asymptotic levels of silvermantest by Hall/York
      x=c(0,0.005,0.010,0.020,0.030,0.040,0.050,0.06,0.07,0.08,0.09,0.1,0.11,0.12,0.13,0.14,0.15,0.16,0.17,0.18,0.19,0.2,0.25,0.30,0.35,0.40,0.50)
      y=c(0,0,0,0.002,0.004,0.006,0.010,0.012,0.016,0.021,0.025,0.032,0.038,0.043,0.050,0.057,0.062,0.07,0.079,0.088,0.094,0.102,0.149,0.202,0.252,0.308,0.423)
      sp = splines::interpSpline(x,y)
      #adjusting the p-value
      if (p<0.005)
        p=0
      else{
        p = predict(sp,p)$y
        p = round(p,digits)
      }
    } else{
      print("The option to adjust the p-value is valid only for k=1")
    }
  }

  #return(list(saved_seed=saved_seed,p_value=p))
  test_obj = new("Silvermantest", data=x, p_value = p, saved_seed=saved_seed, k=k)
  return(test_obj)
}

#OOP mit S4

#definiere Klasse
methods::setClass("Silvermantest",representation = representation(data="numeric",p_value="numeric",saved_seed="numeric",k="numeric"))

#definiere show, bzw. printfunktion um f?r Objekte dieser Klasse
methods::setMethod("show",signature(object="Silvermantest"),
  function(object){
    cat("Silvermantest: Testing the hypothesis if the number of modes is <= ", object@k,"\n")
    cat("The P-Value is ",object@p_value,"\n")
  }
)

# arguments
args=commandArgs(TRUE)
out_dir = args[1]
ks_dir = args[2]  # a directory of files for remained variants after KS test
cell_line = args[3]
depth = args[4]

# param settings
purity_tags = c('n2t97', 'n5t95', 'n2t97', 'n10t90', 'n20t80', 'n30t70', 'n40t60', 'n50t50', 'n60t40', 'n70t30', 'n80t20', 'n90t10', 'n95t5')
dip_out_header = sprintf("%s\t%s\t%s\t%s\t%s", "cell_line", "depth", "contam", "pval", "vaf_cutoff")
silverman_out_header = sprintf("%s\t%s\t%s\t%s", "cell_line", "depth", "contam", "pval")

# path settings
diptest_out_file_path = sprintf("%s/diptest_%s_%s.tsv", out_dir, cell_line, depth)
silverman_out_file_path = sprintf("%s/silverman_%s_%s.tsv", out_dir, cell_line, depth)

diptest_out_lines = c(dip_out_header)
silverman_out_lines = c(silverman_out_header)

for (purity_tag in purity_tags) {
  # in-loop path settings
  var_tsv_path = sprintf("%s/%s.%s.%s.tsv", ks_dir, cell_line, purity_tag, depth)
  message(var_tsv_path)
  if (file.exists(var_tsv_path)) {
    vaf_list = read.table(file=var_tsv_path, header=TRUE)[,6]

    dip_result = dip.test(vaf_list)
    dip_statistic = dip(vaf_list, full.result=TRUE)
    diptest_out_line = sprintf("%s\t%s\t%s\t%s\t%s", cell_line, depth, purity_tag, dip_result$p.val, dip_statistic$xu)
    diptest_out_lines = c(diptest_out_lines, diptest_out_line)

    result = silverman.test(vaf_list, 1)
    silverman_out_line = sprintf("%s\t%s\t%s\t%s", cell_line, depth, purity_tag, result@p_value)
    silverman_out_lines = c(silverman_out_lines, silverman_out_line)
  }
}

# write the results
diptest_out_file = file(diptest_out_file_path)
silverman_out_file = file(silverman_out_file_path)

writeLines(diptest_out_lines, diptest_out_file)
writeLines(silverman_out_lines, silverman_out_file)

close(diptest_out_file)
close(silverman_out_file)
