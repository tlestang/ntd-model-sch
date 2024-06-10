# Load required packages

library(readr)
library("RColorBrewer")
library(matrixStats)

# Load required functions
get_prevs <- function(d, year, age_group, cat){
  return(d[1 + age_group*69 + (year-2018) + (cat-1)*23,])
}


get_numbers <- function(d, year, age_group){
  return(d[5521 + age_group + (year-2018)*80,])
}

get_data_over_years <- function(d, years,  age_group, cat, numAges){
  #age_groups = 0:79
  ks1 = 1 + age_group* length(years)* 4  + (years-min(years)) + (cat-1)*length(years) 
  numberStartPoint=  which(d$measure == "number")[1]
  ks2 = numberStartPoint  + age_group + (years-min(years))*numAges
  all_years_prev = d[ks1, ]
  all_years_number = d[ks2, ]
  
  return(list(all_years_prev, all_years_number))
}



get_number_infecteds <- function(ihme1, years){
  k = which(colnames(ihme1) == 'draw_0')
  age_groups = unique(ihme1$age_start)
  for(i in 1 : length(age_groups)){
    for(cat in 1:3){
      # output ihme and ipm data for given year, age group and prevalence intensity
      a = get_data_over_years(d = ihme1, years,  age_group = age_groups[i], cat = cat,  numAges = length(age_groups))
      
      # construct array to hold all the data
      if((i == 1 & cat == 1)){
        b = a[[1]][, k:ncol(ihme1)] * a[[2]][, k:ncol(ihme1)]
        num_infecteds = array(0, dim = c(dim(b)[1], dim(b)[2], length(age_groups), 3),
                              dimnames = list(rownames(b),
                                              colnames(b)))
        num_infecteds[, , 1, cat] = array(unlist(b))

      }else{
        num_infecteds[, , i, cat]= array(unlist(a[[1]][, k:ncol(ihme1)] * a[[2]][, k:ncol(ihme1)]))

      }
    }
    
    
  }
  return(num_infecteds)
}

write_file_name <- function(IU, scenario, path_header, date_name, IU_data, output){
  coverage_group <- IU_data$CoverageGroup[which(IU_data$IU_ID2==IU)]
  group <- IU_data$Group[which(IU_data$IU_ID2==IU)]
  numt = floor(log10(group)) + 1
  if(numt == 1){
    kk = paste0("00", group)
  }
  if(numt == 2){
    kk = paste0("0", group)
  }
  if(numt == 3){
    kk = group
  }
  
  if(output == "ihme"){
    file_name = paste0(path_header,"endgame-ihme-ipm-outputs-export-",date_name,"/mansoni/ihme-",IU,"-mansoni-group_",kk,"-scenario_",scenario,"_",coverage_group,"-group_", kk,"-200_simulations.csv")
  }else if(output == "ipm"){
    file_name = paste0(path_header,"endgame-ihme-ipm-outputs-export-",date_name,"/mansoni/ipm-",IU,"-mansoni-group_",kk,"-scenario_",scenario,"_",coverage_group,"-group_", kk,"-200_simulations.csv")
  }
  return(file_name)
}


read_and_get_number_infecteds <- function(ihme_name, interval = FALSE, pop_size = 3000){
  ihme1 = read.csv(ihme_name)
  prev_data =ihme1[ which(ihme1$measure == "prevalence"),]
  years = unique(prev_data$year_id)
  age_groups = unique(ihme1$age_start)
  num_infecteds = get_number_infecteds(ihme1, years)
  total_infecteds <- get_total_number_infected(num_infecteds, age_groups)/pop_size
  means <- rowMeans(total_infecteds)
  if(interval == TRUE) {
    lower_upper <- apply(total_infecteds, 1, quantile, probs = c(0.025, 0.975),  na.rm = TRUE)
    return(list(means = means, interval = lower_upper))
  }else{
    return(list(means = means))
  }
}

binom_int <- function(p_hat, n){
  z <- 1.96
  interval <- z * sqrt((p_hat*(1-p_hat))/n)
  return(c(p_hat - interval ,p_hat + interval))
}


read_and_get_number_mda_finished <- function(IU, ipm_name, IU_data, 
                                             interval = FALSE, single_IU = FALSE){
  if(file.exists(ipm_name)){
    ipm1 = read.csv(ipm_name)
    sPass = which(ipm1$measure == "surveyPass")
    k = which(colnames(ipm1) == 'draw_0')
    
    num_mda_finished = ipm1[sPass,k:ncol(ipm1)]
    num_IUs <- IU_data$num_ius[which(IU_data$IU_ID2==IU)]
    if(single_IU == TRUE){
      means <- rowMeans(num_mda_finished)
    }else{
      means <- rowMeans(num_mda_finished)*num_IUs
    }
    if(interval == TRUE & single_IU ==TRUE) {
      lower_upper <- sapply(1:dim(num_mda_finished)[1], function(x) binom_int(p_hat = sum(num_mda_finished[x,])/length(num_mda_finished[x,]), n = length(num_mda_finished[x,])))
      return(list(means = means, interval = lower_upper))
    }else{
      return(list(means = means))
    }
  }
}

get_number_infecteds_multiple_IUS <- function(scenario, IU_data, path_header, date_name){
  count1 <- 0
  count2 <- 0
  for(i in 1:nrow(IU_data)){
    IU <- IU_data$IU_ID2[i]
    ihme_name <- write_file_name(IU, scenario, path_header, date_name, IU_data,
                                 output = "ihme")
    if(file.exists(ihme_name)){
      # calculate number infected
      res <- read_and_get_number_infecteds(ihme_name)
      pop_size <- IU_data$pop[which(IU_data$IU_ID2==IU)]
      res <- res[[1]]*pop_size
      
      coverage_group <- IU_data$CoverageGroup[which(IU_data$IU_ID2==IU)]
      # then store
      if(coverage_group == 1){
        if(count1 == 0){
          a = res
          prop_inf_1 = matrix(NA, nrow(IU_data), length(a))
          count1 = 1
          prop_inf_1[i,] = res
        }else{
          prop_inf_1[i,] = res
          count1 = count1 + 1
        }
        
      }
      if(coverage_group == 2){
        if(count2 == 0){
          a = res
          prop_inf_2 = matrix(NA, nrow(IU_data), length(a))
          count2 = 1
          prop_inf_2[i,] = res
        }else{
          prop_inf_2[i,] = res
          count2 = count2 + 1
        }
        
      }
    }
    print(paste("Done", i,"of", nrow(IU_data)))
  }
  coverage_group <-  IU_data$CoverageGroup
  k1 = which(coverage_group==1)
  k2 = which(coverage_group==2)
  group1_pop = sum(IU_data$pop[k1])
  group2_pop = sum(IU_data$pop[k2])
  
  total_infs_1 = colSums(prop_inf_1[k1,]/group1_pop)
  total_infs_2 = colSums(prop_inf_2[k2,]/group2_pop)
  
  return(list(total_infs_1=total_infs_1, total_infs_2=total_infs_2))
}

get_number_mda_stopped_multiple_IUS <- function(scenario, IU_data, path_header, date_name){
  count1 <- 0
  count2 <- 0
  for(i in 1:nrow(IU_data)){
    IU <- IU_data$IU_ID2[i]
    ipm_name <- write_file_name(IU, scenario, path_header, date_name, IU_data,
                                output = "ipm")
    if(file.exists(ipm_name)){
      # calculate number infected
      res <- read_and_get_number_mda_finished(IU, ipm_name, IU_data, interval = FALSE)
      res <- res[[1]]
      
      coverage_group <- IU_data$CoverageGroup[which(IU_data$IU_ID2==IU)]
      # then store
      if(coverage_group == 1){
        if(count1 == 0){
          a = res
          num_mda_finished_1 = matrix(NA, nrow(IU_data), length(a))
          count1 = 1
          num_mda_finished_1[i,] = res
        }else{
          num_mda_finished_1[i,] = res
          count1 = count1 + 1
        }
        
      }
      if(coverage_group == 2){
        if(count2 == 0){
          a = res
          num_mda_finished_2 = matrix(NA, nrow(IU_data), length(a))
          count2 = 1
          num_mda_finished_2[i,] = res
        }else{
          num_mda_finished_2[i,] = res
          count2 = count2 + 1
        }
        
      }
    }
    print(paste("Done", i,"of", nrow(IU_data)))
  }

  coverage_group <- IU_data$CoverageGroup
  k1 = which(coverage_group==1)
  k2 = which(coverage_group==2)
  
  totalIUS_1 = sum(IU_data$num_ius[k1])
  totalIUS_2 = sum(IU_data$num_ius[k2])
  num_finished_1 = colSums(num_mda_finished_1[k1,])
  num_finished_2 = colSums(num_mda_finished_2[k2,])
  
  return(list(num_finished_1=num_finished_1,num_finished_2=num_finished_2,
              totalIUS_1 = totalIUS_1, totalIUS_2 = totalIUS_2))
}


get_data_over_years_trachoma<- function(d, years){
  k = which(colnames(ihme1) == 'draw_0')
  for(i in 1:length(years)){
    year =  years[i]
    ks1 = (1 + (year-min(years))*240) :  (60 + (year-min(years))*240)
    ks2 = (181+ (year-min(years))*240) :  (240 + (year-min(years))*240)
    all_years_prev = d[ks1, k:ncol(d)]
    all_years_number = d[ks2, k:ncol(d)]
    if(i == 1){
      mean_infs = mean(colSums(all_years_prev * all_years_number))
    }else{
      mean_infs = c(mean_infs, mean(colSums(all_years_prev * all_years_number)))
    }
  }
  return(mean_infs)
}



get_total_number_infected <- function(num_infecteds, age_groups){
  total_number_infected_age_group = array(0, dim = c(dim(num_infecteds)[1], dim(num_infecteds)[2], length(age_groups)))
  for(i in 1 :length(age_groups)){
    total_number_infected_age_group[,,i] =  num_infecteds[, , i, 1] + num_infecteds[, , i, 2] +
      num_infecteds[, , i, 3]
  }
  
  total_number_infected = total_number_infected_age_group[,,1]
  #all_people = numbers_of_people[,,1]
  for(i in 2:length(age_groups)){
    total_number_infected = total_number_infected + total_number_infected_age_group[,,i]
    # all_people = all_people + numbers_of_people[,,i]
  }
  browser(expr = any(is.na(total_number_infected)) == TRUE)
  return(total_number_infected)
}


get_total_number_infected_age_range <- function(num_infecteds, age_range){
  total_number_infected_age_group = array(0, dim = c(dim(num_infecteds)[1], dim(num_infecteds)[2], length(age_range)))
  for(j in 1 :length(age_range)){
    i = age_range[j]
    total_number_infected_age_group[,,i] =  num_infecteds[, , i, 1] + num_infecteds[, , i, 2] +
      num_infecteds[, , i, 3]
  }
  
  total_number_infected = total_number_infected_age_group[,,1]
  #all_people = numbers_of_people[,,1]
  for(i in 2:length(age_range)){
    total_number_infected = total_number_infected + total_number_infected_age_group[,,i]
    # all_people = all_people + numbers_of_people[,,i]
  }
  return(total_number_infected)
}

get_surveyPass_data<- function(d){
  x = which(d$measure == 'surveyPass')
  survey  = d[x, ]
  
  years = unique(survey$year_id)
  
  c1 = which(colnames(nums) == "draw_0")
  
  return(survey[, c1:ncol(survey)])
}


get_number_people_in_age_range <- function(d, minAge, maxAge){
  x = which(d$measure == 'number')
  nums  = d[x, ]
  y = which(nums$age_start >= minAge & nums$age_end <= maxAge)
  nums = nums[y, ]
  years = unique(nums$year_id)
  allNums = matrix(0, length(years), 200)
  c1 = which(colnames(nums) == "draw_0")
  for(i in 1:length(years)){
    j = which(nums$year_id == years[i])
    allNums[i, ] = colSums(nums[j, c1:ncol(nums)])
  }
  return(allNums)
}




