library(readr)
library("RColorBrewer")
source("analyse_simulations.R")
IU_data <- read.csv("mansoniIUs_scenario3.csv")


date_name <- "20220822a"
path_header <- "~/"

colours <- c("#0098FF", "#1b9e77", "#d95f02","#7570b3", '#d4d133')


### one IU
scenario <- "1"
IU <- "MWI28954"  

ihme_name <- write_file_name(IU, scenario, path_header, date_name, IU_data,
                             output = "ihme")
res <- read_and_get_number_infecteds(ihme_name, interval = TRUE)

years <- 2018:2040
plot(years, res$means, type = 'l', lwd = 4, 
     xlab = 'year', ylab = 'prevalence all infections',
     col = colours[1],
     bty = 'n', 
     ylim = c(0,max(res$interval)),
     cex = 1.7, cex.axis = 1.7, cex.lab = 1.7,cex.main = 1.7)
lines(years, res$interval[1,], col = colours[1], lty = 2)
lines(years, res$interval[2,], col = colours[1], lty = 2)


### loop over scenarios
IU <- "AGO02069"
years <- 2018:2040
plot(NA, xlim = c(min(years), max(years)), ylim = c(0, 0.2),
xlab = 'year', ylab = 'prevalence all infections',
cex = 1.7, cex.axis = 1.7, cex.lab = 1.7,cex.main = 1.7)

par(mfrow = c(3,2))
scenario_vec <- c("0", "1", "2", "3a", "3b")
for(i in 1:length(scenario_vec)){
  scenario <- scenario_vec[i]
  ihme_name <- write_file_name(IU, scenario, path_header, date_name, IU_data,
                               output="ihme")
  res <- read_and_get_number_infecteds(ihme_name, interval = TRUE)
  
  plot(years, res$means, type = 'l', col = colours[i], ylim = c(0, 0.2))
  lines(years, res$interval[1,], col = colours[i], lty = 2)
  lines(years, res$interval[2,], col = colours[i], lty = 2)
}

### loop over multiple IUs
IU_data <- read.csv("mansoniIUs_scenario3.csv")
IU_data <- IU_data[1:20,]


scenario_0_res <- get_number_infecteds_multiple_IUS(scenario = "0", IU_data, path_header, date_name)
total_infs0_1 <- scenario_0_res$total_infs_1
total_infs0_2 <- scenario_0_res$total_infs_2


#  get_number_infecteds_multiple_IUS() can be called for each scenario to make the plots 

scenario_1_res <- get_number_infecteds_multiple_IUS(scenario = "1", IU_data, path_header, date_name)
total_infs1_1 <- scenario_1_res$total_infs_1
total_infs1_2 <- scenario_1_res$total_infs_2

scenario_2_res <- get_number_infecteds_multiple_IUS(scenario = "2", IU_data, path_header, date_name)
total_infs2_1 <- scenario_2_res$total_infs_1
total_infs2_2 <- scenario_2_res$total_infs_2

scenario_3a_res <- get_number_infecteds_multiple_IUS(scenario = "3a", IU_data, path_header, date_name)
total_infs3a_1 <- scenario_3a_res$total_infs_1
total_infs3a_2 <- scenario_3a_res$total_infs_2

scenario_3b_res <- get_number_infecteds_multiple_IUS(scenario = "3b", IU_data, path_header, date_name)
total_infs3b_1 <- scenario_3b_res$total_infs_1
total_infs3b_2 <- scenario_3b_res$total_infs_2

# Make group 1 plot
png("Mansoni_Group1_trajectory.png", height = 8, width = 12, res = 300, units = "in")
plot(2018:2040, total_infs0_1, type = 'l', lwd = 4, 
     xlab = 'year', ylab = 'prevalence all infections',
     col = 0,
     bty = 'n', 
     ylim = c(0,max(total_infs0_1)),
     cex = 1.7, cex.axis = 1.7, cex.lab = 1.7,cex.main = 1.7,
     main = 'Mansoni Group 1')


lines(2018:2040,total_infs1_1, type = 'l', lwd = 4, col = cols[2])

lines(2018:2040,total_infs2_1, type = 'l', lwd = 4, col = cols[3])

lines(2018:2040,total_infs3a_1, type = 'l', lwd = 4, 
      col = cols[4],
      bty = 'n')
lines(2018:2040,total_infs3b_1, type = 'l', lwd = 4, 
      col = cols[5],
      bty = 'n')


legend('topright',  legend = c("Scenario 1", "Scenario 2", "Scenario 3a","Scenario 3b"),
       col = cols[-1], lwd = c(4,4,4,4),
       bty = 'n', cex = 1.7)

dev.off()



# Make group 2 plot
png("Mansoni_Group2_trajectory.png", height = 8, width = 12, res = 300, units = "in")
plot(2018:2040, total_infs0_2, type = 'l', lwd = 4, 
     xlab = 'year', ylab = 'prevalence all infections',
     col = 0,
     bty = 'n', ylim = c(0, max(total_infs0_2)),
     
     cex = 1.7, cex.axis = 1.7, cex.lab = 1.7,cex.main = 1.7,
     main = 'Mansoni Group 2')

lines(2018:2040,total_infs1_2, type = 'l', lwd = 4, col = cols[2])
lines(2018:2040,total_infs2_2, type = 'l', lwd = 4, col = cols[3])


lines(2018:2040,total_infs3a_2, type = 'l', lwd = 4, 
      col = cols[4],
      bty = 'n')
lines(2018:2040,total_infs3b_2, type = 'l', lwd = 4, 
      col = cols[5],
      bty = 'n')

legend('topright',  legend = c("Scenario 1", "Scenario 2", "Scenario 3a","Scenario 3b"),
       col = cols[-1], lwd = c(4,4,4,4),
       bty = 'n', cex = 1.7)

dev.off()

