schisto_coverage <- read.csv("schisto_coverage.csv")

## check age bins
# the age bins need to be at least 1 year apart
# check if any age_min is equal to age_max
which_rows <- which(schisto_coverage$age_min==(schisto_coverage$age_max))
# overwrite the age_max with age_max+1
schisto_coverage[which_rows, "age_max"] <- schisto_coverage[which_rows,"age_max"] + 1

# check there are no rows with age_min=age_max
length(which(schisto_coverage$age_min==(schisto_coverage$age_max)))

## reascale coverage to between 0 and 1
# which columns have 20 in the column name (indicative of year colunmn)
which_col <- grepl(20, colnames(schisto_coverage))
schisto_coverage[, which_col] <- schisto_coverage[, which_col]/100

## add intervention type column
# model is expecting intervention type column
schisto_coverage$"Intervention Type" <- rep("Vaccine", dim(schisto_coverage)[1])
# check new column has been added
head(schisto_coverage)


# reorder and rename column names
col_years <- colnames(schisto_coverage)[grepl(20, colnames(schisto_coverage))]
col_ages <- colnames(schisto_coverage)[grepl("age", colnames(schisto_coverage))]
schisto_coverage<- schisto_coverage[, c("ISO", "Intervention Type", "platform", "platform", col_ages, col_years)]

colnames(schisto_coverage)[1:4] <- c("Country/Region", "Intervention Type",	"Platform Type",	"Platform")

write.csv(schisto_coverage, file = "schisto_coverage.csv")

# then copy and paste into mansoni_coverage_scenario_1.xlsx 