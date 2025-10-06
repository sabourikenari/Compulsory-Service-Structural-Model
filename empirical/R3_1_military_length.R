

library(tidyverse)
library(data.table)
library(kableExtra)

#= change the directory to where the data is located
rm(list = ls())
# setwd("C:/Users/ehsa7798/My Drive/Projects/Labor/Github/Compulsory-Service-Structural-Model")
setwd("G:/My Drive/Projects/Labor/Github/Compulsory-Service-Structural-Model")


LFS <- readRDS("./data/LFS/LFS.rds") %>%
  select(year, season, gender, birth_y, birth_m, age, student, labor, marital, W3, C09, education, person, household, RU, C44, C40MAH, C40SAL, C41, DJAYGOZIN, migreason, C37SAL, C37MAH)

LFS <- LFS %>% 
  rename(ocp    = C09,
         last_job      = C41) %>% 
  mutate(year= year+1921) %>% 
  as.data.frame()


# Create a new variable 'education_group' with 3 broad education groups
LFS <- LFS %>%
  mutate(
    education_group = case_when(
      education %in% c("illiterate", "primary school", "guidance school") ~ "Low",
      education %in% c("high school", "diploma") ~ "Medium",
      education %in% c("bachelor", "bachelor", "master", "phd") ~ "High",
      TRUE ~ NA_character_
    )
  )


# calculate average service length for military service by education level 
avg_C37_by_education <- LFS %>%
  mutate(C37 = C37SAL * 12 + C37MAH,
         yob = 1921+birth_y) %>%
  filter(C44 == "military", age >= 18, age <= 35, yob>=1985, yob<=1990, C37<=24 ) %>%
  group_by(education_group) %>%
  summarise(
    avg_C37 = mean(C37, na.rm = TRUE),
    n = n()
  )

print(avg_C37_by_education)







# Define years of schooling based on education
LFS <- LFS %>%
  mutate(
    years_schooling = case_when(
      education %in% c("illiterate") ~ 0,
      education %in% c("primary school") ~ 5,
      education %in% c("guidance school") ~ 8,
      education %in% c("high school", "diploma") ~ 12,
      education %in% c("college", "associate") ~ 14,
      education %in% c("bachelor") ~ 16,
      education %in% c("master") ~ 18,
      education %in% c("PhD") ~ 21,
      TRUE ~ NA_real_
    )
  )

# Calculate average C37 by years of schooling for military sample
avg_C37_by_schooling <- LFS %>%
  mutate(yob = 1921 + birth_y) %>%
  filter( !is.na(years_schooling), yob<=1990, yob>=1985) %>%
  mutate(C37 = C37SAL * 12 + C37MAH) %>% filter(C44=="military", C37<=24) %>% 
  group_by(years_schooling) %>%
  summarise(
    avg_C37 = mean(C37, na.rm = TRUE),
    n = n()
  )

# Plot average C37 by years of schooling, dot size proportional to sample size, only dots
ggplot(avg_C37_by_schooling, aes(x = years_schooling, y = avg_C37, size = n)) +
  geom_point() +
  labs(
    title = "Average C37 by Years of Schooling (Military Sample)",
    x = "Years of Schooling",
    y = "Average C37",
    size = "Sample Size"
  ) +
  theme_minimal()


# Calculate average C37 by years of schooling for military sample
LFS %>%
  mutate(yob = 1921 + birth_y) %>%
  mutate(C37 = C37SAL * 12 + C37MAH) %>%
  filter(yob<=1990, yob>=1985, C44=="military", C37<=24) %>%
  summarise(
    avg_C37 = mean(C37, na.rm = TRUE),
    n = n()
  )

