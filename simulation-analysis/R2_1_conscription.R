## ----------------------------------------------------------------------------
library(tidyverse)
library(data.table)
# library(kableExtra)

rm(list = ls())
path = "C:/Users/ehsa7798/GoogleDrive/Projects/Labor/Codes/"
path = "C:/Users/claudioq/My Drive/Projects/Labor/Codes"
setwd(path)


## load data
LFS <- readRDS("../Data/LFS/LFS.rds") %>%
  select(year:W3, C05, C09, C37SAL, C37MAH, C40SAL, C40MAH, C44, labor) %>% 
  rename(ocp = C09)

# library(foreign)
# write.dta(LFS, "C:/Academic/HEIS/LFS.dta")


## ----------------------------------------------------------------------------
## life-time occupational choice

## over years
LFS <- LFS %>%
  filter(gender=="male", 
         age>=16, age<=65) %>%
  mutate(conscript1 = case_when(
    ocp %in% c(112,510) ~ T,
    migreason=="military service" ~ T,
    TRUE ~ F),
    conscript2 = (!is.na(C44)&C44 == "military") ,
    conscript3 = conscript2 | conscript1,
    month=case_when(season==1 ~ 2,
                    season==2 ~ 5,
                    season==3 ~ 8,
                    season==4 ~ 11),
    across(c(C40SAL, C40MAH), ~replace_na(.x, 0)),
    C40 = C40SAL*12+C40MAH,
    Age = ifelse(C44 == "military"&!is.na(C44), round((year*12-birth_y*12+month-birth_m-C40)/12,0), age),
    occupation = as.integer(str_sub(str_pad(ocp,4,pad = "0"),1,1)),
    choice = case_when(
      conscript3 ~ "conscription",
      occupation %in% c(1,2,3) ~ "white-collar",
      occupation %in% c(4,5,6,7,8,9) ~ "blue-collar",
      student=="Y" ~ "study",
      TRUE ~ "home")) 

LFS %>%
  group_by(year,choice) %>%
  summarize(n=sum(W3)) %>%
  group_by(Year=year+1921) %>%
  mutate(total=sum(n),
         percent=n/total*100,
         blue = choice=="blue-collar",
         choice = fct_reorder(as.factor(choice),-percent)) %>%
  ggplot(aes(Year,percent,color=choice,shape=choice)) +
  geom_line() + geom_point() + theme_bw()+
  facet_grid(!blue ~. ,  scales = "free_y", space = "free_y")+
  scale_x_continuous(breaks = 2005:2020) +
  scale_y_continuous(breaks = seq(0,60,5))+
  labs(title = "Choice share of men 16 to 65 within years") +
  theme(strip.background = element_blank(),
        strip.text = element_blank())
# ggsave("./Data analysis/Results/R2_1_choice_year.pdf",width = 8, height = 5)


## over life-cycle
Choice <- LFS %>%
  mutate(birth_y = birth_y+1921) %>% 
  filter(Age>=16&Age<=63,
         birth_y %in% c( 1985:1990) ) %>%
  group_by(Age,choice) %>%
  summarize(n=sum(W3),sample=n()) %>%
  group_by(Age) %>%
  mutate(total=sum(n),
         sample=sum(sample),
         percent=n/total*100,
         choice = fct_reorder(as.factor(choice),-percent)) %>%
  select(Age, choice, percent, sample)

Choice %>%
  filter(Age<=60) %>% 
  ggplot(aes(Age, percent, color=choice, shape=choice)) +
  geom_point()+ geom_line()+ theme_bw()+ 
  scale_x_continuous(breaks = seq(16,62,3))+
  labs(title = "Choice shares of birth cohorts 1985-90 by age")
# ggsave("./Data analysis/Results/R2_1_choices.pdf",width = 8, height = 5)


# Table <- Choice %>%
#   mutate(percent=round(percent,1)) %>%
#   pivot_wider(c(Age,sample), names_from = "choice", values_from = percent) 
# Table
# Table %>% relocate(sample, .after = last_col()) %>% 
#   relocate(`white-collar`, .after = Age) %>% 
#   kable(format = 'latex', booktabs = TRUE) # latex format

result <- Choice %>%
  group_by(choice) %>%
  summarize(total_value = sum(percent, na.rm = TRUE))




## ----------------------------------------------------------------------------
## estimating moments and corresponding standard errors

data <- LFS %>%
  mutate(birth_y = birth_y+1921) %>% 
  filter(Age>=16&Age<64, birth_y %in% c(1985,1986,1987,1988,1989,1990) )
# data <- LFS %>%  filter(Age>=16&Age<64, birth_y %in% 54:69)
# rm(LFS3,LFS) 

for (num in 1:250){
  
  Sample <- data[sample(nrow(data), round(0.3*nrow(data))), ]
  
  Choice <- Sample %>%
    group_by(Age,choice) %>%
    summarize(n=sum(W3),sample=n()) %>%
    group_by(Age) %>%
    mutate(total=sum(n),
           sample=sum(sample),
           percent=n/total,
           choice = fct_reorder(as.factor(choice),-percent)) %>%
    select(Age, choice, percent, sample)
  
  if (num==1){
    ChoiceSTD <- Choice
  }
  else{
    ChoiceSTD <- ChoiceSTD %>% 
      bind_rows(Choice)
    # trans_STD <- left_join(trans_STD,trans, by=c("age","choice","choice_next")) 
  }
}

ChoiceSTD <- ChoiceSTD %>% 
  group_by(Age, choice) %>% 
  summarise(share = mean(percent),
            STD = var(percent)) %>% 
  mutate(STD = sqrt(STD))

# ChoiceSTD %>%
#   mutate(shareTop = share + 2*STD) %>% 
#   ggplot(aes(Age, share, color=choice, shape=choice)) +
#   geom_point()+ geom_line()+ theme_bw()+ 
#   # scale_x_continuous(breaks = 16:35)+
#   labs(title = "Choice shares of birth cohorts 1985-90 by age")

ChoiceSTD <- ChoiceSTD %>%
  pivot_wider(id_cols="Age",names_from = "choice", values_from="STD",
              names_prefix="STD")

ChoiceSTD <- ChoiceSTD %>% select(c(Age,STDhome,STDstudy,`STDwhite-collar`,`STDblue-collar`,STDconscription))
Choice <- data %>%
  filter(Age>=16&Age<=63 ) %>%
  group_by(Age,choice) %>%
  summarize(n=sum(W3),sample=n()) %>%
  group_by(Age) %>%
  mutate(total=sum(n),
         sample=sum(sample),
         percent=n/total,
         choice = fct_reorder(as.factor(choice),-percent)) %>%
  select(Age, choice, percent) %>%
  pivot_wider(names_from = "choice", values_from="percent") %>% 
  select(c(Age,home,study,'white-collar','blue-collar',conscription)) %>% 
  right_join(ChoiceSTD, by="Age")
  
# write.table(Choice,"./Moments/choiceMomentLFS.csv", row.names = FALSE,col.names = FALSE)




## ----------------------------------------------------------------------------
## choice moment classified by education
# data <- LFS %>%
#   mutate(birth_y = birth_y+1921) %>% 
#   filter(Age>=16&Age<64, birth_y %in% c(1983,1984,1985,1986,1987,1988,1989) )
data <- LFS %>%
  mutate(birth_y = birth_y+1921) %>% 
  filter(Age>=16&Age<64, birth_y %in% 1975:1990)
# rm(LFS3,LFS) 

data <- data %>% 
  mutate(educated = case_when(
    Age<22 & Age>=16 ~ -1,
    literate == "N" ~ 0,
    education %in% c("primary school", "guidance school", "high school", "diploma","other/informal") ~ 0,
    education %in% c("college","bachelor","master","PhD") ~ 1,
      ))

for (num in 1:200){
  
  Sample <- data[sample(nrow(data), round(0.3*nrow(data))), ]
  
  Choice <- Sample %>%
    filter(, !is.na(educated)) %>% 
    group_by(Age,choice,educated) %>%
    summarize(n=sum(W3),sample=n()) %>%
    group_by(Age,educated) %>%
    mutate(total=sum(n),
           sample=sum(sample),
           percent=n/total,
           choice = fct_reorder(as.factor(choice),-percent)) %>%
    select(Age, choice, percent)
  
  if (num==1){
    ChoiceSTD <- Choice
  }
  else{
    ChoiceSTD <- ChoiceSTD %>% 
      bind_rows(Choice)
    # trans_STD <- left_join(trans_STD,trans, by=c("age","choice","choice_next")) 
  }
}


ChoiceSTD <- ChoiceSTD %>% 
  group_by(Age, educated, choice) %>% 
  summarise(share = mean(percent),
            STD = var(percent)) %>% 
  mutate(STD = sqrt(STD))

ChoiceSTD <- ChoiceSTD %>% 
  pivot_wider(id_cols=c("Age","educated"),names_from = "choice", values_from="STD",
              names_prefix="STD")

ChoiceSTD <- ChoiceSTD %>% select(c(Age,educated,STDhome,STDstudy,`STDwhite-collar`,`STDblue-collar`,STDconscription))
Choice <- data %>%
  filter(Age>=16&Age<=65, !is.na(educated)) %>%
  group_by(Age,educated,choice) %>%
  summarize(n=sum(W3),sample=n()) %>%
  group_by(Age,educated) %>%
  mutate(total=sum(n),
         sample=sum(sample),
         percent=n/total,
         choice = fct_reorder(as.factor(choice),-percent)) %>%
  select(Age,educated, choice, percent) %>%
  pivot_wider(names_from = "choice", values_from="percent") %>% 
  select(c(Age,educated,home,study,'white-collar','blue-collar',conscription)) %>% 
  right_join(ChoiceSTD, by=c("Age","educated"))

Choice[is.na(Choice)] <- 0
# write.table(Choice,"./Moments/choiceMomentSTDLFS.csv", row.names = FALSE,col.names = FALSE)


## ----------------------------------------------------------------------------
## share of education categories moments

for (num in 1:200){
  
  Sample <- data[sample(nrow(data), round(0.4*nrow(data))), ]
  
  Choice <- Sample %>%
    filter(Age>=24, Age<=32, !is.na(educated)) %>% 
    group_by(Age) %>%
    summarize(percent = weighted.mean(educated,W3))
  
  if (num==1){
    ChoiceSTD <- Choice
  }
  else{
    ChoiceSTD <- ChoiceSTD %>% 
      bind_rows(Choice)
    # trans_STD <- left_join(trans_STD,trans, by=c("age","choice","choice_next")) 
  }
}

ChoiceSTD <- ChoiceSTD %>% 
  group_by(Age) %>% 
  summarise(share = mean(percent),
            STD = var(percent)) %>% 
  mutate(STD = sqrt(STD))
# write.table(ChoiceSTD,"./Moments/educatedShareSTDLFS.csv", row.names = FALSE,col.names = FALSE)




# ## ----------------------------------------------------------------------------
# ### conscription length among unemployed individuals
# length <- LFS %>% 
#   mutate(across(c(C37SAL, C37MAH), ~replace_na(.x, 0)),
#          C37 = C37SAL*12+C37MAH,
#          edu = fct_collapse(education,`up to diploma`=c("primary school", "guidance school", "high school", "diploma","other/informal"),
#                             bachelor="college")) %>%
#   filter(C44 == "military", C37<=24,!is.na(edu)) 
# 
# summary(length$C37)
# 
# length %>% 
#   # filter(C37>0) %>% 
#   group_by(edu) %>% 
#   summarize(conscription_month=mean(C37),size=n())
# 
# # estimating a military service deduction model
# summary(length %>% filter(C37>0) %>% select(C37))
# 
# ggplot(length, aes(x=C37)) +
#   geom_histogram(binwidth=1, colour="black", fill="white", aes(y=..density..),) +
#   theme_bw()
# 
# # model <- lm(formula= C37 ~ edu, data=length %>% filter(C37>=10) %>% mutate(C37=(C37-10)/14))
# # summary(model)



# ## ----------------------------------------------------------------------------
# ## conscription share based on different definitions
# LFS2 <- LFS %>%
#   filter(gender=="male", 
#          age>=18, age<40,
#          # birth_y %in% 66:70 
#   ) %>%
#   mutate(conscript1 = case_when(
#     ocp %in% c(112,510) & C05 ~ T,
#     ocp %in% c(112,510) & age<30 ~ T,
#     migreason=="military service" ~ T,
#     TRUE ~ F),
#     conscript2 = (!is.na(C44)&C44 == "military") ,
#     conscript3 = conscript2 | conscript1,
#     month=case_when(season==1 ~ 2,
#                     season==2 ~ 5,
#                     season==3 ~ 8,
#                     season==4 ~ 11),
#     across(c(C40SAL, C40MAH), ~replace_na(.x, 0)),
#     C40 = C40SAL*12+C40MAH,
#     Age = ifelse(C44 == "military"&!is.na(C44), round((year*12-birth_y*12+month-birth_m-C40)/12,0), age) ) %>%
#   filter(Age>=18)
# 
# share1 <- prop.table(xtabs(W3~conscript1+age, LFS2),2)
# share2 <- prop.table(xtabs(W3~conscript2+Age, LFS2),2)
# share3 <- prop.table(xtabs(W3~conscript3+Age, LFS2),2)
# 
# tshare1 <- round(apply(share1, 1, sum)["TRUE"]*100*12/mean(length$C37), 1)
# tshare2 <- round(apply(share2, 1, sum)["TRUE"]*100*12/mean(length$C37), 1)
# tshare3 <- round(apply(share3, 1, sum)["TRUE"]*100*12/mean(length$C37), 1)
# 
# share3 %>%
#   as.data.frame() %>%
#   full_join(share1 %>% as.data.frame(), by=c("Age"="age","conscript3"="conscript1")) %>%
#   full_join(share2 %>% as.data.frame(), by=c("Age","conscript3"="conscript2")) %>%
#   filter(conscript3=="TRUE") %>%
#   pivot_longer(starts_with("Freq"), names_to ="variable", names_prefix = "Freq.", values_to = "Freq") %>%
#   mutate(percent=Freq*100,
#          Age=as.numeric(as.character(Age)),
#          definition=case_when(variable=="x"~ paste0("occupation + unemployed (",tshare3,"%)"),
#                               variable=="y" ~ paste0("only occupation (",tshare1,"%)"),
#                               variable=="Freq" ~ paste0("only unemployed (",tshare2,"%)"))) %>%
#   ggplot(aes(x=Age, y=percent, linetype=definition, color=definition)) +
#   geom_line() +
#   theme_bw() + 
#   scale_x_continuous(breaks = 18:40) +
#   labs(title = "conscription percent",
#        color = "Definitions (total percent)", 
#        linetype= "Definitions (total percent)",
#        caption = paste("Estimation using LFS. Average conscription length is assumed" ,round(mean(length$C37),1), "months for computing total percent."))
# 
# # ggsave(paste0(path,"conscription.pdf"),width = 8.5, height = 5)
# 
# rm(LFS2,length)
# gc()





# ## ----------------------------------------------------------------------------
# ## Conscription by age group by adding next year migration in the rotating panel
# ## The results are not very different from original
# df <- LFS %>%
#   filter(DJAYGOZIN!=1) %>%
#   group_by(household, person) %>%
#   mutate(n=n()) %>%
#   filter(n>1) 
# 
# DT <- data.table(df)
# DT <- DT[order(household, person, season, year)] 
# 
# DT <- DT[,':='(Llabor = shift(labor, 1, type ="lead"), 
#                LC44 = shift(C44, 1, type ="lead"),
#                Locp = shift(ocp, 1, type="lead"),
#                Lmigreason = shift(migreason, 1, type="lead"),
#                Lyear = shift(year, 1, type ="lead")),
#          by=.(household, person, season)]
# 
# Panel <- as.data.frame(DT) 
# rm(DT,df)
# Panel[((Panel$year+1<Panel$Lyear)|(Panel$year %in% c(87,91,96)))&!is.na(Panel$Llabor), c("Llabor","LC44","Locp", "Lmigreason")] <- NA
# 
# panel <- Panel %>%
#   filter(gender=="male", 
#          age>=18, age<40,
#          #birth_y %in% 64:69 
#          ) %>%
#   mutate(conscript1 = case_when(
#     ocp %in% c(112,510)  & C05 ~ T,
#     ocp %in% c(112,510) & age<30 ~ T,
#     migreason=="military service" ~ T,
#     Lmigreason=="military end" ~ T,
#     TRUE ~ F),
#     conscript2 = (!is.na(C44) & C44 == "military") ,
#     conscript3 = conscript2 | conscript1,
#     Age = ifelse(C44 == "military"&!is.na(C44), age-C40SAL, age),
#     conscriptlength = case_when(
#       migreason=="military service" & (Locp %in% c(112,510)&!is.na(Llabor))  ~ 2L,
#       migreason=="military service" & (Locp!=112 & Locp!=510 | is.na(Locp)&!is.na(Llabor))  ~ 1L,
#       TRUE ~ NA_integer_ ) ) %>%
#   filter(Age>=18)
# 
# share1 <- prop.table(xtabs(W3~conscript1+age, panel),2)
# share2 <- prop.table(xtabs(W3~conscript2+Age, panel),2)
# share3 <- prop.table(xtabs(W3~conscript3+Age, panel),2)
# 
# tshare1 <- round(apply(share1, 1, sum)["TRUE"]*100*12/mean(length$C37), 1)
# tshare2 <- round(apply(share2, 1, sum)["TRUE"]*100*12/mean(length$C37), 1)
# tshare3 <- round(apply(share3, 1, sum)["TRUE"]*100*12/mean(length$C37), 1)
# 
# share3 %>%
#   as.data.frame() %>%
#   full_join(share1 %>% as.data.frame(), by=c("Age"="age","conscript3"="conscript1")) %>%
#   full_join(share2 %>% as.data.frame(), by=c("Age","conscript3"="conscript2")) %>%
#   filter(conscript3=="TRUE") %>%
#   pivot_longer(starts_with("Freq"), names_to ="variable", names_prefix = "Freq.", values_to = "Freq") %>%
#   mutate(percent=Freq*100,
#          Age=as.numeric(as.character(Age)),
#          definition=case_when(variable=="x"~ paste0("occupation + unemployed (",tshare3,"%)"),
#                               variable=="y" ~ paste0("only occupation (",tshare1,"%)"),
#                               variable=="Freq" ~ paste0("only unemployed (",tshare2,"%)"))) %>%
#   ggplot(aes(x=Age, y=percent, linetype=definition, color=definition)) +
#   geom_line() +
#   theme_bw() + 
#   scale_x_continuous(breaks = 18:40) +
#   labs(title = "conscription percent based on rotating panel sample",
#        color = "Definitions (total percent)", 
#        linetype= "Definitions (total percent)",
#        caption = paste("Estimation using LFS. Average conscription length is assumed" ,round(mean(length$C37),1), "months for computing total percent."))
# # ggsave(paste0(path,"conscription_panel.pdf"),width = 8.5, height = 5)
