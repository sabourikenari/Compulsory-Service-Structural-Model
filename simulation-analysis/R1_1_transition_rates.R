library(tidyverse)
library(data.table)
library(kableExtra)

#= change the directory to where the data is located
rm(list = ls())
setwd("C:/Users/ehsa7798/My Drive/Projects/Labor/Github/Compulsory-Service-Structural-Model")

LFS <- readRDS("./data/LFS/LFS.rds") %>%
  select(year, season, gender, birth_y, birth_m, age, student, labor, marital, W3, C09, education, person, household, RU, C44, C40MAH, C40SAL, C41, DJAYGOZIN, migreason)

LFS <- LFS %>% 
  rename(ocp    = C09,
         last_job      = C41) %>% 
  mutate(year= year+1921) %>% 
  as.data.frame()

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
    age = ifelse(C44 == "military"&!is.na(C44), round((year*12-birth_y*12+month-birth_m-C40)/12,0), age),
    occupation = as.integer(str_sub(str_pad(ocp,4,pad = "0"),1,1)),
    choice = case_when(
      conscript3 ~ "conscription",
      occupation %in% c(1,2,3) ~ "white-collar",
      occupation %in% c(4,5,6,7,8,9) ~ "blue-collar",
      student=="Y" ~ "study",
      TRUE ~ "home")) 

next_year <- LFS %>% 
  select(household,year,person,gender,season,RU,age,DJAYGOZIN,choice) %>% 
  filter(DJAYGOZIN==2) %>% 
  mutate(year=year-1, age=age-1)

LFS <- left_join(
  x=LFS,
  y=next_year,
  by=c("household","year","person","gender","season","RU","age","DJAYGOZIN"),
  suffix=c("","_next")
)
rm(next_year)


## ----------------------------------------------------------------------------
## simulated data transition rate
sim <- read.table("./data/simulation/simNew.csv",header = FALSE,sep=",")
colnames(sim) <- c("age","education","x3","x4","choice","income","educated","x5","type","Emax","choice_next","homeSinceSchool") 

sim <- sim %>% 
  mutate(id = (row_number()-1)%/%50+1,
         choice=case_when(choice==1 ~ "home",
                          choice==2 ~ "study",
                          choice==3 ~ "white-collar",
                          choice==4 ~ "blue-collar",
                          choice==5 ~ "conscription"),
         choice= as.factor(choice)) %>% 
         select(-choice_next) 



## -----------------------------------------------------------------------------
# transitions rates in model and actual data
next_year <- sim %>% 
  select(id,age,choice) %>% 
  mutate(age=age-1)

sim <- left_join(
  x       = sim ,
  y       = next_year,
  by      = c("id","age"),
  suffix  = c("","_next")
)
remove(next_year)

simulated_trans <- sim %>% 
  group_by(age,choice) %>% 
  summarise(
    "study" = mean(choice_next=="study"),
    "blue-collar" = mean(choice_next=="blue-collar"),
    "white-collar" = mean(choice_next=="white-collar"),
    "conscription" = mean(choice_next=="conscription"),
    "home" = mean(choice_next=="home")
  ) %>%
  pivot_longer(cols = study:home,names_to="choice_next",values_to="model")


# transition rates of cohort 1985-1990
actual_trans <- LFS %>% 
  filter(!is.na(choice),!is.na(choice_next),age>=16,age<=65) %>%
  filter(! year %in% c(91+1921,87+1921)) %>%
  filter(birth_y>=64,birth_y<=69, gender=="male") %>% 
  group_by(age,choice) %>% 
  summarise(
    "study" = weighted.mean(choice_next=="study",W3),
    "blue-collar" = weighted.mean(choice_next=="blue-collar",W3),
    "white-collar" = weighted.mean(choice_next=="white-collar",W3),
    "conscription" = weighted.mean(choice_next=="conscription",W3),
    "home" = weighted.mean(choice_next=="home",W3)
  ) %>% 
  pivot_longer(cols = study:home,names_to="choice_next",values_to="data") %>% 
  group_by(age,choice) %>% 
  mutate(rank = order(order(data,decreasing = TRUE)))

trans <- left_join(actual_trans,simulated_trans,by=c("age","choice","choice_next"))
remove(actual_trans, simulated_trans)

trans <- trans %>% pivot_longer(c("data","model"),values_to="transition",names_to="type")

share <- LFS %>%
  filter(gender=="male",!is.na(choice),birth_y>=64,birth_y<=69,age>=16) %>%
  group_by(age) %>%
  summarise(
    "study" = weighted.mean(choice=="study",W3),
    "blue-collar" = weighted.mean(choice=="blue-collar",W3),
    "white-collar" = weighted.mean(choice=="white-collar",W3),
    "conscription" = weighted.mean(choice=="conscription",W3),
    "home" = weighted.mean(choice=="home",W3)
  ) %>%
  pivot_longer(study:home,names_to="choice",values_to="share")
trans <- left_join(trans,share,by=c("age","choice"))
remove(share)



## -----------------------------------------------------------------------------
std <- read.table("./data/Moments/transMomentStdBoot.csv")
colnames(std) <- c("age","choice","choice_next","transition","std","simulation")

std <- std %>% 
  mutate(
    choice=case_when(choice==1 ~ "home",
                     choice==2 ~ "study",
                     choice==3 ~ "white-collar",
                     choice==4 ~ "blue-collar",
                     choice==5 ~ "conscription"),
    choice= as.factor(choice) ,
    choice_next=case_when(choice_next==1 ~ "home",
                          choice_next==2 ~ "study",
                          choice_next==3 ~ "white-collar",
                          choice_next==4 ~ "blue-collar",
                          choice_next==5 ~ "conscription"),
    choice_next= as.factor(choice_next) 
  )

# std <- trans_STD %>% mutate(std=STD)

std <- std %>% 
  mutate(
   trans_high = (transition + 2.5*std) * 100,
   trans_low  = (transition - 2.5*std) * 100,
   type = "upper and lower bound"
  )


# graph_name <- c("home"="home" 
#                 ,"study"="study",
#                 "white-collar"="white-collar"
#                 ,"blue-collar"="blue-collar"
#                 ,"conscription"="conscription")


ggplot()+ 
  facet_wrap(~factor(choice, levels=c("blue-collar","study","white-collar","home","conscription"))
             ,scales = "free",ncol = 5) +
  geom_line(data = trans %>% mutate(transition=transition*100) %>% filter(age<=35,age>=19,choice==choice_next,type=="data") ,
            mapping = aes(x=age,y=transition,linetype="data",color="data") ,size=1.4)+
  
  geom_line(data = trans %>% mutate(transition=transition*100) %>% filter(age<=35,age>=19,choice==choice_next,type=="model") ,
            mapping = aes(x=age,y=transition,linetype="model",color="model") ,size=1.4)+
  
  geom_line(data=std %>% filter(age<=35,choice==choice_next,age>=19),
            mapping = aes(x=age,y=trans_high,linetype="95% CI",color="95% CI"),size=0.2) +
  
  geom_line(data=std %>% filter(age<=35,choice==choice_next,age>=19),
            mapping = aes(x=age,y=trans_low,linetype="95% CI",color="95% CI"),size=0.2) +
  
  scale_linetype_manual(name = "type",
                        breaks = c("data", "model","95% CI"),
                        values = c("data" = "solid", "model" = "twodash", "95% CI"="dashed")) +
  scale_color_manual(name = "type",
                     breaks = c("data", "model","95% CI"),
                     values = c("data" = "blue", "model" = "darkgreen", "95% CI"="blue" )) +
  theme(legend.position = "bottom", legend.box = "vertical")+
  theme(strip.background =element_rect(colour="white",fill="white"))+
  theme(strip.text = element_text(colour = 'black'),
        panel.grid.major = element_blank() ) +
  scale_x_continuous( breaks = seq(19,33,3) ,limits =c(18.5,32)) +
  scale_y_continuous(limits = c(0,100)) + labs(x="age",y="transition rate")

# ggsave("./Data analysis/Results/R1_1_fit_transition_rate_data.pdf",width = 12, height = 4)



# Prepare data
trans_base <- trans %>%
  mutate(transition = transition * 100) %>%
  filter(age >= 19, age <= 35, rank <= 5, share >= 0)

trans_data  <- filter(trans_base, type == "data")
trans_model <- filter(trans_base, type == "model")
std_base    <- filter(std, age >= 19, age <= 35)

# Plot
ggplot(trans_base, aes(x = age, y = transition, color = type, linetype = type)) +
  facet_grid(choice_next ~ choice, scales = "fixed") +
  
  geom_line(data = trans_data,  aes(color = "data",  linetype = "data"),  size = 0.8) +
  geom_line(data = trans_model, aes(color = "model", linetype = "model"), size = 0.6) +
  geom_line(data = std_base, aes(y = trans_high, color = "95% CI", linetype = "95% CI"), size = 0.2) +
  geom_line(data = std_base, aes(y = trans_low,  color = "95% CI", linetype = "95% CI"), size = 0.2) +
  
  scale_linetype_manual(
    name   = "type",
    breaks = c("data", "model", "95% CI"),
    values = c("data" = "solid", "model" = "twodash", "95% CI" = "dashed")
  ) +
  scale_color_manual(
    name   = "type",
    breaks = c("data", "model", "95% CI"),
    values = c("data" = "blue", "model" = "darkgreen", "95% CI" = "blue")
  ) +
  
  theme_light() +
  theme(
    legend.position  = "bottom",
    legend.box       = "vertical",
    strip.background = element_rect(colour = "white", fill = "white"),
    strip.text       = element_text(colour = "black"),
    panel.grid       = element_blank()
  ) +
  labs(y = "transition rate") +
  scale_x_continuous(breaks = seq(19, 33, 4))

ggsave("./results/simulation-analysis/R1_1_fit_transition_rate_data_all2.pdf", width = 9,height = 8)





## -----------------------------------------------------------------------------
# calculate transition moment condition and standard errors
# remove(sim,df)

data <- LFS %>% 
  filter(!is.na(choice),!is.na(choice_next),age>=16,age<=65) %>%
  filter(! year %in% c(91+1921,87+19211)) %>%
  filter(birth_y>=64,birth_y<=69, gender=="male") %>% 
  select(age,choice,choice_next,W3)

for (num in 1:100){
  
  Sample <- data[sample(nrow(data), round(0.5*nrow(data))), ]
  
  TRANS <- Sample %>% 
    group_by(age,choice) %>% 
    summarise(
      "study" = weighted.mean(choice_next=="study",W3),
      "blue-collar" = weighted.mean(choice_next=="blue-collar",W3),
      "white-collar" = weighted.mean(choice_next=="white-collar",W3),
      "conscription" = weighted.mean(choice_next=="conscription",W3),
      "home" = weighted.mean(choice_next=="home",W3)
    ) %>% 
    pivot_longer(cols = study:home,names_to="choice_next",values_to="data")#paste("data",num,sep = ""))
  
  if (num==1){
    trans_STD <- TRANS
  }
  else{
    trans_STD <- trans_STD %>% 
      bind_rows(TRANS)
   # trans_STD <- left_join(trans_STD,trans, by=c("age","choice","choice_next")) 
  }
}
remove(data, Sample, TRANS)

trans_STD <- trans_STD %>% 
  group_by(age,choice,choice_next) %>% 
  summarise(transition = mean(data),
            STD = var(data)) %>% 
  mutate(STD = sqrt(STD),
         ratio = transition/STD)

trans_STD <- left_join(trans_STD, share, by=c("age","choice"))

trans_STD <- trans_STD %>% 
  mutate(choice=case_when(choice=="home"          ~ 1,
                          choice=="study"         ~ 2,
                          choice=="white-collar"  ~ 3,
                          choice=="blue-collar"   ~ 4,
                          choice=="conscription"  ~ 5),
         choice_next=case_when(choice_next=="home"          ~ 1,
                               choice_next=="study"         ~ 2,
                               choice_next=="white-collar"  ~ 3,
                               choice_next=="blue-collar"   ~ 4,
                               choice_next=="conscription"  ~ 5))


trans_STD <- trans_STD %>% 
  select(- share,- ratio) %>% 
  mutate(transitionSim = 0.0)
# filter(share>0.07, transition>=0.05) %>% 

# write.table(trans_STD, file="./Moments/transMomentStdBoot.csv",col.names = FALSE,row.names = FALSE)

