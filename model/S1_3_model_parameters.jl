



# parameters in the utility functions
#**********************
ω1T1 = 18.024650104928808     ;   # the intercept of staying home α10 for type 1
ω1T2 = 17.955416493410194       ;   # the intercept of staying home α10 for type 2
ω1T3 = 16.682086435181933      ;   # the intercept of staying home α10 for type 3

#**********************
ω2T1 = 17.74664705449791      ;    # the intercept of studying for type 1
ω2T2 = 18.69050689327409     ;    # the intercept of studying for type 2
ω2T3 = 18.925100092916386      ;    # the intercept of studying for type 3

α21 = log(6.478139410089405e7* 1.655)     ;    # study in (t-1)?
tc1T1 = log(7.2875303658261e7 * 0.89)    ;    # education >= 12?
# tc1T2 = 4.5553275303767666e7    ;    # education >= 12?
# tc1T3 = 4.5553275303767666e7    ;    # education >= 12?
# tc1T4 = 4.553275303767666e7    ;    # education >= 12?
tc2 = log(5.768484020567006e7 * 1.96)     ;    # education >= 16?

α22 = 0.013300228480131503 # reward of getting diploma
α23 = 0.11905513439049648 # reward of graduating college

# α24 = 0.137 # reward of getting diploma
α25 = 0.08438895479276255 # reward of graduating college


#**********************
#= occupational choices: 3=white, 4=blue collar =#
α3, α4 = log(9.951978593232237e6)   , 0 ;          # the intercept outside exp()

#= the intercept inside exp() for type 1 =#
ω3T1, ω4T1 = 15.248424468389986   , 17.227179189695196    ;
#= the intercept inside exp() for type 2 =#
ω3T2, ω4T2 = 15.511731215693877   ,  17.181200742305314  ;
#= the intercept inside exp() for type 3 =#
ω3T3, ω4T3 = 15.51531219560882   , 16.466374321169333   ;


#**********************
#= share of each type for those education less than 10 in 15 years old =#
πE1T1 = 0.7350832838561688
πE1T2 = 0.16063685478858179
πE1T3 = 1- πE1T1- πE1T2

# den = 1/(1+πE1T1+πE1T2+πE1T3)
πE1T1exp = log(πE1T1/πE1T3)
πE1T2exp = log(πE1T2/πE1T3)


#= share of each type for those education equalls 10 in 15 years old =#
πE2T1 = 0.6496483687430918
πE2T2 = 0.16759113096398324
πE2T3 = 1- πE2T1- πE2T2

# den = 1/(1-0.6286483687430918-0.14859113096398324-0.22276050029292488)
πE2T1exp = log(πE2T1/πE2T3)
πE2T2exp = log(πE2T2/πE2T3)



#**********************
#= education coefficients =#
α31, α41 =  0.13255810152514091 , 0.03322414122357448 ;
#= experience in white collar =#
α32, α42 = 0.06105854604797636 , 0.0 ;
#= experience in blue collar =#
α33, α43 = 0.0 , 0.07852835990498894 ;
#= experience^2 in white collar =#
α34, α44 = -0.0013898110123866892 , -0.0 ;
#= experience^2 in blue collar =#
α35, α45 = -0.0 , -0.0016269663508598476 ;

#= Job finding cost of an individual without job-specific experience =#
α36 = 0.08685497695354861 ; # In white-collar occupation
α46 = 0.0241949247996808 ; # In bllue-collar occupation

#= Job finding cost of a college graduate relaltive to others =#
α37 = 0.03808437166357896 ; # In white-collar occupation
α47 = 0.0091201976579784 ; # In bllue-collar occupation

#= Not working in the same occupation in the previous period=#
α38 = -0.032323353209471405 ; # In white-collar occupation
α48 = -0.01060669835779854 ; # In bllue-collar occupation

#**********************
α50 = 13.414080014460028 # intercept in util5 (conscription)
α51 = log(8.119970291315692e6)*1.090 ;    # util5 coeff for if educ >= 12
α52 = log(9.7071807618222445e6)*1.062 ;     # util5 coeff for if educ >= 16

#**********************
#= Variance-covariance of shocks =#
σ1 = log(4.1052186750061806e14) ;  # variance of ε1 - staying home
σ2 = log(4.563404594520161e13) ;  # variance of ε2 - studying
σ3 = 0.27505422080935854 ;    # variance of ε3 - white collar
σ4 = 0.24676189791392824 ;    # variance of ε4 - blue collar
σ34 = +0.0949239790308403 ;    # Covariance of white and blue collar shocks

σ5 = log(7.95221587709376e14) ;

# π1 = 0.7 ;     # share of individuals type 1
# π1T1exp = -log((1/0.66)-1)
# π1T2exp = -log((1/0.66)-1)
# π1T3exp = -log((1/0.66)-1)


#= New parameters in the model =#
α11 = -log(2.578597211142504e6)*0.94  # if age<=18
α12 = log(2.0906593164248873e7)*1.003                # if educ >=13
α13 = -log(1.2709425369629218e-6) *0.990               # if age>=30
α14 = log(3.369199198084097e6)*1.04

α30study = -log(1.4050382321279654e6)*1.18

probability_get_one_year = 0



Params=[
    ω1T1, ω1T2, ω1T3, α11, α12, α13, α14,
    ω2T1, ω2T2, ω2T3,
    α21, tc1T1, tc2, α22, α23, α25, α30study,
    α3, ω3T1, ω3T2, ω3T3, α31, α32, α33, α34, α35, α36, α37, α38,
        ω4T1, ω4T2, ω4T3, α41, α42, α43, α44, α45, α46, α47, α48,
    α50, α51, α52,
    σ1, σ2, σ3, σ4, σ34 ,σ5,
    πE1T1exp, πE1T2exp,
    πE2T1exp, πE2T2exp, probability_get_one_year
];


