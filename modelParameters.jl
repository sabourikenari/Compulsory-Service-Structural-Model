#=***************************************************

    The replication code for the follwoing research paper:

     "
        The Effect of Compulsory Military Service on Education and Income of Men in Iran;
        A Structural Model Estimation
     "

    Authors:
        Ehsan Sabouri Kenari
        Mohammad Hoseini

    Contanct us at:
        ehsansaboori75@gmail.com

****************************************************=#

#=
    Solving dynamic programming
    Two main groups of individuals:
    conscription group 2 : Not obligated to attend conscription
         Alternatives: 4 mutually exclusive choices
         choice 1 : stay home
         choice 2 : study
         choice 3 : white-collar occupation
         choice 4 : blue-collar occupation

    conscription group 1 : obligated to attend conscription
         Alternatives: 5 mutually exclusive choices
         choice 1 : stay home
         choice 2 : study
         choice 3 : white-collar occupation
         choice 4 : blue-collar occupation
         choice 5 : compulsory military service
=#




#= code for reading in server =#
if ENV["USER"] == "sabouri"
    wageMomentData= readdlm("/home/sabouri/Labor/DataMoments/wageMoment2.csv",',')      ;
    choiceMomentData = readdlm("/home/sabouri/Labor/DataMoments/choiceMoment2.csv",',') ;
    wageMomentStdBoot= readdlm("/home/sabouri/Labor/DataMoments/wageMomentStdBoot.csv",',')      ;
    choiceMomentStdBoot = readdlm("/home/sabouri/Labor/DataMoments/choiceMomentStdBoot.csv",',') ;
    educatedShareStdBoot = readdlm("/home/sabouri/Labor/DataMoments/educatedShareStdBoot.csv",',') ;

end
#= code for reading in Linux operating system =#
if ENV["USER"] == "ehsan"
    wageMomentData= readdlm("/home/ehsan/Dropbox/Labor/Codes/Moments/wageMoment2.csv",',')       ;
    choiceMomentData = readdlm("/home/ehsan/Dropbox/Labor/Codes/Moments/choiceMoment2.csv",',')  ;
    wageMomentStdBoot = readdlm("/home/ehsan/Dropbox/Labor/Codes/Moments/wageMomentStdBoot.csv",',')
    choiceMomentStdBoot = readdlm("/home/ehsan/Dropbox/Labor/Codes/Moments/choiceMomentStdBoot.csv",',')
    educatedShareStdBoot = readdlm("/home/ehsan/Dropbox/Labor/Codes/Moments/educatedShareStdBoot.csv",',')
end
#= code for reading in windows operating system =#
if ENV["USER"] == "claudioq"
    wageMomentData= readdlm("C:/Users/claudioq/Dropbox/Labor/Codes/Moments/wageMoment2.csv",',') ;
    choiceMomentData = readdlm("C:/Users/claudioq/Dropbox/Labor/Codes/Moments/choiceMoment2.csv",',') ;
end
#= reading Moment Data file on Dr. Hosseini computer =#
# wageMomentData = readdlm("C:/Users/Mohammad/Desktop/Moments/wageMoment.csv",',',Float64) ;
# choiceMomentData = readdlm("C:/Users/Mohammad/Desktop/Moments/choiceMoment.csv",',') ;




###############################################################################

# #= Initial parameters =#
#
# # parameters in the utility functions
# #**********************
# ω1T1 = 17.193643796698176      ;   # the intercept of staying home α10 for type 1
# ω1T2 = 17.10101202090928       ;   # the intercept of staying home α10 for type 2
# ω1T3 = 17.247155615996915      ;   # the intercept of staying home α10 for type 3
# ω1T4 = 17.116436018425216      ;   # the intercept of staying home α10 for type 4
#
# #**********************
# ω2T1 = 17.12341272473545      ;    # the intercept of studying for type 1
# ω2T2 = 17.790206579882735     ;    # the intercept of studying for type 2
# ω2T3 = 18.95548071836632      ;    # the intercept of studying for type 3
# ω2T4 = 18.71809027001068      ;    # the intercept of studying for type 4
#
# α21 = log(3.115121860959169e7)     ;    # study in (t-1)?
# tc1T1 = log(4.5553275303767666e7)    ;    # education >= 12?
# # tc1T2 = 4.5553275303767666e7    ;    # education >= 12?
# # tc1T3 = 4.5553275303767666e7    ;    # education >= 12?
# # tc1T4 = 4.553275303767666e7    ;    # education >= 12?
# tc2 = log(4.708012735120168e7)     ;    # education >= 16?
#
# α22 = 0.137 # reward of getting diploma
# α23 = 0.280 # reward of graduating college
#
# # α24 = 0.137 # reward of getting diploma
# α25 = 0.100 # reward of graduating college
#
#
# #**********************
# #= occupational choices: 3=white, 4=blue collar =#
# α3, α4 = log(2.912102156105642e6)   , 0 ;          # the intercept outside exp()
#
# #= the intercept inside exp() for type 1 =#
# ω3T1, ω4T1 = 14.923587474508264   , 16.68237380204532    ;
# #= the intercept inside exp() for type 2 =#
# ω3T2, ω4T2 = 14.36700982271307   , 15.993043719456187   ;
# #= the intercept inside exp() for type 3 =#
# ω3T3, ω4T3 = 15.149554695776371   , 16.341533053640374   ;
# #= the intercept inside exp() for type 4 =#
# ω3T4, ω4T4 = 15.081354895531176   , 16.736029404970487   ;
#
#
# #**********************
# #= share of each type for those education less than 10 in 15 years old =#
# πE1T1 = 0.7229226597006355
# πE1T2 = 0.200245804890741
# πE1T3 = 0.05042791889785734
# πE1T4 = 1- πE1T1- πE1T2- πE1T3
#
# den = 1/(1-0.7229226597006355-0.200245804890741-0.05042791889785734)
# πE1T1exp = log(den*0.7229226597006355)
# πE1T2exp = log(den*0.200245804890741)
# πE1T3exp = log(den*0.05042791889785734)
#
#
# #= share of each type for those education equalls 10 in 15 years old =#
# πE2T1 = 0.532182272493524
# πE2T2 = 0.21200626083052643
# πE2T3 = 0.121615000603791
# # πE2T4 = 1- πE2T1- πE2T2- πE2T3
#
# den = 1/(1-0.532182272493524-0.21200626083052643-0.1216150006037918)
# πE2T1exp = log(den*0.532182272493524)
# πE2T2exp = log(den*0.21200626083052643)
# πE2T3exp = log(den*0.1216150006037918)
#
#
#
# #**********************
# #= education coefficients =#
# α31, α41 =  0.13314223937325274 , 0.05543705296821224 ;
# #= experience in white collar =#
# α32, α42 = 0.09101988190579493 , 0.02939220222274944 ;
# #= experience in blue collar =#
# α33, α43 = 0.0200014722980203 , 0.1129179772059813 ;
# #= experience^2 in white collar =#
# α34, α44 = -0.0019514727935415903 ,-0.0021253464755022385 ;
# #= experience^2 in blue collar =#
# α35, α45 = -0.003269082102255282 , -0.002950986951463705 ;
#
# #= entry cost of without experience =#
# # α36, α46 = 0.0 , 0.0 ;
#
# #**********************
# α50 = 14.883024878263451 # intercept in util5 (conscription)
# α51 = log(4.1091249722694878e6) ;    # util5 coeff for if educ >= 12
# α52 = log(3.117584747501996e6) ;     # util5 coeff for if educ >= 16
#
# #**********************
# #= Variance-covariance of shocks =#
# σ1 = log(5.38353612340567e14) ;  # variance of ε1 - staying home
# σ2 = log(3.801914530676497e13) ;  # variance of ε2 - studying
# σ3 = 0.4980352741234879 ;    # variance of ε3 - white collar
# σ4 = 0.322421463218912 ;    # variance of ε4 - blue collar
# σ34 = 0.17007193198363868 ;    # Covariance of white and blue collar shocks
#
# σ5 = log(9.163008268122894e13) ;
#
# # π1 = 0.79 ;     # share of individuals type 1
# π1T1exp = -log((1/0.805)-1)
# π1T2exp = -log((1/0.835)-1)
# π1T3exp = -log((1/0.93)-1)
# π1T4exp = -log((1/0.93)-1)
#
#
#
# δ = 0.7937395498108646 ;      # discount factor
#
# #= New parameters in the model =#
# α11 = -log(6.2705530131153148e6)  # if age<=18
# α12 = log(1.38e7)                # if educ >=13
# α13 = -log(8.22e6)                # if age>=30
#
# α30study = -log(1.12e7)


#### ## ## #########################################################################













# parameters in the utility functions
#**********************
ω1T1 = 17.21236860868671     ;   # the intercept of staying home α10 for type 1
ω1T2 = 16.80220123598417       ;   # the intercept of staying home α10 for type 2
ω1T3 = 17.861115086202267      ;   # the intercept of staying home α10 for type 3
ω1T4 = 17.318684429739754      ;   # the intercept of staying home α10 for type 4

#**********************
ω2T1 = 16.32129586002007      ;    # the intercept of studying for type 1
ω2T2 = 18.137250774594587     ;    # the intercept of studying for type 2
ω2T3 = 19.377387203462717      ;    # the intercept of studying for type 3
ω2T4 = 19.738135594293446      ;    # the intercept of studying for type 4

α21 = log(8.451957709656079e7)     ;    # study in (t-1)?
tc1T1 = log(1.9908930961649176e8)    ;    # education >= 12?
# tc1T2 = 4.5553275303767666e7    ;    # education >= 12?
# tc1T3 = 4.5553275303767666e7    ;    # education >= 12?
# tc1T4 = 4.553275303767666e7    ;    # education >= 12?
tc2 = log(5.930208744183692e7)     ;    # education >= 16?

α22 = 0.10319803518449737 # reward of getting diploma
α23 = 0.1673486313229665 # reward of graduating college

# α24 = 0.137 # reward of getting diploma
α25 = 0.11018985798785595 # reward of graduating college


#**********************
#= occupational choices: 3=white, 4=blue collar =#
α3, α4 = log(3.5572583053788496e6)   , 0 ;          # the intercept outside exp()

#= the intercept inside exp() for type 1 =#
ω3T1, ω4T1 = 14.061564284889846   , 16.547926589269313    ;
#= the intercept inside exp() for type 2 =#
ω3T2, ω4T2 = 15.100369559956459   ,  15.993697465203543  ;
#= the intercept inside exp() for type 3 =#
ω3T3, ω4T3 = 14.738045918179062   , 16.012170936180958   ;
#= the intercept inside exp() for type 4 =#
ω3T4, ω4T4 = 14.720031826971303   , 16.290410696077926   ;


#**********************
#= share of each type for those education less than 10 in 15 years old =#
πE1T1 = 0.8057980638885157
πE1T2 = 0.082876300875887534
πE1T3 = 0.0509917859875172893
πE1T4 = 1- πE1T1- πE1T2- πE1T3

# den = 1/(1+πE1T1+πE1T2+πE1T3)
πE1T1exp = log(πE1T1/πE1T4)
πE1T2exp = log(πE1T2/πE1T4)
πE1T3exp = log(πE1T3/πE1T4)


#= share of each type for those education equalls 10 in 15 years old =#
πE2T1 = 0.5428268801717322
πE2T2 = 0.22049100659282447
πE2T3 = 0.11228664060393663
πE2T4 = 1- πE2T1- πE2T2- πE2T3

# den = 1/(1-0.5428268801717322-0.22049100659282447-0.11228664060393663)
πE2T1exp = log(πE2T1/πE2T4)
πE2T2exp = log(πE2T2/πE2T4)
πE2T3exp = log(πE2T3/πE2T4)
#


#**********************
#= education coefficients =#
α31, α41 =  0.12988121571957434 , 0.05875448225958381 ;
#= experience in white collar =#
α32, α42 = 0.09376990725285181 , 0.01713462654530423 ;
#= experience in blue collar =#
α33, α43 = 0.052302536609589 , 0.11392481791539259 ;
#= experience^2 in white collar =#
α34, α44 = -0.004345355193469356 , -0.006233553337469315 ;
#= experience^2 in blue collar =#
α35, α45 = -0.005891123236720835 , -0.0035162316835964857 ;

#= entry cost of without experience =#
# α36, α46 = 0.0 , 0.0 ;

#**********************
α50 = 15.969795443533842 # intercept in util5 (conscription)
α51 = log(1.313564992722831e6) ;    # util5 coeff for if educ >= 12
α52 = log(5.403804344709424e6) ;     # util5 coeff for if educ >= 16

#**********************
#= Variance-covariance of shocks =#
σ1 = log(1.0805430552997179e15) ;  # variance of ε1 - staying home
σ2 = log(8.277143696562533e13) ;  # variance of ε2 - studying
σ3 = 0.6089796764947927 ;    # variance of ε3 - white collar
σ4 = 0.3156394688613694 ;    # variance of ε4 - blue collar
σ34 = 0.182718181403573 ;    # Covariance of white and blue collar shocks

σ5 = log(1.238333261911126e15) ;

# π1 = 0.79 ;     # share of individuals type 1
π1T1exp = -log((1/0.8233144932575312)-1)
π1T2exp = -log((1/0.844424210654026)-1)
π1T3exp = -log((1/0.939382702067374)-1)
π1T4exp = -log((1/0.939880934193405)-1)


δ = 0.92 ;      # discount factor

#= New parameters in the model =#
α11 = -log(4.306492688224627e6)  # if age<=18
α12 = log(9.285576218815763e6)                # if educ >=13
α13 = -log(2.839815713751588e7)                # if age>=30

α30study = -log(0.8664274605780955e7)




Params=[ω1T1, ω1T2, ω1T3, ω1T4, α11, α12, α13 ,
        ω2T1, ω2T2, ω2T3, ω2T4,
        α21, tc1T1, tc2, α22, α23, α25, α30study,
        α3, ω3T1, ω3T2, ω3T3, ω3T4, α31, α32, α33, α34, α35,
            ω4T1, ω4T2, ω4T3, ω4T4, α41, α42, α43, α44, α45,
        α50, α51, α52,
        σ1, σ2, σ3, σ4, σ34 ,σ5,
        πE1T1exp, πE1T2exp, πE1T3exp,
        πE2T1exp, πE2T2exp, πE2T3exp,
        π1T1exp, π1T2exp, π1T3exp, π1T4exp  ]



# Params = readdlm("C:/Users/claudioq/Dropbox/Labor/Codes/parameters.csv")
# Params = readdlm("/home/sabouri/Labor/CodeOutput/parameters.csv")


# Params = [
#     16.917434228867574
#     16.483804629448578
#     17.633415603342822
#     17.47411781178098
#     -15.451601874903858
#     17.41742296264525
#     -17.40804154841029
#     16.355799355364777
#     17.89870543196756
#     19.656910332496498
#     20.022849206120572
#     18.491539613548454
#     19.277537271293195
#     18.1215937858313
#     0.12554399214151854
#     0.17047586448522004
#     0.11249468518233659
#     -16.58888334826892
#     14.914841035414884
#     14.01970094762609
#     15.070616530660272
#     14.693631327347475
#     14.682465789714316
#     0.12972719474327205
#     0.09079425893600641
#     0.053774475416476494
#     -0.00478455868782153
#     -0.006558055298671339
#     16.601590501593257
#     16.027132199241766
#     16.05239090725628
#     16.243956190798873
#     0.05848880067966815
#     0.018100566055436303
#     0.11419559465800871
#     -0.0059345391653508614
#     -0.0033499772461499776
#     16.200289812228966
#     16.627392865628675
#     17.60414716943579
#     34.558612791181986
#     31.737331116858716
#     0.6184611399162183
#     0.3209003249592074
#     0.18606655314124987
#     44.59614472430866
#     2.9058085872875528
#     0.3227428419803427
#     -0.16992981941227842
#     1.4952424048512816
#     0.5813461453721587
#     -0.10316498601791842
#     1.3650930856410735
#     1.6413023465850745
#     2.615046539268781
#     2.622772460026167
# ]
