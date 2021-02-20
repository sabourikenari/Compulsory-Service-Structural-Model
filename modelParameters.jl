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

#= Initial parameters =#

# parameters in the utility functions
#**********************
ω1T1 = 17.193643796698176      ;   # the intercept of staying home α10 for type 1
ω1T2 = 17.10101202090928       ;   # the intercept of staying home α10 for type 2
ω1T3 = 17.247155615996915      ;   # the intercept of staying home α10 for type 3
ω1T4 = 17.116436018425216      ;   # the intercept of staying home α10 for type 4

#**********************
ω2T1 = 17.12341272473545      ;    # the intercept of studying for type 1
ω2T2 = 17.790206579882735     ;    # the intercept of studying for type 2
ω2T3 = 18.95548071836632      ;    # the intercept of studying for type 3
ω2T4 = 18.71809027001068      ;    # the intercept of studying for type 4

α21 = log(3.115121860959169e7)     ;    # study in (t-1)?
tc1T1 = log(4.5553275303767666e7)    ;    # education >= 12?
# tc1T2 = 4.5553275303767666e7    ;    # education >= 12?
# tc1T3 = 4.5553275303767666e7    ;    # education >= 12?
# tc1T4 = 4.553275303767666e7    ;    # education >= 12?
tc2 = log(4.708012735120168e7)     ;    # education >= 16?

α22 = 0.137 # reward of getting diploma
α23 = 0.280 # reward of graduating college

# α24 = 0.137 # reward of getting diploma
α25 = 0.100 # reward of graduating college


#**********************
#= occupational choices: 3=white, 4=blue collar =#
α3, α4 = log(2.912102156105642e6)   , 0 ;          # the intercept outside exp()

#= the intercept inside exp() for type 1 =#
ω3T1, ω4T1 = 14.923587474508264   , 16.68237380204532    ;
#= the intercept inside exp() for type 2 =#
ω3T2, ω4T2 = 14.36700982271307   , 15.993043719456187   ;
#= the intercept inside exp() for type 3 =#
ω3T3, ω4T3 = 15.149554695776371   , 16.341533053640374   ;
#= the intercept inside exp() for type 4 =#
ω3T4, ω4T4 = 15.081354895531176   , 16.736029404970487   ;


#**********************
#= share of each type for those education less than 10 in 15 years old =#
πE1T1 = 0.7229226597006355
πE1T2 = 0.200245804890741
πE1T3 = 0.05042791889785734
πE1T4 = 1- πE1T1- πE1T2- πE1T3

den = 1/(1-0.7229226597006355-0.200245804890741-0.05042791889785734)
πE1T1exp = log(den*0.7229226597006355)
πE1T2exp = log(den*0.200245804890741)
πE1T3exp = log(den*0.05042791889785734)


#= share of each type for those education equalls 10 in 15 years old =#
πE2T1 = 0.532182272493524
πE2T2 = 0.21200626083052643
πE2T3 = 0.121615000603791
# πE2T4 = 1- πE2T1- πE2T2- πE2T3

den = 1/(1-0.532182272493524-0.21200626083052643-0.1216150006037918)
πE2T1exp = log(den*0.532182272493524)
πE2T2exp = log(den*0.21200626083052643)
πE2T3exp = log(den*0.1216150006037918)



#**********************
#= education coefficients =#
α31, α41 =  0.13314223937325274 , 0.05543705296821224 ;
#= experience in white collar =#
α32, α42 = 0.09101988190579493 , 0.02939220222274944 ;
#= experience in blue collar =#
α33, α43 = 0.0200014722980203 , 0.1129179772059813 ;
#= experience^2 in white collar =#
α34, α44 = -0.0019514727935415903 ,-0.0021253464755022385 ;
#= experience^2 in blue collar =#
α35, α45 = -0.003269082102255282 , -0.002950986951463705 ;

#= entry cost of without experience =#
# α36, α46 = 0.0 , 0.0 ;

#**********************
α50 = 14.883024878263451 # intercept in util5 (conscription)
α51 = log(4.1091249722694878e6) ;    # util5 coeff for if educ >= 12
α52 = log(3.117584747501996e6) ;     # util5 coeff for if educ >= 16

#**********************
#= Variance-covariance of shocks =#
σ1 = log(5.38353612340567e14) ;  # variance of ε1 - staying home
σ2 = log(3.801914530676497e13) ;  # variance of ε2 - studying
σ3 = 0.4980352741234879 ;    # variance of ε3 - white collar
σ4 = 0.322421463218912 ;    # variance of ε4 - blue collar
σ34 = 0.17007193198363868 ;    # Covariance of white and blue collar shocks

σ5 = log(9.163008268122894e13) ;

# π1 = 0.79 ;     # share of individuals type 1
π1T1exp = -log((1/0.805)-1)
π1T2exp = -log((1/0.835)-1)
π1T3exp = -log((1/0.93)-1)
π1T4exp = -log((1/0.93)-1)



δ = 0.7937395498108646 ;      # discount factor

#= New parameters in the model =#
α11 = -log(6.2705530131153148e6)  # if age<=18
α12 = log(1.38e7)                # if educ >=13
α13 = -log(8.22e6)                # if age>=30

α30study = -log(1.12e7)


#################################################################################







#
# # parameters in the utility functions
# #**********************
# ω1T1 = 17.821135744931293     ;   # the intercept of staying home α10 for type 1
# ω1T2 = 17.15004146644513       ;   # the intercept of staying home α10 for type 2
# ω1T3 = 17.246433733426983      ;   # the intercept of staying home α10 for type 3
# ω1T4 = 17.115490814239713      ;   # the intercept of staying home α10 for type 4
#
# #**********************
# ω2T1 = 17.119188980274988      ;    # the intercept of studying for type 1
# ω2T2 = 17.62777379230195     ;    # the intercept of studying for type 2
# ω2T3 = 18.92034595966366      ;    # the intercept of studying for type 3
# ω2T4 = 18.659296846027726      ;    # the intercept of studying for type 4
#
# α21 = log(3.1267153079689432e7)     ;    # study in (t-1)?
# tc1T1 = log(5.584834912170728e7)    ;    # education >= 12?
# # tc1T2 = 4.5553275303767666e7    ;    # education >= 12?
# # tc1T3 = 4.5553275303767666e7    ;    # education >= 12?
# # tc1T4 = 4.553275303767666e7    ;    # education >= 12?
# tc2 = log(4.691439603355582e7)     ;    # education >= 16?
#
# α22 = 0.12733868752138708 # reward of getting diploma
# α23 = 0.2690049377167906 # reward of graduating college
#
# # α24 = 0.137 # reward of getting diploma
# α25 = 0.09749960712781927 # reward of graduating college
#
#
# #**********************
# #= occupational choices: 3=white, 4=blue collar =#
# α3, α4 = log(2.9093899068247094e6)   , 0 ;          # the intercept outside exp()
#
# #= the intercept inside exp() for type 1 =#
# ω3T1, ω4T1 = 14.983322039452377   , 16.781267388567404    ;
# #= the intercept inside exp() for type 2 =#
# ω3T2, ω4T2 = 14.45389803802563   ,  15.996931754009072  ;
# #= the intercept inside exp() for type 3 =#
# ω3T3, ω4T3 = 15.149883841504563   , 16.340254459742845   ;
# #= the intercept inside exp() for type 4 =#
# ω3T4, ω4T4 = 15.152167145692376   , 16.79930212229079   ;
#
#
# #**********************
# #= share of each type for those education less than 10 in 15 years old =#
# πE1T1 = 0.7220606259286249
# πE1T2 = 0.20108355034376238
# πE1T3 = 0.05046766358080422
# πE1T4 = 1- πE1T1- πE1T2- πE1T3
#
# # den = 1/(1+πE1T1+πE1T2+πE1T3)
# πE1T1exp = log(πE1T1/πE1T4)
# πE1T2exp = log(πE1T2/πE1T4)
# πE1T3exp = log(πE1T3/πE1T4)
#
#
# #= share of each type for those education equalls 10 in 15 years old =#
# πE2T1 = 0.530882517036717
# πE2T2 = 0.21216088412560616
# πE2T3 = 0.12239449348585607
# πE2T4 = 1- πE2T1- πE2T2- πE2T3
#
# # den = 1/(1-0.530882517036717-0.21216088412560616-0.12239449348585607)
# πE2T1exp = log(πE2T1/πE2T4)
# πE2T2exp = log(πE2T2/πE2T4)
# πE2T3exp = log(πE2T3/πE2T4)
#
#
#
# #**********************
# #= education coefficients =#
# α31, α41 =  0.12216226807760436 , 0.02841539777375748 ;
# #= experience in white collar =#
# α32, α42 = 0.08672667020848523 , 0.03265215947361173 ;
# #= experience in blue collar =#
# α33, α43 = 0.019363099917443486 , 0.10915693436623712 ;
# #= experience^2 in white collar =#
# α34, α44 = -0.0032402574017335382 , -0.001988432162186431 ;
# #= experience^2 in blue collar =#
# α35, α45 = -0.00012359503531649587 , -0.002015387979204839 ;
#
# #= entry cost of without experience =#
# # α36, α46 = 0.0 , 0.0 ;
#
# #**********************
# α50 = 14.882579733172172 # intercept in util5 (conscription)
# α51 = log(4.118803655369918e6) ;    # util5 coeff for if educ >= 12
# α52 = log(3.1215552728984714e6) ;     # util5 coeff for if educ >= 16
#
# #**********************
# #= Variance-covariance of shocks =#
# σ1 = log(5.4083437880157475e14) ;  # variance of ε1 - staying home
# σ2 = log(3.805309933866672e13) ;  # variance of ε2 - studying
# σ3 = 0.4999282805508579 ;    # variance of ε3 - white collar
# σ4 = 0.31983281415948445 ;    # variance of ε4 - blue collar
# σ34 = 0.1713998915956022 ;    # Covariance of white and blue collar shocks
#
# σ5 = log(9.130083112135336e13) ;
#
# # π1 = 0.79 ;     # share of individuals type 1
# π1T1exp = -log((1/0.8546724633161266)-1)
# π1T2exp = -log((1/0.8651298737968329)-1)
# π1T3exp = -log((1/0.870033456338808)-1)
# π1T4exp = -log((1/0.8801452913294438)-1)
#
#
# #= New parameters in the model =#
# α11 = -log(6.229194436639572e6)  # if age<=18
# α12 = log(1.3840258447282169e7)                # if educ >=13
# α13 = -log(8.239259299270849e6)                # if age>=30
#
# α30study = -log(1.1187216090761894e7)





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






#=
    the best parameters optimized when initial vector of Parameter
    was the optimized with delta equal 0.78
=#
# Params = [
#     17.82168315153809
#     17.15007926098177
#     17.24660614265214
#     17.115606952136623
#     -15.644850436651897
#     16.44322704410178
#     -15.923724809108771
#     17.119574531702852
#     17.627845735792466
#     18.920259656725246
#     18.659213735633564
#     17.258472655068033
#     17.838066847173735
#     17.66355390495239
#     0.12718417241481733
#     0.26916137433720455
#     0.09752163341530008
#     -16.23063772725193
#     14.883772885750753
#     14.923302437684791
#     14.363700966583844
#     15.150113687028343
#     15.082349325079504
#     0.12281107524707463
#     0.0871417715046926
#     0.021861241963971455
#     -0.002677506795761851
#     -0.0002036184899365944
#     16.68128035328377
#     15.996641609141951
#     16.34000927281364
#     16.759299850287857
#     0.029006432014350637
#     0.033300794963460785
#     0.11214875796170792
#     -0.0013317696666555565
#     -0.0018932366157536854
#     14.882694472771657
#     15.230986471341183
#     14.953714949738638
#     33.92443363327087
#     31.270017174539657
#     0.49978989614415165
#     0.3195046479204882
#     0.17100432028454915
#     32.14521926971432
#     3.309083856281416
#     2.030967442605618
#     0.6485283917813318
#     1.3722584515561207
#     0.45557110085179653
#     -0.09480332566990823
#     1.4156660852097778
#     1.6225046279245914
#     2.5870214488750594
#     2.5891070900286084
# ]



#=
    the best found from changing the best result from first draf (:
=#
Params = [
17.821135744931293
17.15004146644513
17.246433733426983
17.115490814239713
-15.644757578492984
16.443092181916015
-15.92442100698004
17.119188980274988
17.62777379230195
18.92034595966366
18.659296846027726
17.25807868210704
17.838150523896807
17.663835137991445
0.12733868752138708
0.2690049377167906
0.09749960712781927
-16.230282263881172
14.883453963155073
14.983322039452377
14.45389803802563
15.149883841504563
15.152167145692376
0.12216226807760436
0.08672667020848523
0.021363099917443486
-0.0032402574017335382
0.00012359503531649587
16.781267388567404
15.996931754009072
16.340254459742845
16.79930212229079
0.02841539777375748
0.03265215947361173
0.10915693436623712
-0.001988432162186431
-0.002015387979204839
14.882579733172172
15.231073304248538
14.953841920486695
33.92413420890598
31.27000365128608
0.4999282805508579
0.31983281415948445
0.1713998915956022
32.1451810066792
3.309193674530016
2.0308050647750893
0.6484173757852266
1.372514905948539
0.4553190319516045
-0.09477646109032327
1.6934129075613464
1.7754156968779518
2.587203379959395
2.588923306575371
]

#= ***************************************************** =#

function ParametersOutput(Params)

    ω1T1, ω1T2, ω1T3, ω1T4, α11, α12, α13 ,
            ω2T1, ω2T2, ω2T3, ω2T4,
            α21, tc1T1, tc2, α22, α23, α25, α30study,
            α3, ω3T1, ω3T2, ω3T3, ω3T4, α31, α32, α33, α34, α35,
                ω4T1, ω4T2, ω4T3, ω4T4, α41, α42, α43, α44, α45,
            α50, α51, α52,
            σ1, σ2, σ3, σ4, σ34 ,σ5,
            πE1T1exp, πE1T2exp, πE1T3exp,
            πE2T1exp, πE2T2exp, πE2T3exp,
            π1T1exp, π1T2exp, π1T3exp, π1T4exp  = Params


    string = """ω1T1 = $ω1T1
    ω1T2 = $ω1T2
    ω1T3 = $ω1T3
    ω1T4 = $ω1T4
    α11 = $α11
    α12 = $α12
    α13 = $α13
    ω2T1 = $ω2T1
    ω2T2 = $ω2T2
    ω2T3 = $ω2T3
    ω2T4 = $ω2T4
    α21 = $α21
    tc1T1 = $tc1T1
    tc2 = $tc2
    α22 = $α22
    α23 = $α23
    α25 = $α25
    α30study = $α30study
    α3 = $α3
    ω3T1 = $ω3T1
    ω3T2 = $ω3T2
    ω3T3 = $ω3T3
    ω3T4 = $ω3T4
    α31 = $α31
    α32 = $α32
    α33 = $α33
    α34 = $α34
    α35 = $α35
    ω4T1 = $ω4T1
    ω4T2 = $ω4T2
    ω4T3 = $ω4T3
    ω4T4 = $ω4T4
    α41 = $α41
    α42 = $α42
    α43 = $α43
    α44 = $α44
    α45 = $α45
    α50 = $α50
    α51 = $α51
    α52 = $α52
    σ1 = $σ1
    σ2 = $σ2
    σ3 = $σ3
    σ4 = $σ4
    σ34 = $σ34
    σ5 = $σ5
    πE1T1exp = $πE1T1exp
    πE1T2exp = $πE1T2exp
    πE1T3exp = $πE1T3exp
    πE2T1exp = $πE2T1exp
    πE2T2exp = $πE2T2exp
    πE2T3exp = $πE2T3exp
    π1T1exp = $π1T1exp
    π1T2exp = $π1T2exp
    π1T3exp = $π1T3exp
    π1T4exp = $π1T4exp """

    print(string)
end





function ParametersWide(Params)
    ω1T1, ω1T2, ω1T3, ω1T4, α11, α12, α13 ,
            ω2T1, ω2T2, ω2T3, ω2T4,
            α21, tc1T1, tc2, α22, α23, α25, α30study,
            α3, ω3T1, ω3T2, ω3T3, ω3T4, α31, α32, α33, α34, α35,
                ω4T1, ω4T2, ω4T3, ω4T4, α41, α42, α43, α44, α45,
            α50, α51, α52,
            σ1, σ2, σ3, σ4, σ34 ,σ5,
            πE1T1exp, πE1T2exp, πE1T3exp,
            πE2T1exp, πE2T2exp, πE2T3exp,
            π1T1exp, π1T2exp, π1T3exp, π1T4exp  = Params

    α21 = exp(α21)
    tc1T1 = exp(tc1T1)
    tc2 = exp(tc2)
    α3 = exp(α3)
    α51 = exp(α51)
    α52 = exp(α52)
    σ1 = exp(σ1)
    σ2 = exp(σ2)
    σ5 = exp(σ5)
    α11 = exp(-α11)
    α12 = exp(α12)
    α13 = exp(-α13)
    α30study = exp(-α30study)

    πE1T1 = exp(πE1T1exp)/(exp(πE1T1exp)+exp(πE1T2exp)+exp(πE1T3exp)+1)
    πE1T2 = exp(πE1T2exp)/(exp(πE1T1exp)+exp(πE1T2exp)+exp(πE1T3exp)+1)
    πE1T3 = exp(πE1T3exp)/(exp(πE1T1exp)+exp(πE1T2exp)+exp(πE1T3exp)+1)
    πE1T4 = exp(0)/(exp(πE1T1exp)+exp(πE1T2exp)+exp(πE1T3exp)+1)

    πE2T1 = exp(πE2T1exp)/(exp(πE2T1exp)+exp(πE2T2exp)+exp(πE2T3exp)+1)
    πE2T2 = exp(πE2T2exp)/(exp(πE2T1exp)+exp(πE2T2exp)+exp(πE2T3exp)+1)
    πE2T3 = exp(πE2T3exp)/(exp(πE2T1exp)+exp(πE2T2exp)+exp(πE2T3exp)+1)
    πE2T4 = exp(0)/(exp(πE2T1exp)+exp(πE2T2exp)+exp(πE2T3exp)+1)

    π1T1 = exp(π1T1exp) / (1+exp(π1T1exp))
    π1T2 = exp(π1T2exp) / (1+exp(π1T2exp))
    π1T3 = exp(π1T3exp) / (1+exp(π1T3exp))
    π1T4 = exp(π1T4exp) / (1+exp(π1T4exp))


    output = """
    # parameters in the utility functions
    #**********************
    ω1T1 = $ω1T1     ;   # the intercept of staying home α10 for type 1
    ω1T2 = $ω1T2       ;   # the intercept of staying home α10 for type 2
    ω1T3 = $ω1T3      ;   # the intercept of staying home α10 for type 3
    ω1T4 = $ω1T4      ;   # the intercept of staying home α10 for type 4

    #**********************
    ω2T1 = $ω2T1      ;    # the intercept of studying for type 1
    ω2T2 = $ω2T2     ;    # the intercept of studying for type 2
    ω2T3 = $ω2T3      ;    # the intercept of studying for type 3
    ω2T4 = $ω2T4      ;    # the intercept of studying for type 4

    α21 = log($α21)     ;    # study in (t-1)?
    tc1T1 = log($tc1T1)    ;    # education >= 12?
    # tc1T2 = 4.5553275303767666e7    ;    # education >= 12?
    # tc1T3 = 4.5553275303767666e7    ;    # education >= 12?
    # tc1T4 = 4.553275303767666e7    ;    # education >= 12?
    tc2 = log($tc2)     ;    # education >= 16?

    α22 = $α22 # reward of getting diploma
    α23 = $α23 # reward of graduating college

    # α24 = 0.137 # reward of getting diploma
    α25 = $α25 # reward of graduating college


    #**********************
    #= occupational choices: 3=white, 4=blue collar =#
    α3, α4 = log($α3)   , 0 ;          # the intercept outside exp()

    #= the intercept inside exp() for type 1 =#
    ω3T1, ω4T1 = $ω3T1   , $ω4T1    ;
    #= the intercept inside exp() for type 2 =#
    ω3T2, ω4T2 = $ω3T2   ,  $ω4T2  ;
    #= the intercept inside exp() for type 3 =#
    ω3T3, ω4T3 = $ω3T3   , $ω4T3   ;
    #= the intercept inside exp() for type 4 =#
    ω3T4, ω4T4 = $ω3T4   , $ω4T4   ;


    #**********************
    #= share of each type for those education less than 10 in 15 years old =#
    πE1T1 = $πE1T1
    πE1T2 = $πE1T2
    πE1T3 = $πE1T3
    πE1T4 = 1- πE1T1- πE1T2- πE1T3

    # den = 1/(1+πE1T1+πE1T2+πE1T3)
    πE1T1exp = log(πE1T1/πE1T4)
    πE1T2exp = log(πE1T2/πE1T4)
    πE1T3exp = log(πE1T3/πE1T4)


    #= share of each type for those education equalls 10 in 15 years old =#
    πE2T1 = $πE2T1
    πE2T2 = $πE2T2
    πE2T3 = $πE2T3
    πE2T4 = 1- πE2T1- πE2T2- πE2T3

    # den = 1/(1-$πE2T1-$πE2T2-$πE2T3)
    πE2T1exp = log(πE2T1/πE2T4)
    πE2T2exp = log(πE2T2/πE2T4)
    πE2T3exp = log(πE2T3/πE2T4)



    #**********************
    #= education coefficients =#
    α31, α41 =  $α31 , $α41 ;
    #= experience in white collar =#
    α32, α42 = $α32 , $α42 ;
    #= experience in blue collar =#
    α33, α43 = $α33 , $α43 ;
    #= experience^2 in white collar =#
    α34, α44 = $α34 , $α44 ;
    #= experience^2 in blue collar =#
    α35, α45 = $α35 , $α45 ;

    #= entry cost of without experience =#
    # α36, α46 = 0.0 , 0.0 ;

    #**********************
    α50 = $α50 # intercept in util5 (conscription)
    α51 = log($α51) ;    # util5 coeff for if educ >= 12
    α52 = log($α52) ;     # util5 coeff for if educ >= 16

    #**********************
    #= Variance-covariance of shocks =#
    σ1 = log($σ1) ;  # variance of ε1 - staying home
    σ2 = log($σ2) ;  # variance of ε2 - studying
    σ3 = $σ3 ;    # variance of ε3 - white collar
    σ4 = $σ4 ;    # variance of ε4 - blue collar
    σ34 = $σ34 ;    # Covariance of white and blue collar shocks

    σ5 = log($σ5) ;

    # π1 = 0.79 ;     # share of individuals type 1
    π1T1exp = -log((1/$π1T1)-1)
    π1T2exp = -log((1/$π1T2)-1)
    π1T3exp = -log((1/$π1T3)-1)
    π1T4exp = -log((1/$π1T4)-1)


    #= New parameters in the model =#
    α11 = -log($α11)  # if age<=18
    α12 = log($α12)                # if educ >=13
    α13 = -log($α13)                # if age>=30

    α30study = -log($α30study)

    """

    print("\n\n\n",output)
end




# #= last estimated parameter that is in the latex report =#
#
# Params = [
# 17.206210410475922
# 16.80100236132292
# 17.845861390236394
# 17.39358653667415
# -15.27987838574698
# 15.803763734603676
# -16.325474365196612
# 16.12457636690639
# 18.034200729767793
# 19.37158270996246
# 19.51895799429412
# 18.254132971155958
# 19.121830152254837
# 18.007441620639526
# 0.08884414372265025
# 0.15605743021706184
# 0.10262473045850386
# -15.979217275160869
# 15.087580455904826
# 14.060415350108283
# 15.103207892523105
# 14.84526014652997
# 15.10029646125166
# 0.12510480220108106
# 0.09769641152996936
# 0.05697138338811244
# -0.0034315012779852086
# -0.00609999092194062
# 16.555366133312653
# 15.996091476794433
# 16.063528415111286
# 16.527716929179867
# 0.05837921241858132
# 0.014457002818340598
# 0.11401686720491694
# -0.006195809800063991
# -0.002988104177238432
# 15.965528634915069
# 16.3959001959431
# 15.501101446915408
# 34.61421572341715
# 32.04664948154755
# 0.6158522459978923
# 0.31519634955735093
# 0.17540348640347714
# 34.75747729938343
# 2.5991609189280798
# 0.31580968464617004
# -0.16162980840009109
# 1.4739172889195529
# 0.5716440914037556
# 0.004417678382582084
# 1.8501419314430443
# 1.9465414469678233
# 2.745313182766822
# 2.751444085270356
# ]
