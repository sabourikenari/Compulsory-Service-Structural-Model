


################################################################################
#= Define estimation Function =#

function estimation(params,
    choiceMomentData, wageMomentData, educatedShareData, transMoment)


    #=****************************************************=#
    #= parameters =#

    ω1T1, ω1T2, ω1T3, α11, α12, α13, α14,
    ω2T1, ω2T2, ω2T3,
    α21, tc1, tc2, α22, α23, α25, α30study,
    α3, ω3T1, ω3T2, ω3T3, α31, α32, α33, α34, α35, α36, α37, α38,
        ω4T1, ω4T2, ω4T3, α41, α42, α43, α44, α45, α46, α47, α48,
    α50, α51, α52,
    σ1, σ2, σ3, σ4, σ34 ,σ5,
    πE1T1exp, πE1T2exp,
    πE2T1exp, πE2T2exp               = params


    #=****************************************************=#
    α21 = exp(α21)
    tc1 = exp(tc1)
    tc2 = exp(tc2)
    α3 = exp(α3)
    α50 = exp(α50)
    α51 = exp(α51)
    α52 = exp(α52)
    σ1 = exp(σ1)
    σ2 = exp(σ2)
    σ5 = exp(σ5)
    α11 = -exp(-α11)
    α12 = exp(α12)
    α13 = exp(α13)
    α30study = -exp(-α30study)
    α14 = exp(α14)

    #=
    Some parameters are passed to the estimation function in logarithm scale,
    this is just for easier interpretion of paramaters.
    =#
    ω1T1 = exp(ω1T1)  ;   # the intercept of staying home α10 for type 1
    ω1T2 = exp(ω1T2)  ;   # the intercept of staying home α10 for type 2
    ω1T3 = exp(ω1T3)  ;   # the intercept of staying home α10 for type 3

    ω2T1 = exp(ω2T1) ;   # the intercept of studying for type 1
    ω2T2 = exp(ω2T2) ;   # the intercept of studying for type 2
    ω2T3 = exp(ω2T3) ;   # the intercept of studying for type 3

    #=****************************************************=#
    #= check the validity of the input parameters =#

    πE1T1 = exp(πE1T1exp)/(exp(πE1T1exp)+exp(πE1T2exp)+1)
    πE1T2 = exp(πE1T2exp)/(exp(πE1T1exp)+exp(πE1T2exp)+1)
    πE1T3 = exp(0)/(exp(πE1T1exp)+exp(πE1T2exp)+1)

    πE2T1 = exp(πE2T1exp)/(exp(πE2T1exp)+exp(πE2T2exp)+1)
    πE2T2 = exp(πE2T2exp)/(exp(πE2T1exp)+exp(πE2T2exp)+1)
    πE2T3 = exp(0)/(exp(πE2T1exp)+exp(πE2T2exp)+1)


    # π1 = 0.70 ;     # share of individuals type 1
    π1T1exp = -log((1/(1-0.345))-1)
    π1T2exp = -log((1/(1-0.386))-1)
    π1T3exp = -log((1/(1-0.335))-1)

    # π1T1exp = -log((1/(1-0.415))-1)
    # π1T2exp = -log((1/(1-0.406))-1)
    # π1T3exp = -log((1/(1-0.375))-1)

    π1T1 = exp(π1T1exp) / (1+exp(π1T1exp))
    π1T2 = exp(π1T2exp) / (1+exp(π1T2exp))
    π1T3 = exp(π1T3exp) / (1+exp(π1T3exp))


    # # Counterfactual: For no conscription system uncomment four following lines
    # π1T1 = 1.0
    # π1T2 = 1.0
    # π1T3 = 1.0


    #= discount factor set outside the estimation process =#
    δ = 0.92 ;

    M = 250

    α4 = 0.0  ;  # non pecuniary utility of blue-collar asssumed zero

    #= We assume that tuition cost is equall for all 4 different types =#
    # tc1T2 = tc1T1
    # tc1T3 = tc1T1
    # tc1T4 = tc1T1

    #= We assume that high school graduation effect on skills and
    consequently wages are similar in white- and blue-collars occupations =#
    α24 = α22

    N = 100 * 1000 ;   # number of individual to simulate their behaviour

    #=
    share of each education level at 15 years old
    levels are 0, 5, 8, 10
    =#
    # educShare =   [0.029 ,0.198 ,0.241 ,0.542]
    educShare =   [0.022 ,0.152 ,0.210 ,0.616]

    x3Max = 30
    x4Max = 30
    homeSinceSchoolMax = 2

    p = (
        ω1       = (ω1T1,ω1T2,ω1T3),
        α11      = α11,
        α12      = α12,
        α13      = α13,
        α14      = α14,
        ω2       = (ω2T1,ω2T2,ω2T3),
        α21      = α21,
        tc1      = tc1,
        tc2      = tc2,
        α22      = α22,
        α23      = α23,
        α24      = α24,
        α25      = α25,
        α30study = α30study,
        α3       = α3,
        ω3       = (ω3T1,ω3T2,ω3T3),
        α31      = α31,
        α32      = α32,
        α33      = α33,
        α34      = α34,
        α35      = α35,
        α36      = α36,
        α37      = α37,
        α38      = α38,
        α4       = α4,
        ω4       = (ω4T1,ω4T2,ω4T3),
        α41      = α41,
        α42      = α42,
        α43      = α43,
        α44      = α44,
        α45      = α45,
        α46      = α46,
        α47      = α47,
        α48      = α48,
        α50      = α50,
        α51      = α51,
        α52      = α52,
        σ1       = σ1,
        σ2       = σ2,
        σ3       = σ3,
        σ4       = σ4,
        σ34      = σ34 ,
        σ5       = σ5,
        δ        = δ,
        x3Max    = x3Max,
        x4Max    = x4Max,
        MonteCarloCount = M,
        homeSinceSchoolMax = homeSinceSchoolMax,
    )

    # print(p,"\n")
    # print(p.homeSinceSchoolMax,"  sdf ds fdsf dsfds")
    #=
    Save the result in a csv file
    this helps when the optimization is running on the server
    to catch the best candidater through run time easily
    however it makes a little inconsistecy, because Julia can not understand
    the type of input in compile time, but it does not make a trouble fro performance
    =#
    # bestResult = readdlm("/home/sabouri/Labor/CodeOutput/result.csv") ;
    # contributionsBest = readdlm("/home/sabouri/Labor/CodeOutput/contributionsBest.csv")


    # wrongParametersOutputForOptimizationContinue = Dict(
    #     "value"=> bestResult[1]*3.5*3.5,
    #     "root_contributions"=> contributionsBest.*3.5
    # )



    #=****************************************************=#
    #= solve the model =#

    #=     conscription goup 2     =#
    epsSolveMeanGroup2= [0.0, 0.0, 0.0, 0.0]
    epsSolveσGroup2= [ σ1   0.0  0.0   0.0 ;
                      0.0  σ2   0.0   0.0 ;
                      0.0  0.0  σ3    σ34 ;
                      0.0  0.0  σ34   σ4  ]

    #= check if the variance-covariance matrix is valid =#
    if !isposdef(epsSolveσGroup2)
        println("epsSolveσGroup2 : Wrong parameters were given as input!")
        return wrongParametersOutputForOptimizationContinue
        # return wrongParametersReturn
    end

    epssolveGroup2= rand(MersenneTwister(1234),
                        MvNormal(epsSolveMeanGroup2, epsSolveσGroup2), M) ;


    EmaxGroup2AllType = solveGroup2AllType(p, epssolveGroup2)


    #=     conscription goup 1     =#
    epsSolveMeanGroup1= [0.0, 0.0, 0.0, 0.0, 0.0] ;
    epsSolveσGroup1=[σ1   0.0  0.0  0.0  0.0 ;
                    0.0  σ2   0.0  0.0  0.0 ;
                    0.0  0.0  σ3   σ34  0.0 ;
                    0.0  0.0  σ34  σ4   0.0 ;
                    0.0  0.0  0.0  0.0  σ5  ] ;

    #= check if the variance-covariance matrix is valid =#
    if !isposdef(epsSolveσGroup1)
        println("epsSolveσGroup1 : Wrong parameters were given as input!")
        return wrongParametersOutputForOptimizationContinue
        # return wrongParametersReturn
    end

    epssolveGroup1= rand(MersenneTwister(4321),
                        MvNormal(epsSolveMeanGroup1, epsSolveσGroup1) , M) ;


    EmaxGroup1AllType =  solveGroup1AllType(p, epssolveGroup1) ;

    # EmaxGroup1AllType, EmaxGroup2AllType = solveAllGroupAllType(p,epssolveGroup1, epssolveGroup2)

    #=****************************************************=#
    #= simulate the model =#

    #= each column of simulated data is as follows: =#
    simCol = Dict(
        "age"      => 1,
        "educ"     => 2,
        "x3"       => 3,
        "x4"       => 4,
        "choice"   => 5,
        "income"   => 6,
        "educated" => 7,
        "x5"       => 8,
        "type"     => 9,
        "Emax"     => 10,
        "choice_next" => 11,
        "homeSinceSchool" => 12
    )
    p = merge(p, (simCol=simCol,))

    # πE1T4 = 1 - πE1T1 - πE1T2 - πE1T3
    # πE2T4 = 1 - πE2T1 - πE2T2 - πE2T3


    E1 = convert(Int, round(educShare[1]*N))
    E1T1 = convert(Int, round(πE1T1*E1))
    E1T2 = convert(Int, round(πE1T2*E1))
    E1T3 = E1 - E1T1 - E1T2

    E2 = convert(Int, round(educShare[2]*N))
    E2T1 = convert(Int, round(πE1T1*E2))
    E2T2 = convert(Int, round(πE1T2*E2))
    E2T3 = E2 - E2T1 - E2T2

    E3 = convert(Int, round(educShare[3]*N))
    E3T1 = convert(Int, round(πE1T1*E3))
    E3T2 = convert(Int, round(πE1T2*E3))
    E3T3 = E3 - E3T1 - E3T2

    E4 = N - E1 - E2 - E3
    E4T1 = convert(Int, round(πE2T1*E4))
    E4T2 = convert(Int, round(πE2T2*E4))
    E4T3 = E4 - E4T1 - E4T2



    weightsT1 = [
        E1T1*1.0,
        E2T1*1.0,
        E3T1*1.0,
        E4T1*1.0
    ]
    NGroup2T1 = convert(Int, round(sum(weightsT1) * π1T1))
    if NGroup2T1 > 0
        simGroup2T1= simulateGroup2(p, NGroup2T1, EmaxGroup2AllType, weightsT1; Seed=1111, type=1)
        simGroup2T1[:, simCol["type"]] .= 1
    else
        simGroup2T1 = Array{Float64,2}(undef,(0,12))
    end

    weightsT2 = [
        E1T2*1.0,
        E2T2*1.0,
        E3T2*1.0,
        E4T2*1.0
    ]
    NGroup2T2 = convert(Int, round(sum(weightsT2) * π1T2))
    if NGroup2T2 > 0
        simGroup2T2= simulateGroup2(p, NGroup2T2, EmaxGroup2AllType, weightsT2; Seed=2222, type=2)
        simGroup2T2[:, simCol["type"]] .= 2
    else
        simGroup2T2 = Array{Float64,2}(undef,(0,12))
    end

    weightsT3 = [
        E1T3*1.0,
        E2T3*1.0,
        E3T3*1.0,
        E4T3*1.0
    ]
    NGroup2T3 = convert(Int, round(sum(weightsT3) * π1T3))
    if NGroup2T3 > 0
        simGroup2T3= simulateGroup2(p, NGroup2T3, EmaxGroup2AllType, weightsT3; Seed=1345, type=3)
        simGroup2T3[:, simCol["type"]] .= 3
    else
        simGroup2T3 = Array{Float64,2}(undef,(0,12))
    end






    NGroup1T1 = E1T1+E2T1+E3T1+E4T1 - NGroup2T1
    if NGroup1T1 > 0
        simGroup1T1= simulateGroup1(p, NGroup1T1, EmaxGroup1AllType, weightsT1; Seed=3333, type=1)
        simGroup1T1[:, simCol["type"]] .= 1
    else
        simGroup1T1 = Array{Float64,2}(undef,(0,12))
    end
    NGroup1T2 = E1T2+E2T2+E3T2+E4T2 - NGroup2T2
    if NGroup1T2 > 0
        simGroup1T2= simulateGroup1(p, NGroup1T2, EmaxGroup1AllType, weightsT2; Seed=4444, type=2)
        simGroup1T2[:, simCol["type"]] .= 2
    else
        simGroup1T2 = Array{Float64,2}(undef,(0,12))
    end

    NGroup1T3 = E1T3+E2T3+E3T3+E4T3 - NGroup2T3
    if NGroup1T3 > 0
        simGroup1T3= simulateGroup1(p, NGroup1T3, EmaxGroup1AllType, weightsT3; Seed=5234, type=3)
        simGroup1T3[:, simCol["type"]] .= 3
    else
        simGroup1T3 = Array{Float64,2}(undef,(0,12))
    end


    #= Concatenate two simulation =#
    sim = [simGroup2T1; simGroup2T2; simGroup2T3;
           simGroup1T1; simGroup1T2; simGroup1T3 ] ;


    #=****************************************************=#
    #=
    Calculating moment from simulation
    we build some arrays to put data moment and model moment aside

    generating an Array named wageMoment
    to store average income of the simulated moments
    also we embed data moment in this array
    wageMomentData is the given wage moment from data
    =#
    wageMoment= Array{Float64,2}(undef, (size(wageMomentData,1),8))
    wageCol = Dict(
        "age"             => 1,
        # "educated"        => 2,
        "collar"          => 2,
        "incomeData"      => 3,
        "incomeStdBoot"   => 4,
        "devData"         => 5,
        "devStdBoot"      => 6,
        "incomeSim"       => 7,
        "devSim"          => 8
    )
    wageMoment[:,1:6]= wageMomentData[:,1:6];

    #=
    generating an Array named choiceMoment
    to store simulated share of alternatives
    also we embed data moment in this array
    choiceMomentData is the given alternative share moment from data
    =#
    choiceMoment= Array{Float64,2}(undef, (size(choiceMomentData,1),17) )
    choiceCol = Dict(
        "age"             => 1,
        "educated"        => 2,
        "homeData"        => 3,
        "studyData"       => 4,
        "whiteData"       => 5,
        "blueData"        => 6,
        "milData"         => 7,
        "homeStdBoot"     => 8,
        "studyStdBoot"    => 9,
        "whiteStdBoot"    => 10,
        "blueStdBoot"     => 11,
        "milStdBoot"      => 12,
        "homeSim"         => 13,
        "studySim"        => 14,
        "whiteSim"        => 15,
        "blueSim"         => 16,
        "milSim"          => 17
    )
    choiceMoment[ :, 1:12] =choiceMomentData[: ,1:12]

    #=
    removing share below 1 percent for two reason:
    it is not informative about distribution of choices
    also increases the error of coumputation daramatically large if they remain
    =#
    for i in 3:6
        choiceMoment[(choiceMoment[:,i].<0.005) ,i] .= NaN
    end

    #=
    generating an Array named educatedShare
    to store simulated share of educated individuals
    also we embed data moment in this array
    educatedShareData is the given educated share moment from data
    =#
    educatedCol = Dict(
        "age"                 => 1,
        "educatedData"        => 2,
        "educatedStdBoot"     => 3,
        "educatedSim"         => 4
    )
    educatedShare = Array{Float64,2}(undef, (size(educatedShareData,1), length(educatedCol)))
    educatedShare[:,1:3]= educatedShareData[:,1:3];
    [i for i in 16:20]

    transCol = Dict(
        "age"          => 1,
        "choice"       => 2,
        "choice_next"  => 3,
        "transData"    => 4,
        "transStdBoot" => 5,
        "transSim"     => 6
    )
    #=****************************************************=#
    #=
    sim is simulation of N people behaviour
    here we update data moment condition
    =#
    ageInterval= unique(choiceMoment[:,choiceCol["age"]])
    ageMax= maximum(ageInterval)

    for age in ageInterval

        #= mean income for each occupation moment condition =#

        # for educated in [0,1]#unique(wageMoment[ wageMoment[:,wageCol["age"]].== age , wageCol["educated"] ])
        #     for collar in unique(wageMoment[ wageMoment[:,wageCol["age"]].== age , wageCol["collar"] ])
        #
        #         #= amendment =#
        #         if age < 22
        #             educated = -1
        #         end
        #         #=
        #         mapping each collar code to choice alternative in the model
        #         in the file working with data, we defined:
        #         colar 0 : white-collar occupation
        #         colar 1 : blure-collar occupation
        #         colar 2 : compulsory military service
        #         =#
        #         if collar == 0
        #             choice= 3
        #         elseif collar==1
        #             choice= 4
        #         elseif collar==2
        #             choice= -10
        #         end
        #
        #         # flag = sim[ (sim[:,simCol["educated"]] .== convert(Int,educated) ).&
        #                     # (sim[:,simCol["choice"]].== choice) , simCol["income"]]
        #         flag = sim[ (sim[:,simCol["age"]].==age).&
        #                     (sim[:,simCol["choice"]].== choice).&
        #                     (sim[:,simCol["educated"]].==educated) , simCol["income"]]
        #
        #         if age<22
        #             educated = 0
        #         end
        #         wageMoment[ (wageMoment[:,wageCol["age"]].==age).&
        #                     (wageMoment[:,wageCol["collar"]].==collar).&
        #                     (wageMoment[:,wageCol["educated"]].==educated)
        #                     , wageCol["incomeSim"]] .= mean(filter(!isnan, flag ))
        #
        #         wageMoment[ (wageMoment[:,wageCol["age"]].==age).&
        #                     (wageMoment[:,wageCol["collar"]].==collar).&
        #                     (wageMoment[:,wageCol["educated"]].==educated)
        #                     , wageCol["devSim"]] .= std(filter(!isnan, flag ))
        #
        #     end #for collar
        # end #for educated

        for collar in unique(wageMoment[ wageMoment[:,wageCol["age"]].== age , wageCol["collar"] ])

            #=
            mapping each collar code to choice alternative in the model
            in the file working with data, we defined:
            colar 0 : white-collar occupation
            colar 1 : blure-collar occupation
            colar 2 : compulsory military service
            =#
            if collar == 0
                choice= 3
            elseif collar==1
                choice= 4
            elseif collar==2
                choice= -10
            end

            # flag = sim[ (sim[:,simCol["educated"]] .== convert(Int,educated) ).&
                        # (sim[:,simCol["choice"]].== choice) , simCol["income"]]
            flag = sim[ (sim[:,simCol["age"]].==age).&
                        (sim[:,simCol["choice"]].== choice), simCol["income"]]


            wageMoment[ (wageMoment[:,wageCol["age"]].==age).&
                        (wageMoment[:,wageCol["collar"]].==collar)
                        , wageCol["incomeSim"]] .= mean(filter(!isnan, flag ))

            wageMoment[ (wageMoment[:,wageCol["age"]].==age).&
                        (wageMoment[:,wageCol["collar"]].==collar)
                        , wageCol["devSim"]] .= std(filter(!isnan, flag ))

        end #for collar


        #= share of each alternative moment conditions =#

        for educated in
            convert.(
                Int,
                unique(choiceMoment[
                    choiceMoment[:, choiceCol["age"]].==age,
                    choiceCol["educated"],
                ]),
            )

            flag2 = [
                count(
                    x -> x == i,
                    sim[(sim[:, simCol["age"]].==age).&
                    (sim[:,simCol["educated"]].==educated)
                    , simCol["choice"]],
                    ) for i = 1:5
            ]

            choiceMoment[(choiceMoment[:, choiceCol["age"]].==age).&
                (choiceMoment[:,choiceCol["educated"]].==educated), choiceCol["homeSim"]:choiceCol["milSim"]] =
                flag2 / sum(flag2)

        end #educated


        #= share of educated people in each age between 24 and 32 =#
        if (age >= 24) & (age <= 32)

            flag = sim[ (sim[:,simCol["age"]].==age) , simCol["educated"]]

            educatedShare[(educatedShare[:,educatedCol["age"]] .== age),
                educatedCol["educatedSim"]] .= mean(filter(!isnan, flag ))
        end

        # #= transitoin rates =#
        # for choice in 1.0:5.0
        #     for choice_next in 1.0:5.0
        #
        #         flag = sim[ (sim[:,simCol["age"]].==age).&
        #                     (sim[:,simCol["choice"]].== choice) , :]
        #         count = length(flag)
        #         countNext = length(flag[(flag[:,simCol["choice_next"]] .== choice_next), :])
        #
        #         transMoment[(transMoment[:,transCol["age"]].==age).&
        #                     (transMoment[:,transCol["choice"]].==choice).&
        #                     (transMoment[:,transCol["choice_next"]].==choice_next),
        #                      transCol["transSim"]
        #         ] .= countNext / count
        #
        #
        #     end
        # end




    end#for age

    # for i = 8:12
    #     choiceMoment[ isnan.(choiceMoment[:,i]) , i ] .= 0
    # end


    #=****************************************************=#
    #=
    calculating error = sum squared of percentage distance
    between data moment and moment from model simulation
    =#
    contributions = [1.0]

    result, contributions= SMMCalculate(choiceMoment, wageMoment, educatedShare, transMoment,
            wageCol, choiceCol, educatedCol, transCol,
            contributions)


    # #=
    # Putting all moment in a vector for calculating jacobian of
    # the moment by changing parameters
    # =#
    # momentSim = [wageMoment[:,6] ; wageMoment[:,7]]
    #
    # for i = 8:12
    #     momentSim = [momentSim ; choiceMoment[:,i]]
    # end
    #
    # momentData = [wageMoment[:,4] ; wageMoment[:,5]]
    # for i = 3:7
    #     momentData = [momentData ; choiceMoment[:,i]]
    # end
    #
    # moment = momentSim-momentData # (momentSim-momentData)./momentData

    #=****************************************************=#
    #=
        Set some constraint for the moment estimated after age 36
        where we do not see the choices of men in the data for the
        specified cohort.
        1. share of men working in the white-collar occupations not
           far away from 0.12
    =#

    ##****************
    ConstraintError, contributions = constraintError(sim, simCol, contributions)
    result = result + ConstraintError

    contributions = contributions[2:end]
    # contributions = contributions[contributions.!=Inf]
    # contributions = contributions[contributions.!=NaN]
    replace!(contributions, Inf=>0)
    replace!(contributions, NaN=>0)
    result = sum(contributions.^2)

    #= return SMM error calculated =#
    print(" SMM error = ", round(result))
    writedlm( "data/simulation/simNewTest.csv",  sim, ',');
    # print(" SMM error = ", round(result), " Best Result: ",round(bestResult[1]))
    # print(contributions)

    #=****************************************************=#
    # if ENV["USER"]=="sabouri"
    #     if true #(result < bestResult[1])
    #     #= Server =#
    #     writedlm( "/home/sabouri/Labor/CodeOutput/simNew.csv",  sim, ',');
    #     # writedlm("/home/sabouri/Labor/CodeOutput/parameters.csv", params , ',') ;
    #     # writedlm("/home/sabouri/Labor/CodeOutput/result.csv", result , ',')     ;
    #     # writedlm("/home/sabouri/Labor/CodeOutput/contributionsBest.csv", contributions , ',') ;

    #     # writedlm("/home/sabouri/Labor/CodeOutput/transMomentxxxxx.csv",transMoment)

    #     # ***************************************************
    #     # send email after completing the optimization
    #     # opt = SendOptions(
    #     #   isSSL = true,
    #     #   username = "juliacodeserver@gmail.com",
    #     #   passwd = "JuliaCodeServer")
    #     # #Provide the message body as RFC5322 within an IO
    #     # body = IOBuffer(
    #     #   "Date: Fri, 18 Oct 2013 21:44:29 +0100\r\n" *
    #     #   "From: You <juliacodeserver@gmail.com>\r\n" *
    #     #   "To: ehsansaboori75@gmail.com\r\n" *
    #     #   "Subject: Julia Code\r\n" *
    #     #   "\r\n" *
    #     #   "Better solution found (: \r\n")
    #     # url = "smtps://smtp.gmail.com:465"
    #     # rcpt = ["<ehsansaboori75@gmail.com>"]
    #     # from = "<juliacodeserver@gmail.com>"
    #     # resp = send(url, rcpt, from, body, opt)
    #     else
    #         writedlm("/home/sabouri/Labor/CodeOutput/parametersLastIteration.csv", params , ',') ;
    #         # writedlm( "/home/sabouri/Labor/CodeOutput/simLastIteration.csv",  sim, ',');

    #     end
    # end


    out = Dict(
        "value"=> result,
        "root_contributions"=> contributions
    )
    return out
    # return result, contributions #, moment, momentData #, choiceMoment, wageMoment, sim
end

