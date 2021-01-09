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
    The codes needed for running on the server
    and also extracting the simulation results from Server
    to my ubuntu operating system.
=#

# include("/home/sabouri/thesis/code,version11_Parallel.jl")
# scp sabouri@192.168.84.5:/home/sabouri/thesis/moments/data/wageMoment.csv /home/ehsan/Dropbox/Labor/Codes/Moments/data/
# scp sabouri@192.168.84.5:/home/sabouri/thesis/moments/data/choiceMoment.csv /home/ehsan/Dropbox/Labor/Codes/Moments/data/


#=
    Initialize the result on the hard drive

    In the estimation process and in each iteration of optimization,
    each time that a SMM error is calculated for a specific parameters,
    it will be compared to the best resul, Then if the result is better than
    the previous ones, the final result will be replaced and aslo the
    parameres will be saved.

    Note. this is only for avoiding the loss of results during the optimization
    due to computer shutdown, error in the code and etc.
=#

using DelimitedFiles
result = 1.0e50
if ENV["USER"] == "sabouri"
    writedlm("/home/sabouri/Labor/CodeOutput/result.csv", result )
end

#################################################################################
#=
    Solving dynamic programming
    Two main groups of individuals:
    conscription group 1 : Not obligated to attend conscription
         Alternatives: 4 mutually exclusive choices
         choice 1 : stay home
         choice 2 : study
         choice 3 : white-collar occupation
         choice 4 : blue-collar occupation

    conscription group 2 : obligated to attend conscription
         Alternatives: 5 mutually exclusive choices
         choice 1 : stay home
         choice 2 : study
         choice 3 : white-collar occupation
         choice 4 : blue-collar occupation
         choice 5 : compulsory military service
=#

using Pkg

using Random
# using Statistics
# using BenchmarkTools
# using Profilex
using Distributions
# using LinearAlgebra

using StatsBase

# using Cubature

using DelimitedFiles

using NamedArrays

using Dates

using Optim

# using NLopt

using BlackBoxOptim

using Distributions

using Compat.Dates

using SharedArrays

using LinearAlgebra

# using Test

if ENV["USER"] == "sabouri"
    using SMTPClient # for sending email
end

using Distributed

#= Initialize cpu workers for computation =#
if ENV["USER"] == "sabouri"
    addprocs(4)
elseif ENV["USER"] == "ehsan"
    addprocs(2)
end
println("worker cores are: " , workers())


################################################################################
#= contemporaneous utility function =#

#= utility when choice is stay home =#
@everywhere function util1(α10, α11, α12, α13, age, educ, ε1)
    util=  α10 .+ α11*(age <= 19).+ α12*(educ>=13) .+ α13*(age>=30) .+ ε1
    return util
end

#= utility when choice is study =#
@everywhere function util2(α20, α21, tc1, tc2, sl, educ, ε2, age, α30study)
    util= (α20 - α21*(sl == 0)- tc1*(educ>12)- tc2*(educ>16) + α30study*(age>=30) ) .+ ε2
    return util
end


#= utility when choice is whitel-collar occupation =#
@everywhere function util3(α3, α30, α31, α32, α33, α34, α35, α36, x3, x4, educ, ε3, α22, α23)
    util= (exp.((α30+ α31*educ+ α32*x3+ α33*x4+ α34*(x3^2)+ α35*(x4^2))+ α36*(x3==0)+ α22*(educ>=12)+ α23*(educ>=16).+ ε3) .+ α3)
    return util
end

#= utility when choice is blue-collar occupation =#
@everywhere function util4(α4, α40, α41, α42, α43, α44, α45, α46, x3, x4, educ, ε4, α24, α25)
    util= (exp.((α40+ α41*educ+ α42*x3+ α43*x4+ α44*(x3^2)+ α45*(x4^2))+ α46*(x4==0)+ α24*(educ>=12)+ α25*(educ>=16).+ ε4) .+ α4)
    return util
end

#= utility when choice is compulsory military service =#
@everywhere function util5(α50, α51, α52, educ, ε5)
    util= α50 + α51*(educ>12) + α52*(educ>16) .+ ε5
    return util
end

################################################################################
#=
    conscription group 1 value function and solve Emax function
    group 1: Not obligated to attend conscription
    value function: given state vector at an age, it denotes the maxiual value
    at age a over all possible career decisions.
=#

#= value function for type 1: Not obligated to attent conscription =#
@everywhere function valueFunctionGroup1(α10, α11, α12, α13,
                α20, α21, tc1, tc2, α22, α23, α24, α25, α30study,
                α3, α30, α31, α32, α33, α34, α35, α36,
                α4, α40, α41, α42, α43, α44, α45, α46,
                δ,
                epssolve,
                age, educ, sl, x3, x4,
                Emax)

    x3Max = 30
    x4Max = 30

    ε1=epssolve[1,:]
    u1= util1(α10, α11, α12, α13, age, educ, ε1)

    ε2=epssolve[2,:]
    u2= util2(α20, α21, tc1, tc2, sl, educ+1, ε2, age, α30study)

    ε3=epssolve[3,:]
    u3= util3(α3, α30, α31, α32, α33, α34, α35, α36, x3, x4, educ, ε3, α22, α23)

    ε4=epssolve[4,:]
    u4= util4(α4, α40, α41, α42, α43, α44, α45, α46, x3, x4, educ, ε4, α24, α25)


    value= -1 # this is for when no if conditon binds
    if age == 65

        if educ < 22
            s=0.0
            @simd for i in 1:length(u1)
                s+=max(u1[i], u2[i], u3[i], u4[i])
            end
            value= s/length(u1) #mean(max(u1,u2,u3,u4))
        else
            s=0.0
            @simd for i in 1:length(u1)
                s+=max(u1[i], u3[i], u4[i])
            end
            value= s/length(u1)#mean(max(u1,u2,u4))
        end
    else

        # begin inbounds
        # assume that maximum amount of experience is 30 years.
        @inbounds u1= u1 .+ δ*Emax[age-17+1+1, educ+1,             0+1,x3+1  ,x4+1 ]
        @inbounds u2= u3 .+ δ*Emax[age-17+1+1, educ+1+1*(educ!=22),1+1,x3+1  ,x4+1 ]
        @inbounds u3= u3 .+ δ*Emax[age-17+1+1, educ+1,             0+1,x3+1+1*(x3!=x3Max),x4+1]
        @inbounds u4= u4 .+ δ*Emax[age-17+1+1, educ+1,             0+1,x3+1,x4+1+1*(x4!=x4Max)]
        # end

        if educ < 22
            s=0.0
            @simd for i in 1:length(u1)
                s+=max(u1[i], u2[i], u3[i], u4[i])
            end
            value= s/length(u1) #mean(max(u1,u2,u3,u4))
        else
            s=0.0
            @simd for i in 1:length(u1)
                s+=max(u1[i], u3[i], u4[i])
            end
            value= s/length(u1) #mean(max(u1,u2,u4))
        end
    end
    return value
end





#= solve Emax for conscription group 1: Not obligated to attent conscription =#
function solveGroup1(α10, α11, α12, α13,
            α20, α21, tc1, tc2, α22, α23, α24, α25, α30study,
            α3, α30, α31, α32, α33, α34, α35, α36,
            α4, α40, α41, α42, α43, α44, α45, α46,
            δ,
            epssolve)

    x3Max = 30
    x4Max = 30

    #=
    Pre-allocating Emax
    Emax function is calcuted until age 17

    The arguments are in orders:
        age(17-65),                                  # 49
        education(0-22),                             # 23
        school status of last year(0,1),             # 2
        years of experience in white-collar(0-30),   # 31
        years of experience in blue-collar(0-30),    # 31
    State space size= 49*23*2*31*31=           2,166,094
    =#

    Emax= SharedArray{Float64,5}(49, 23, 2, x3Max+1, x4Max+1);


    ageState  = 65 :-1 :17   # age age of the individual
    educState = 0 :1 :22     # educ number of completed education
    slState   = [0,1]        # sl schooling status of last period
    x3State   = 0 :1 : x3Max     # x3 experience in white-collar
    x4State   = 0 :1 : x4Max     # x4 experience in blue-collar


    for age in ageState
        @sync @distributed for educ in educState
            for sl in slState, x3 in 0:1:min(30, age-5-educ)
                for x4 in 0:1:min(30, age-5-educ-x3)
                    @inbounds Emax[age-16, educ+1, sl+1, x3+1, x4+1] =
                            valueFunctionGroup1(α10, α11, α12, α13,
                                α20, α21, tc1, tc2,  α22, α23, α24, α25, α30study,
                                α3, α30, α31, α32, α33, α34, α35, α36,
                                α4, α40, α41, α42, α43, α44, α45, α46,
                                δ,
                                epssolve,
                                age, educ, sl, x3, x4,
                                Emax)
                end
            end #x3
        end #educ
    end #age

    return Emax

end


# # test section
# # here, we check whether Emax function is workign perfect or not.
# epsSolveMean=[0.0, 0.0, 0.0, 0.0] ;
# epsSolveσ=[ σ1   0.0  0.0   0.0 ;
#             0.0  σ2   0.0   0.0 ;
#             0.0  0.0  σ3    σ34 ;
#             0.0  0.0  σ34   σ4  ] ;
#
# M = 100 ;
# epssolve=rand(MersenneTwister(1234),MvNormal(epsSolveMean, epsSolveσ) , M) ;
#
#
# for i in 1:3
#     print("Emax Group 1 calculation: \n")
#     start = Dates.unix2datetime(time())
#
#     EmaxGroup1 = solveGroup1(ω1T1, α11,
#                     ω2T1, α21, tc1, tc2,
#                     α3, ω3T1, α31, α32, α33, α34, α35,
#                     α4, ω4T1, α41, α42, α43, α44, α45,
#                     δ,
#                     epssolve) ;
#
#     finish = convert(Int, Dates.value(Dates.unix2datetime(time())- start))/1000;
#     print("TOTAL ELAPSED TIME: ", finish, " seconds. \n")
# end





################################################################################
#=
conscription goup 2 value function and solve Emax function
conscription goup 2: obligated to attend conscription
=#

#= value function for conscription goup 2: obligated to attend conscription =#
@everywhere function valueFunctionGroup2(α10, α11, α12, α13,
                α20, α21, tc1, tc2, α22, α23, α24, α25, α30study,
                α3, α30, α31, α32, α33, α34, α35, α36,
                α4, α40, α41, α42, α43, α44, α45, α46,
                α50, α51, α52,
                δ,
                epssolve,
                age, educ, sl, x3, x4, x5, LastSchool,
                Emax)

    x3Max = 30
    x4Max = 30

    ε1= epssolve[1,:]
    u1= util1(α10, α11, α12, α13, age, educ, ε1)

    ε2= epssolve[2,:]
    u2= util2(α20, α21, tc1, tc2, sl, educ+1, ε2, age, α30study)

    ε3= epssolve[3,:]
    u3= util3(α3, α30, α31, α32, α33, α34, α35, α36, x3, x4, educ, ε3, α22, α23)

    ε4= epssolve[4,:]
    u4= util4(α4, α40, α41, α42, α43, α44, α45, α46, x3, x4, educ, ε4, α24, α25)

    ε5= epssolve[5,:]
    u5= util5(α50, α51, α52, educ, ε5)

    value= -1 # this is for when no if conditon binds
    if age == 65

        if educ < 22
            s=0.0
            @simd for i in 1:length(u1)
                s+=max(u1[i], u2[i], u3[i], u4[i])
            end
            value= s/length(u1) #mean(max(u1,u2,u3,u4))
        else
            s=0.0
            @simd for i in 1:length(u1)
                s+=max(u1[i], u3[i], u4[i])
            end
            value= s/length(u1)#mean(max(u1,u2,u4))
        end
    else

        #= assume that maximum years of experience is 30 years. =#

        u1= u1 .+ δ*Emax[age-16+1 ,educ+1,             0+1 ,x3+1           ,x4+1           ,x5+1         ,LastSchool+1+1*(LastSchool!=2)]

        u2= u3 .+ δ*Emax[age-16+1 ,educ+1+1*(educ!=22),1+1 ,x3+1           ,x4+1           ,x5+1         ,0+1                           ]

        u3= u3 .+ δ*Emax[age-16+1 ,educ+1,             0+1 ,x3+1+1*(x3!=x3Max),x4+1           ,x5+1      ,LastSchool+1+1*(LastSchool!=2)]

        u4= u4 .+ δ*Emax[age-16+1 ,educ+1,             0+1 ,x3+1           ,x4+1+1*(x4!=x4Max),x5+1      ,LastSchool+1+1*(LastSchool!=2)]

        u5= u5 .+ δ*Emax[age-16+1 ,educ+1,             0+1 ,x3+1           ,x4+1           ,x5+1+1*(x5<2),LastSchool+1+1*(LastSchool!=2)]

        ######
        ######
        if age > 18
            if x5 == 2
                if educ < 22
                    s=0.0
                    @simd for i in 1:length(u1)
                        s+=max(u1[i], u2[i], u3[i], u4[i])
                    end
                    value= s/length(u1)
                else
                    s=0.0
                    @simd for i in 1:length(u1)
                        s+=max(u1[i], u3[i], u4[i])
                    end
                    value= s/length(u1)
                end
            elseif x5 == 1
                s=0.0
                @simd for i in 1:length(u5)
                    s+= u5[i]
                end
                value= s/length(u5)
            else
                if educ == 22
                    s=0.0
                    @simd for i in 1:length(u5)
                        s+= max(u5[i])
                    end
                    value= s/length(u5)
                else
                    if LastSchool == 0
                        s=0.0
                        @simd for i in 1:length(u5)
                            s+= max(u1[i], u2[i], u5[i])
                        end
                    elseif LastSchool < 2
                        s=0.0
                        @simd for i in 1:length(u5)
                            s+= max(u1[i], u5[i])
                        end
                    else
                        s=0.0
                        @simd for i in 1:length(u5)
                            s+= max(u2[i], u5[i])
                        end
                    end
                    value= s/length(u5)
                end
            end

        elseif age <= 18
            s=0.0
            @simd for i in 1:length(u1)
                s+=max(u1[i], u2[i], u3[i], u4[i])
            end
            value= s/length(u1)
        end
    end
    return value
end




#= Solve Emax for conscription goup 2: obligated to attent conscription =#
function solveGroup2(α10, α11, α12, α13,
            α20, α21, tc1, tc2, α22, α23, α24, α25, α30study,
            α3, α30, α31, α32, α33, α34, α35, α36,
            α4, α40, α41, α42, α43, α44, α45, α46,
            α50, α51, α52,
            δ,
            epssolve)

    x3Max = 30
    x4Max = 30
    #=
    Pre-allocating Emax
    Emax function is calcuted until age 17

    The arguments are in orders:
        age(17-65),                                  # 49
        education(0-22),                             # 23
        school status of last year(0,1),             # 2
        years of experience in white-collar(0-30),   # 31
        years of experience in blue-collar(0-30),    # 31
        years attending conscription(0,1,2)          # 3
        Last time at school befor conscription       # 3 {0,1,2}
    Stata space size= 49*23*2*31*31*3=         6,498,282
    =#
    Emax= SharedArray{Float64,7}(49, 23, 2, x3Max+1, x4Max+1, 3, 3);


    ageState  = 65 :-1 :17   # age age of the individual
    educState = 0 :1 :22     # educ number of completed education
    slState   = [0,1]        # sl schooling status of last period
    x3State   = 0 :1 : x3Max     # x3 experience in white-collar
    x4State   = 0 :1 : x4Max     # x4 experience in blue-collar
    x5State   = [0,1,2]      # x5 indicate the years attending conscription
    LastSchoolState = [0,1,2]


    for age in ageState
        @sync @distributed for educ in educState
            for sl in slState,x5 in x5State, LastSchool in LastSchoolState
                for x3 in 0:1:min(30, age-5-educ-x5)
                    for x4 in 0:1:min(30, age-5-educ-x5-x3)
                        Emax[age-16, educ+1, sl+1, x3+1, x4+1, x5+1, LastSchool+1] =
                            valueFunctionGroup2(α10, α11, α12, α13,
                                α20, α21, tc1, tc2, α22, α23, α24, α25, α30study,
                                α3, α30, α31, α32, α33, α34, α35, α36,
                                α4, α40, α41, α42, α43, α44, α45, α46,
                                α50, α51, α52,
                                δ,
                                epssolve,
                                age, educ, sl, x3, x4, x5, LastSchool,
                                Emax)
                    end
                end
            end#sl
        end#educ
    end#age

    return Emax

end#


# #= test section =#
# #= here we check whether Emax function is workign perfect or not. =#
# epsSolveMeanGroup2= [0.0, 0.0, 0.0, 0.0, 0.0] ;
# epsSolveσGroup2=[σ1   0.0  0.0  0.0  0.0 ;
#                 0.0  σ2   0.0  0.0  0.0 ;
#                 0.0  0.0  σ3   σ34  0.0 ;
#                 0.0  0.0  σ34  σ4   0.0
#                 0.0  0.0  0.0  0.0  σ5  ] ;
# M=10;
# epssolveGroup2= rand(MersenneTwister(1234),MvNormal(epsSolveMeanGroup2, epsSolveσGroup2) , M) ;
#
#
# for i in 1:1
#     print("Emax Group 2 calculation: \n")
#     start = Dates.unix2datetime(time())
#
#     EmaxGroup2= solveGroup2(0, α11, α12, α13,
#                 0, α21, tc1T1, tc2, α22, α23, 0, α25, α30study,
#                 α3, 0, α31, α32, α33, α34, α35, 0,
#                 α4, 0, α41, α42, α43, α44, α45, 0,
#                 α50, α51, α52,
#                 δ,
#                 epssolveGroup2) ;
#
#     finish = convert(Int, Dates.value(Dates.unix2datetime(time())- start))/1000;
#     print("TOTAL ELAPSED TIME: ", finish, " seconds. \n")
# end






################################################################################
#= simulate conscription goup 1 =#

function simulateGroup1(α10, α11, α12, α13,
                    α20, α21, tc1, tc2, α22, α23, α24, α25, α30study,
                    α3, α30, α31, α32, α33, α34, α35, α36,
                    α40, α41, α42, α43, α44, α45, α46,
                    α50, α51, α52,
                    σ1, σ2, σ3, σ4, σ34 ,σ5,
                    N, Emax, weights, Seed)

    x3Max = 30
    x4Max = 30
    #= simCol is a dictionary specifying order of 'simresult' for output =#
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
        "LastSchool" => 11
    );
    #= Pre-allocating each person-year's state=#
    sim = Array{Float64, 2}(undef, (N*50, 11))
    sim[:,simCol["x5"]] .= NaN
    sim[:,simCol["LastSchool"]] .= NaN

    #= education distribution in age 16 of people: =#
    educLevel = [0    ,5    ,8    ,10  ]
    # weights =   [0.02 ,0.20 ,0.24 ,0.54]
    #= drawing educ level exogenously form this distribution =#
    a = sample(MersenneTwister(Seed),educLevel, Weights(weights), N)


    epsSolveMean= [0.0, 0.0, 0.0, 0.0]
    epsSolveσ=[σ1   0.0  0.0  0.0 ;
               0.0  σ2   0.0  0.0 ;
               0.0  0.0  σ3   σ34 ;
               0.0  0.0  σ34  σ4  ]
    epsestimation =
        rand(MersenneTwister(Seed), MvNormal(epsSolveMean, epsSolveσ), 50 * N)

    for id in 1:N


        for age in 16:65

            index= 50*(id-1)+ age-15
            sim[index, simCol["age"]]= age

            if age==16
                x3=0
                x4=0
                educ=a[id]
                sl= 1 #0+1*(educ==10)+1*(educ==8)
            else
                x3  = convert(Int,sim[index-1,simCol["x3"]])
                x4  = convert(Int,sim[index-1,simCol["x4"]])
                educ= convert(Int,sim[index-1,simCol["educ"]])
                sl  =1*(sim[index-1,simCol["choice"]] == 2)
            end

            #= four shocks to person i in age 'age': =#
            ε1,ε2,ε3,ε4= epsestimation[ : , index]

            #= comtemporaneous utility from each decision : =#
            u1= util1(α10, α11, α12, α13, age, educ, ε1)
            u2= util2(α20, α21, tc1, tc2, sl, educ+1, ε2, age, α30study)
            u3= util3(α3, α30, α31, α32, α33, α34, α35, α36, x3, x4, educ, ε3, α22, α23)
            u4= util4(α4, α40, α41, α42, α43, α44, α45, α46, x3, x4, educ, ε4, α24, α25)

            ########################
            ########################

            if age==65
                if educ < 22
                    utility= [u1, u2, u3, u4]
                elseif educ==22
                    utility= [u1, -1e20, u3, u4]
                end
                choice= argmax(utility)
                maxUtility = maximum(utility)
            else
                u1= u1 +δ*Emax[age+1-17+1 ,educ+1             ,0+1 ,x3+1            ,x4+1]
                u2= u2 +δ*Emax[age+1-17+1 ,educ+1+1*(educ< 22),1+1 ,x3+1            ,x4+1]
                u3= u3 +δ*Emax[age+1-17+1 ,educ+1             ,0+1 ,x3+1+1*(x3< x3Max) ,x4+1]
                u4= u4 +δ*Emax[age+1-17+1 ,educ+1             ,0+1 ,x3+1 ,x4+1+1*(x4< x4Max)]

                if educ < 22
                    utility= [u1, u2, u3, u4]
                elseif educ==22
                    utility= [u1, -1e20, u3, u4]
                end
                choice= argmax(utility)
                maxUtility = maximum(utility)
            end

            #= writing 'choice' in results =#
            sim[index, simCol["choice"]] = choice

            #= specifying subsequent period state based on 'choice' of this period =#
            if     choice==1
                sim[index, simCol["income"]]= NaN
                sim[index, simCol["x3"]]    = x3
                sim[index, simCol["x4"]]    = x4
                sim[index, simCol["educ"]]  = educ
            elseif choice==2
                sim[index, simCol["income"]]= NaN
                sim[index, simCol["x3"]]    = x3
                sim[index, simCol["x4"]]    = x4
                sim[index, simCol["educ"]]  = educ+ 1
            elseif choice==3
                sim[index, simCol["income"]]= exp(α30+ α31*educ+
                             α32*x3+ α33*x4+ α34*(x3^2)+ α35*(x4^2)+ α36*(x3==0)+ α22*(educ>=12)+ α23*(educ>=16)+
                             ε3)
                sim[index, simCol["x3"]]  = x3 +1*(x3<30)
                sim[index, simCol["x4"]]  = x4
                sim[index, simCol["educ"]]= educ

            elseif choice==4
                sim[index, simCol["income"]]= exp(α40+ α41*educ+
                             α42*x3+ α43*x4+ α44*(x3^2)+ α45*(x4^2)+ α46*(x4==0)+ α24*(educ>=12)+ α25*(educ>=16)+
                             ε4)
                sim[index, simCol["x3"]]  = x3
                sim[index, simCol["x4"]]  = x4 +1*(x4<30)
                sim[index, simCol["educ"]]= educ
            end
            sim[index, simCol["Emax"]] = maxUtility #utility[choice]

            #=
            specifying if persion is educated or not (educ > 12 or not)
            this helps in calculating moment conditions from simulated data
            =#
            if choice == 2
                educ = educ + 1
            end
            if age < 22
                sim[index, simCol["educated"]] = -1
            else
                if educ > 12
                    sim[index, simCol["educated"]] = 1
                elseif educ <= 12
                    sim[index, simCol["educated"]] = 0
                end
            end

        end

    end

    return sim
end#simulate

################################################################################
#= simulate conscription goup 2 =#

function simulateGroup2(α10, α11, α12, α13,
                    α20, α21, tc1, tc2, α22, α23, α24, α25, α30study,
                    α3, α30, α31, α32, α33, α34, α35, α36,
                    α40, α41, α42, α43, α44, α45, α46,
                    α50, α51, α52,
                    σ1, σ2, σ3, σ4, σ34 ,σ5,
                    N, Emax, weights, Seed)

    x3Max = 30
    x4Max = 30
    #= simCol is a dictionary specifying order of 'simresult' for output =#
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
        "LastSchool" => 11
    );
    #= Pre-allocating each person-year's state =#
    sim = Array{Float64, 2}(undef, (N*50, 11))

    #= education distribution in age 16 of people: =#
    educLevel = [0    ,5    ,8    ,10  ]
    # weights =   [0.02 ,0.20 ,0.24 ,0.54]
    # drawing educ level exogenously form this distribution
    a=sample(MersenneTwister(Seed),educLevel, Weights(weights), N)


    epsSolveMean= [0.0, 0.0, 0.0, 0.0, 0.0]
    epsSolveσ=[σ1   0.0  0.0  0.0  0.0 ;
               0.0  σ2   0.0  0.0  0.0 ;
               0.0  0.0  σ3   σ34  0.0 ;
               0.0  0.0  σ34  σ4   0.0 ;
               0.0  0.0  0.0  0.0  σ5  ]
    epsestimation=rand(MersenneTwister(Seed),MvNormal(epsSolveMean, epsSolveσ) , 50*N)

    for id in 1:N


        for age in 16:65

            index= 50*(id-1)+ age-15
            sim[index, simCol["age"]]= age

            if age==16
                x3   = 0
                x4   = 0
                educ = a[id]
                sl   = 1 #0+1*(educ==10)+1*(educ==8)
                x5   = 0
                LastSchool   = 0
            else
                x3   = convert(Int,sim[index-1,simCol["x3"]])
                x4   = convert(Int,sim[index-1,simCol["x4"]])
                educ = convert(Int,sim[index-1,simCol["educ"]])
                sl   = 1*(sim[index-1,simCol["choice"]] == 2)
                x5   = convert(Int, sim[index-1,simCol["x5"]])
                LastSchool   = convert(Int, sim[index-1,simCol["LastSchool"]])
            end

            #= four shocks to person i in age 'age': =#
            ε1, ε2, ε3, ε4, ε5 = epsestimation[:, index]

            #= comtemporaneous utility from each decision : =#
            u1= util1(α10, α11, α12, α13, age, educ, ε1)
            u2= util2(α20, α21, tc1, tc2, sl, educ+1, ε2, age, α30study)
            u3= util3(α3, α30, α31, α32, α33, α34, α35, α36, x3, x4, educ, ε3, α22, α23)
            u4= util4(α4, α40, α41, α42, α43, α44, α45, α46, x3, x4, educ, ε4, α24, α25)
            u5= util5(α50, α51, α52, educ, ε5)

            ########################
            ########################

            if age==65
                if educ < 22
                    utility= [u1, u2, u3, u4, -1e20]
                elseif educ==22
                    utility= [u1, -1e20, u3, u4, -1e20]
                end
                choice= argmax(utility)
                maxUtility = maximum(utility)
            else

                u1= u1 +δ*Emax[age+1-17+1 ,educ+1              ,0+1 ,x3+1            ,x4+1            ,x5+1           ,LastSchool+1+1*(LastSchool!=2) ]
                u2= u2 +δ*Emax[age+1-17+1 ,educ+1+1*(educ< 22) ,1+1 ,x3+1            ,x4+1            ,x5+1           ,0+1 ]
                u3= u3 +δ*Emax[age+1-17+1 ,educ+1              ,0+1 ,x3+1+1*(x3< x3Max) ,x4+1            ,x5+1        ,LastSchool+1+1*(LastSchool!=2) ]
                u4= u4 +δ*Emax[age+1-17+1 ,educ+1              ,0+1 ,x3+1            ,x4+1+1*(x4< x4Max) ,x5+1        ,LastSchool+1+1*(LastSchool!=2) ]
                u5= u5 +δ*Emax[age+1-17+1 ,educ+1              ,0+1 ,x3+1            ,x4+1            ,x5+1+1*(x5<2)  ,LastSchool+1+1*(LastSchool!=2) ]

                if age > 18
                    if x5 == 2
                        if educ < 22
                            utility= [u1, u2, u3, u4, -1e20]
                        else
                            utility= [u1, -1e20, u3, u4, -1e20]
                        end#if educ
                    elseif x5 == 1
                            utility= [-1e20, -1e20, -1e20, -1e20, u5]
                    else
                        if educ == 22
                            utility= [-1e20, -1e20, -1e20, -1e20, u5]
                        else
                            if LastSchool == 0
                                utility= [u1, u2, -1e20, -1e20, u5]
                            elseif LastSchool < 2
                                utility= [u1, -1e20, -1e20, -1e20, u5]
                            else
                                utility= [-1e20, -1e20, -1e20, -1e20, u5]
                            end
                        end#if educ
                    end#if x5

                elseif age <= 18
                    utility= [u1, u2, u3, u4, -1e20]
                end#if age

                choice= argmax(utility)
                maxUtility = maximum(utility)

            end


            #= writing 'choice' in results =#
            sim[index, simCol["choice"]] = choice

            #= specifying subsequent period state based on 'choice' of this period =#
            if     choice==1
                sim[index, simCol["income"]]= NaN
                sim[index, simCol["x3"]]    = x3
                sim[index, simCol["x4"]]    = x4
                sim[index, simCol["educ"]]  = educ
                sim[index, simCol["x5"]]    = x5
            elseif choice==2
                sim[index, simCol["income"]]= NaN
                sim[index, simCol["x3"]]    = x3
                sim[index, simCol["x4"]]    = x4
                sim[index, simCol["educ"]]  = educ+ 1
                sim[index, simCol["x5"]]    = x5
            elseif choice==3
                sim[index, simCol["income"]]= exp(α30+ α31*educ+
                             α32*x3+ α33*x4+ α34*(x3^2)+ α35*(x4^2)+ α36*(x3==0)+ α22*(educ>=12)+ α23*(educ>=16)+
                             ε3)
                sim[index, simCol["x3"]]  = x3 +1*(x3<30)
                sim[index, simCol["x4"]]  = x4
                sim[index, simCol["educ"]]= educ
                sim[index, simCol["x5"]]    = x5

            elseif choice==4
                sim[index, simCol["income"]]= exp(α40+ α41*educ+
                             α42*x3+ α43*x4+ α44*(x3^2)+ α45*(x4^2)+ α46*(x4==0)+ α24*(educ>=12)+ α25*(educ>=16)+
                             ε4)
                sim[index, simCol["x3"]]  = x3
                sim[index, simCol["x4"]]  = x4 +1*(x4<30)
                sim[index, simCol["educ"]]= educ
                sim[index, simCol["x5"]]    = x5
            elseif choice==5
                sim[index, simCol["income"]]= NaN
                sim[index, simCol["x3"]]    = x3
                sim[index, simCol["x4"]]    = x4
                sim[index, simCol["educ"]]  = educ
                sim[index, simCol["x5"]]    = x5 + 1
            end
            sim[index, simCol["Emax"]] = maxUtility #utility[choice]

            if (age>18) & (LastSchool == 0)
                sim[index, simCol["LastSchool"]] = LastSchool + 1*(choice!=2)
            elseif (age>18) & (LastSchool < 2)  & (x5 == 0)
                sim[index, simCol["LastSchool"]] = LastSchool + 1*(choice!=5)
            else
                sim[index, simCol["LastSchool"]] = LastSchool
            end

            #=
             specifying if persion is educated or not (educ > 12 or not)
             this helps in calculating moment conditions from simulated data
            =#
            if choice == 2
                educ = educ + 1
            end
            if age < 22
                sim[index, simCol["educated"]] = -1
            else
                if educ>12
                    sim[index, simCol["educated"]] = 1
                elseif educ<=12
                    sim[index, simCol["educated"]] = 0
                end
            end
        end

    end

    return sim
end#simulate



################################################################################
#=
    Define SMMCalculate :
    It takes moment from data and model Estimation
    and calculate the error
=#
function SMMCalculate(choiceMoment, wageMoment, wageCol, choiceCol)


    wageWhiteError  = 0.0
    wageBlueError   = 0.0
    devWhiteError   = 0.0
    devBlueError    = 0.0
    homeError       = 0.0
    studyError      = 0.0
    whiteError      = 0.0
    blueError       = 0.0
    milError        = 0.0


    for i in 1:size(wageMoment, 1)

        # percentage error of mean income moment
        error = (
            (
                wageMoment[i, wageCol["incomeData"]] -wageMoment[i, wageCol["incomeSim"]]
            ) / wageMoment[i, wageCol["incomeStdBoot"]]
        )

        # if error > 0.5
        #     error = error*2
        # end

        #=
        If error is NaN, it means no one is working in that occupation, thus we
        make this error bigger to force the optimization algorithm to avoid
        from this area of paramater's domain.
        =#
        if isnan(error)
            error = 10.0
        end

        if wageMoment[i,3] == 0.0
            wageWhiteError += error^2
        end

        if wageMoment[i,3] == 1.0
            # age = wageMoment[i, wageCol["age"]]
            # educated = wageMoment[i, wageCol["educated"]]
            # print("data= ",log(wageMoment[i, wageCol["incomeData"]]),", sim= ",log(wageMoment[i, wageCol["incomeSim"]]),"\n")
            # print("age= ",age,", educated= ",educated,", error= ",error,"\n")
            wageBlueError += error^2
        end


        #= percentage error of mean income standard deviation =#
        error = (
            (
                wageMoment[i, wageCol["devData"]] -wageMoment[i, wageCol["devSim"]]
            ) / wageMoment[i, wageCol["devStdBoot"]]
        )

        # if error > 0.5
        #     error = error*2
        # end

        if isnan(error)
            error = 0.0
        end

        if wageMoment[i,3] == 0.0
            devWhiteError += error^2
        end

        if wageMoment[i,3] == 1.0
            devBlueError += error^2
        end



    end



    for i in 1:size(choiceMoment,1)

        #= choice 1: home production =#
        error = (
            (
                choiceMoment[i, choiceCol["homeData"]] -
                choiceMoment[i, choiceCol["homeSim" ]]
            ) / choiceMoment[i, choiceCol["homeStdBoot"]]
        )

        # if error > 0.5
        #     error = error*2
        # end

        if isnan(error)
            error =  0.0
        end
        if error == 1.0
            error =  30.0
        end

        homeError += error^2

        #= choice 2: study =#
        error = (
            (
                choiceMoment[i, choiceCol["studyData"]] -
                choiceMoment[i, choiceCol["studySim" ]]
            ) / choiceMoment[i, choiceCol["studyStdBoot"]]
        )
        # age = choiceMoment[i, choiceCol["age"]]
        # educated = choiceMoment[i, choiceCol["educated"]]
        # print("data= ",choiceMoment[i, choiceCol["studyData"]],", sim= ",choiceMoment[i, choiceCol["studySim"]],"\n")
        # print("age= ",age,", educated= ",educated,", error= ",error,"\n")

        # if error > 0.5
        #     error = error*2
        # end

        if isnan(error)
            error =  0.0
        end

        studyError += error^2

        #= choice 3: white-collar occupation =#
        error = (
            (
                choiceMoment[i, choiceCol["whiteData"]] -
                choiceMoment[i, choiceCol["whiteSim" ]]
            ) / choiceMoment[i, choiceCol["whiteStdBoot"]]
        )

        # age = choiceMoment[i, choiceCol["age"]]
        # educated = choiceMoment[i, choiceCol["educated"]]
        # print("data= ",choiceMoment[i, choiceCol["whiteData"]],", sim= ",choiceMoment[i, choiceCol["whiteSim"]],"\n")
        # print("age= ",age,", educated= ",educated,", error= ",error,"\n")

        # if error > 0.5
        #     error = error*2
        # end


        if isnan(error)
            error =  0.0
        end

        if error == 1.0
            error =  50.0
        end


        whiteError += error^2

        #= choice 4: blue-collar occupation =#
        error = (
            (
                choiceMoment[i, choiceCol["blueData"]] -
                choiceMoment[i, choiceCol["blueSim" ]]
            ) / choiceMoment[i, choiceCol["blueStdBoot"]]
        )

        # if error > 0.5
        #     error = error*2
        # end

        if isnan(error)
            error =  0.0
        end

        if error == 1.0
            error =  50.0
        end

        blueError += error^2

        #= choice 5: compulsory military service =#
        error = (
            (
                choiceMoment[i, choiceCol["milData"]] -
                choiceMoment[i, choiceCol["milSim" ]]
            ) / choiceMoment[i, choiceCol["milStdBoot"]]
        )
        # if error > 0.5
        #     error = error*2
        # end

        if isnan(error)
            error =  0.0
        end
        if choiceMoment[i, choiceCol["age"]] > 18
            milError += error^2
        end

    end


    # #= Printing each error seperately =#
    # print("\n wageWhiteError  = ", wageWhiteError)
    # print("\n wageBlueError   = ", wageBlueError )
    # print("\n homeError       = ", homeError     )
    # print("\n studyError      = ", studyError    )
    # print("\n whiteError      = ", whiteError    )
    # print("\n blueError       = ", blueError     )
    # print("\n milError        = ", milError      )
    # print("\n devWhiteError   = ", devWhiteError )
    # print("\n devBlueError    = ", devBlueError  )


    #=
    Shift the error term when estimation is going to areas of parameters
    where no one employ in white-collar or blue-occupation
    =#
    if wageWhiteError == 0.0
        wageWhiteError = 1000.0
    end
    if wageBlueError == 0.0
        wageBlueError = 1000.0
    end
    # if whiteError > 21
    #     whiteError = whiteError * 4
    # end
    # if blueError > 10
    #     blueError = blueError * 4
    # end


    #= total error =#
    SMMError = (
        wageWhiteError +
        wageBlueError +
        homeError +
        studyError +
        whiteError +
        blueError +
        milError +
        devWhiteError +
        devBlueError
    )

    return SMMError
end

################################################################################
#= Define estimation Function =#

function estimation(params, choiceMomentData, wageMomentData)


    #=****************************************************=#
    #= parameters =#

    ω1T1, ω1T2, ω1T3, ω1T4, α11, α12, α13,
    ω2T1, ω2T2, ω2T3, ω2T4,
    α21, tc1T1, tc2, α22, α23, α25, α30study,
    α3, ω3T1, ω3T2, ω3T3, ω3T4, α31, α32, α33, α34, α35,
        ω4T1, ω4T2, ω4T3, ω4T4, α41, α42, α43, α44, α45,
    α50, α51, α52,
    σ1, σ2, σ3, σ4, σ34 ,σ5,
    πE1T1, πE1T2, πE1T3,
    πE2T1, πE2T2, πE2T3,
    π1T1, π1T2, π1T3, π1T4                 = params


    #=****************************************************=#
    α21 = exp(α21)
    tc1T1 = exp(tc1T1)
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
    α13 = -exp(-α13)
    α30study = -exp(-α30study)

    #=
    Some parameters are passed in logarithm scale, this is just for
    easier interpreting them.
    =#
    ω1T1 = exp(ω1T1)  ;   # the intercept of staying home α10 for type 1
    ω1T2 = exp(ω1T2)  ;   # the intercept of staying home α10 for type 2
    ω1T3 = exp(ω1T3)  ;   # the intercept of staying home α10 for type 3
    ω1T4 = exp(ω1T4)  ;   # the intercept of staying home α10 for type 4

    ω2T1 = exp(ω2T1) ;   # the intercept of studying for type 1
    ω2T2 = exp(ω2T2) ;   # the intercept of studying for type 2
    ω2T3 = exp(ω2T3) ;   # the intercept of studying for type 3
    ω2T4 = exp(ω2T4) ;   # the intercept of studying for type 4

    α4 = 0.0  ;  # non pecuniary utility of blue-collar asssumed zero
    # π1 = 0.78 ;  # share of individuals type 1

    #= entry cost of without experience =#
    α36, α46 = 0.0 , 0.0 ;
    tc1T2 = tc1T1
    tc1T3 = tc1T1
    tc1T4 = tc1T1
    α24 = α22

    N = 50 * 1000 ;   # number of individual to simulate their behaviour

    #= share of each education level at 15 years old =#
    # levels are 0, 5, 8, 10
    educShare =   [0.019 ,0.198 ,0.241 ,0.542]
    # educShare =   [0.023 ,0.138 ,0.185 ,0.654]

    #=
    Save the result in a text file
    this helps when the optimization is running on the server
    to catch the best candidater through run time easily
    however it makes a little inconsistecy, because Julia can not understand
    the type of input in compile time, but it does not make a trouble fro performance
    =#
    if ENV["USER"] == "sabouri"
        bestResult = readdlm("/home/sabouri/Labor/CodeOutput/result.csv") ;
    end

    δ = 0.90 #0.7937395498108646 ;      # discount factor

    #=****************************************************=#
    #= check the validity of the input parameters =#
    if (δ > 1) | (δ < 0)
        return 1e4
    end
    if (πE1T1 > 1) | (πE1T1 < 0)
        return 1e4
    end
    if (πE1T2 > 1) | (πE1T2 < 0)
        return 1e4
    end
    if (πE1T3 > 1) | (πE1T3 < 0)
        return 1e4
    end
    if (1-πE1T1-πE1T2-πE1T3) < 0
        return 1e4
    end

    if (πE2T1 > 1) | (πE2T1 < 0)
        return 1e4
    end
    if (πE2T2 > 1) | (πE2T2 < 0)
        return 1e4
    end
    if (πE2T3 > 1) | (πE2T3 < 0)
        return 1e4
    end
    if (1-πE2T1-πE2T2-πE2T3) < 0
        return 1e4
    end



    #=****************************************************=#
    #= solve the model =#
    M = 150 #200

    #=     conscription goup 1     =#
    epsSolveMeanGroup1= [0.0, 0.0, 0.0, 0.0]
    epsSolveσGroup1= [ σ1   0.0  0.0   0.0 ;
                      0.0  σ2   0.0   0.0 ;
                      0.0  0.0  σ3    σ34 ;
                      0.0  0.0  σ34   σ4  ]

    #= check if the variance-covariance matrix is valid =#
    if !isposdef(epsSolveσGroup1)
        return 1e4
    end

    epssolveGroup1= rand(MersenneTwister(1234),
                        MvNormal(epsSolveMeanGroup1, epsSolveσGroup1), M) ;

    EmaxGroup1T1= solveGroup1(ω1T1, α11, α12, α13,
                ω2T1, α21, tc1T1, tc2, α22, α23, α24, α25, α30study,
                α3, ω3T1, α31, α32, α33, α34, α35, α36,
                α4, ω4T1, α41, α42, α43, α44, α45, α46,
                δ,
                epssolveGroup1)

    EmaxGroup1T2= solveGroup1(ω1T2, α11, α12, α13,
                ω2T2, α21, tc1T2, tc2, α22, α23, α24, α25, α30study,
                α3, ω3T2, α31, α32, α33, α34, α35, α36,
                α4, ω4T2, α41, α42, α43, α44, α45, α46,
                δ,
                epssolveGroup1)

    EmaxGroup1T3= solveGroup1(ω1T3, α11, α12, α13,
                ω2T3, α21, tc1T3, tc2, α22, α23, α24, α25, α30study,
                α3, ω3T3, α31, α32, α33, α34, α35, α36,
                α4, ω4T3, α41, α42, α43, α44, α45, α46,
                δ,
                epssolveGroup1)

    EmaxGroup1T4= solveGroup1(ω1T4, α11, α12, α13,
                ω2T4, α21, tc1T4, tc2, α22, α23, α24, α25, α30study,
                α3, ω3T4, α31, α32, α33, α34, α35, α36,
                α4, ω4T4, α41, α42, α43, α44, α45, α46,
                δ,
                epssolveGroup1)


    #=     conscription goup 2     =#
    epsSolveMeanGroup2= [0.0, 0.0, 0.0, 0.0, 0.0] ;
    epsSolveσGroup2=[σ1   0.0  0.0  0.0  0.0 ;
                    0.0  σ2   0.0  0.0  0.0 ;
                    0.0  0.0  σ3   σ34  0.0 ;
                    0.0  0.0  σ34  σ4   0.0 ;
                    0.0  0.0  0.0  0.0  σ5  ] ;

    #= check if the variance-covariance matrix is valid =#
    if !isposdef(epsSolveσGroup2)
        return 1e4
    end

    epssolveGroup2= rand(MersenneTwister(4321),
                        MvNormal(epsSolveMeanGroup2, epsSolveσGroup2) , M) ;

    EmaxGroup2T1= solveGroup2(ω1T1, α11, α12, α13,
                ω2T1, α21, tc1T1, tc2, α22, α23, α24, α25, α30study,
                α3, ω3T1, α31, α32, α33, α34, α35, α36,
                α4, ω4T1, α41, α42, α43, α44, α45, α46,
                α50, α51, α52,
                δ,
                epssolveGroup2) ;

    EmaxGroup2T2= solveGroup2(ω1T2, α11, α12, α13,
                ω2T2, α21, tc1T2, tc2, α22, α23, α24, α25, α30study,
                α3, ω3T2, α31, α32, α33, α34, α35, α36,
                α4, ω4T2, α41, α42, α43, α44, α45, α46,
                α50, α51, α52,
                δ,
                epssolveGroup2) ;

    EmaxGroup2T3= solveGroup2(ω1T3, α11, α12, α13,
                ω2T3, α21, tc1T3, tc2, α22, α23, α24, α25, α30study,
                α3, ω3T3, α31, α32, α33, α34, α35, α36,
                α4, ω4T3, α41, α42, α43, α44, α45, α46,
                α50, α51, α52,
                δ,
                epssolveGroup2) ;

    EmaxGroup2T4= solveGroup2(ω1T4, α11, α12, α13,
                ω2T4, α21, tc1T4, tc2, α22, α23, α24, α25, α30study,
                α3, ω3T4, α31, α32, α33, α34, α35, α36,
                α4, ω4T4, α41, α42, α43, α44, α45, α46,
                α50, α51, α52,
                δ,
                epssolveGroup2) ;

    Emax = [EmaxGroup1T1,EmaxGroup1T2,EmaxGroup1T3,EmaxGroup1T4
            ,EmaxGroup2T1,EmaxGroup2T2,EmaxGroup2T3,EmaxGroup2T4]

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
        "Emax"     => 10
    )

    πE1T4 = 1 - πE1T1 - πE1T2 - πE1T3
    πE2T4 = 1 - πE2T1 - πE2T2 - πE2T3


    E1 = convert(Int, round(educShare[1]*N))
    E1T1 = convert(Int, round(πE1T1*E1))
    E1T2 = convert(Int, round(πE1T2*E1))
    E1T3 = convert(Int, round(πE1T3*E1))
    E1T4 = E1 - E1T1 - E1T2 - E1T3

    E2 = convert(Int, round(educShare[2]*N))
    E2T1 = convert(Int, round(πE1T1*E2))
    E2T2 = convert(Int, round(πE1T2*E2))
    E2T3 = convert(Int, round(πE1T3*E2))
    E2T4 = E2 - E2T1 - E2T2 - E2T3

    E3 = convert(Int, round(educShare[3]*N))
    E3T1 = convert(Int, round(πE1T1*E3))
    E3T2 = convert(Int, round(πE1T2*E3))
    E3T3 = convert(Int, round(πE1T3*E3))
    E3T4 = E3 - E3T1 - E3T2 - E3T3

    E4 = N - E1 - E2 - E3
    E4T1 = convert(Int, round(πE2T1*E4))
    E4T2 = convert(Int, round(πE2T2*E4))
    E4T3 = convert(Int, round(πE2T3*E4))
    E4T4 = E4 - E4T1 - E4T2 - E4T3



    weightsT1 = [
        E1T1*1.0,
        E2T1*1.0,
        E3T1*1.0,
        E4T1*1.0
    ]
    NGroup1T1 = convert(Int, round(sum(weightsT1) * π1T1))
    if NGroup1T1 > 0
        simGroup1T1= simulateGroup1(ω1T1, α11, α12, α13,
                                ω2T1, α21, tc1T1, tc2, α22, α23, α24, α25, α30study,
                                α3, ω3T1, α31, α32, α33, α34, α35, α36,
                                    ω4T1, α41, α42, α43, α44, α45, α46,
                                α50, α51, α52,
                                σ1, σ2, σ3, σ4, σ34 ,σ5,
                                NGroup1T1, EmaxGroup1T1, weightsT1, 1111)
        simGroup1T1[:, simCol["type"]] .= 1
    else
        simGroup1T1 = reshape([],0,11)
    end

    weightsT2 = [
        E1T2*1.0,
        E2T2*1.0,
        E3T2*1.0,
        E4T2*1.0
    ]
    NGroup1T2 = convert(Int, round(sum(weightsT2) * π1T2))
    if NGroup1T2 > 0
        simGroup1T2= simulateGroup1(ω1T2, α11, α12, α13,
                                ω2T2, α21, tc1T2, tc2, α22, α23, α24, α25, α30study,
                                α3, ω3T2, α31, α32, α33, α34, α35, α36,
                                    ω4T2, α41, α42, α43, α44, α45, α46,
                                α50, α51, α52,
                                σ1, σ2, σ3, σ4, σ34 ,σ5,
                                NGroup1T2, EmaxGroup1T2, weightsT2, 2222)
        simGroup1T2[:, simCol["type"]] .= 2
    else
        simGroup1T2 = reshape([],0,11)
    end

    weightsT3 = [
        E1T3*1.0,
        E2T3*1.0,
        E3T3*1.0,
        E4T3*1.0
    ]
    NGroup1T3 = convert(Int, round(sum(weightsT3) * π1T3))
    if NGroup1T3 > 0
        simGroup1T3= simulateGroup1(ω1T3, α11, α12, α13,
                                ω2T3, α21, tc1T3, tc2, α22, α23, α24, α25, α30study,
                                α3, ω3T3, α31, α32, α33, α34, α35, α36,
                                    ω4T3, α41, α42, α43, α44, α45, α46,
                                α50, α51, α52,
                                σ1, σ2, σ3, σ4, σ34 ,σ5,
                                NGroup1T3, EmaxGroup1T3, weightsT3, 1345)
        simGroup1T3[:, simCol["type"]] .= 3
    else
        simGroup1T3 = reshape([],0,11)
    end

    weightsT4 = [
        E1T4*1.0,
        E2T4*1.0,
        E3T4*1.0,
        E4T4*1.0
    ]

    NGroup1T4 = convert(Int, round(sum(weightsT4) * π1T4))
    if NGroup1T4 > 0
        simGroup1T4= simulateGroup1(ω1T4, α11, α12, α13,
                                ω2T4, α21, tc1T4, tc2, α22, α23, α24, α25, α30study,
                                α3, ω3T4, α31, α32, α33, α34, α35, α36,
                                    ω4T4, α41, α42, α43, α44, α45, α46,
                                α50, α51, α52,
                                σ1, σ2, σ3, σ4, σ34 ,σ5,
                                NGroup1T4, EmaxGroup1T4, weightsT4, 5432)
        simGroup1T4[:, simCol["type"]] .= 4
    else
        simGroup1T4 = reshape([],0,11)
    end




    NGroup2T1 = E1T1+E2T1+E3T1+E4T1 - NGroup1T1
    if NGroup2T1 > 0
        simGroup2T1= simulateGroup2(ω1T1, α11, α12, α13,
                                ω2T1, α21, tc1T1, tc2, α22, α23, α24, α25, α30study,
                                α3, ω3T1, α31, α32, α33, α34, α35, α36,
                                    ω4T1, α41, α42, α43, α44, α45, α46,
                                α50, α51, α52,
                                σ1, σ2, σ3, σ4, σ34 ,σ5,
                                NGroup2T1, EmaxGroup2T1, weightsT1, 3333)
        simGroup2T1[:, simCol["type"]] .= 1
    else
        simGroup2T1 = reshape([],0,11)
    end
    NGroup2T2 = E1T2+E2T2+E3T2+E4T2 - NGroup1T2
    if NGroup2T2 > 0
        simGroup2T2= simulateGroup2(ω1T2, α11, α12, α13,
                                ω2T2, α21, tc1T2, tc2, α22, α23, α24, α25, α30study,
                                α3, ω3T2, α31, α32, α33, α34, α35, α36,
                                    ω4T2, α41, α42, α43, α44, α45, α46,
                                α50, α51, α52,
                                σ1, σ2, σ3, σ4, σ34 ,σ5,
                                NGroup2T2, EmaxGroup2T2, weightsT2, 4444)
        simGroup2T2[:, simCol["type"]] .= 2
    else
        simGroup2T2 = reshape([],0,11)
    end

    NGroup2T3 = E1T3+E2T3+E3T3+E4T3 - NGroup1T3
    if NGroup2T3 > 0
        simGroup2T3= simulateGroup2(ω1T3, α11, α12, α13,
                                ω2T3, α21, tc1T3, tc2, α22, α23, α24, α25, α30study,
                                α3, ω3T3, α31, α32, α33, α34, α35, α36,
                                    ω4T3, α41, α42, α43, α44, α45, α46,
                                α50, α51, α52,
                                σ1, σ2, σ3, σ4, σ34 ,σ5,
                                NGroup2T3, EmaxGroup2T3, weightsT3, 5234)
        simGroup2T3[:, simCol["type"]] .= 3
    else
        simGroup2T3 = reshape([],0,11)
    end

    NGroup2T4 = E1T4+E2T4+E3T4+E4T4 - NGroup1T4
    if NGroup2T4 > 0
        simGroup2T4= simulateGroup2(ω1T4, α11, α12, α13,
                                ω2T4, α21, tc1T4, tc2, α22, α23, α24, α25, α30study,
                                α3, ω3T4, α31, α32, α33, α34, α35, α36,
                                    ω4T4, α41, α42, α43, α44, α45, α46,
                                α50, α51, α52,
                                σ1, σ2, σ3, σ4, σ34 ,σ5,
                                NGroup2T4, EmaxGroup2T4, weightsT4, 7764)
        simGroup2T4[:, simCol["type"]] .= 4
    else
        simGroup2T4 = reshape([],0,11)
    end


    #= Concatenate two simulation =#
    sim = [simGroup1T1; simGroup1T2; simGroup1T3; simGroup1T4;
           simGroup2T1; simGroup2T2; simGroup2T3; simGroup2T4 ] ;


    #=****************************************************=#
    #=
    Calculating moment from simulation
    we build some arrays to put data moment and model moment aside

    generating an Array named wageMoment
    to store average income of the simulated moments
    also we embed data moment in this array
    wageMomentData is the given wage moment from data
    =#
    wageMoment= Array{Float64,2}(undef, (size(wageMomentData,1),9))
    wageCol = Dict(
        "age"         => 1,
        "educated"    => 2,
        "collar"      => 3,
        "incomeData"  => 4,
        "incomeStdBoot" =>5,
        "devData"     => 6,
        "devStdBoot"  => 7,
        "incomeSim"   => 8,
        "devSim"      => 9
    )
    wageMoment[:,1:7]= wageMomentData[:,1:7];

    #=
    generating an Array named choiceMoment
    to store simulated share of alternatives
    also we embed data moment in this array
    choiceMomentData is the given alternative share moment from data
    =#
    choiceMoment= Array{Float64,2}(undef, (size(choiceMomentData,1),17) )
    choiceCol = Dict(
        "age"       => 1,
        "educated"  => 2,
        "homeData"  => 3,
        "studyData" => 4,
        "whiteData" => 5,
        "blueData"  => 6,
        "milData"   => 7,
        "homeStdBoot"  => 8,
        "studyStdBoot" => 9,
        "whiteStdBoot" => 10,
        "blueStdBoot"  => 11,
        "milStdBoot"   => 12,
        "homeSim"   => 13,
        "studySim"  => 14,
        "whiteSim"  => 15,
        "blueSim"   => 16,
        "milSim"    => 17
    )
    choiceMoment[ :, 1:12] =choiceMomentData[: ,1:12]

    #=
    removing share below 1 percent for two reason:
    it is not informative about distribution of choices
    also increases the error of coumputation daramatically large if they remain
    =#
    for i in 3:6
        choiceMoment[(choiceMoment[:,i].<0.01) ,i] .= NaN
    end

    #=****************************************************=#
    #=
    sim is simulation of N people behaviour
    here we update data moment conditio
    =#
    ageInterval= unique(choiceMoment[:,choiceCol["age"]])
    ageMax= maximum(ageInterval)

    for age in ageInterval

        #= mean income for each occupation moment condition =#

        for educated in unique(wageMoment[ wageMoment[:,wageCol["age"]].== age , wageCol["educated"] ])
            for collar in unique(wageMoment[ wageMoment[:,wageCol["age"]].== age , wageCol["collar"] ])

                #= amendment =#
                if age < 22
                    educated = -1
                end
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
                            (sim[:,simCol["choice"]].== choice).&
                            (sim[:,simCol["educated"]].==educated) , simCol["income"]]

                if age<22
                    educated = 0
                end
                wageMoment[ (wageMoment[:,wageCol["age"]].==age).&
                            (wageMoment[:,wageCol["collar"]].==collar).&
                            (wageMoment[:,wageCol["educated"]].==educated)
                            , wageCol["incomeSim"]] .= mean(filter(!isnan, flag ))

                wageMoment[ (wageMoment[:,wageCol["age"]].==age).&
                            (wageMoment[:,wageCol["collar"]].==collar).&
                            (wageMoment[:,wageCol["educated"]].==educated)
                            , wageCol["devSim"]] .= std(filter(!isnan, flag ))

            end #for collar
        end #for educated


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

    end#for age

    # for i = 8:12
    #     choiceMoment[ isnan.(choiceMoment[:,i]) , i ] .= 0
    # end


    #=****************************************************=#
    #=
    calculating error = sum squared of percentage distance
    between data moment and moment from model simulation
    =#
    result = SMMCalculate(choiceMoment, wageMoment, wageCol, choiceCol)


    #=
    Putting all moment in a vector for calculating jacobian of
    the moment by changing parameters
    =#
    momentSim = [wageMoment[:,6] ; wageMoment[:,7]]

    for i = 8:12
        momentSim = [momentSim ; choiceMoment[:,i]]
    end

    momentData = [wageMoment[:,4] ; wageMoment[:,5]]
    for i = 3:7
        momentData = [momentData ; choiceMoment[:,i]]
    end

    moment = momentSim-momentData # (momentSim-momentData)./momentData


    #=****************************************************=#
    if ENV["USER"] == "ehsan"
        ## Linux ##
        writedlm("/home/ehsan/Dropbox/Labor/Codes/Moments/data/sim.csv", sim, ',')
    end
    if ENV["USER"]=="sabouri"
        if (result < bestResult[1])
        ## Server ##
        writedlm("/home/sabouri/Labor/CodeOutput/result.csv", result , ',') ;
        writedlm("/home/sabouri/Labor/CodeOutput/parameters.csv", params , ',') ;
        # writedlm( "/home/sabouri/Labor/CodeOutput/choiceMoment.csv",  choiceMoment, ',');
        # writedlm( "/home/sabouri/Labor/CodeOutput/wageMoment.csv",  wageMoment, ',');
        writedlm( "/home/sabouri/Labor/CodeOutput/sim.csv",  sim, ',');

        ## Windows ##
        # writedlm("C:/Users/claudioq/Dropbox/Labor/Codes/Moments/data/sim.csv", sim, ',')

        # ***************************************************
        # send email after completing the optimization
        # opt = SendOptions(
        #   isSSL = true,
        #   username = "juliacodeserver@gmail.com",
        #   passwd = "JuliaCodeServer")
        # #Provide the message body as RFC5322 within an IO
        # body = IOBuffer(
        #   "Date: Fri, 18 Oct 2013 21:44:29 +0100\r\n" *
        #   "From: You <juliacodeserver@gmail.com>\r\n" *
        #   "To: ehsansaboori75@gmail.com\r\n" *
        #   "Subject: Julia Code\r\n" *
        #   "\r\n" *
        #   "Better solution found (: \r\n")
        # url = "smtps://smtp.gmail.com:465"
        # rcpt = ["<ehsansaboori75@gmail.com>"]
        # from = "<juliacodeserver@gmail.com>"
        # resp = send(url, rcpt, from, body, opt)
        end
    end


    #= return SMM error calculated =#
    print("\n SMM error = ", result, " ")
    return result, moment, momentData #, choiceMoment, wageMoment, sim
end
















################################################################################
#= read data moment files =#

#= code for reading in server =#
if ENV["USER"] == "sabouri"
    wageMomentData= readdlm("/home/sabouri/Labor/DataMoments/wageMoment2.csv",',')      ;
    choiceMomentData = readdlm("/home/sabouri/Labor/DataMoments/choiceMoment2.csv",',') ;
    wageMomentStdBoot= readdlm("/home/sabouri/Labor/DataMoments/wageMomentStdBoot.csv",',')      ;
    choiceMomentStdBoot = readdlm("/home/sabouri/Labor/DataMoments/choiceMomentStdBoot.csv",',') ;
end
#= code for reading in Linux operating system =#
if ENV["USER"] == "ehsan"
    wageMomentData= readdlm("/home/ehsan/Dropbox/Labor/Codes/Moments/wageMoment2.csv",',')       ;
    choiceMomentData = readdlm("/home/ehsan/Dropbox/Labor/Codes/Moments/choiceMoment2.csv",',')  ;
    wageMomentStdBoot = readdlm("/home/ehsan/Dropbox/Labor/Codes/Moments/wageMomentStdBoot.csv",',')
    choiceMomentStdBoot = readdlm("/home/ehsan/Dropbox/Labor/Codes/Moments/choiceMomentStdBoot.csv",',')
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
# πE1T4 = 1- πE1T1- πE1T2- πE1T3

#= share of each type for those education equalls 10 in 15 years old =#
πE2T1 = 0.532182272493524
πE2T2 = 0.21200626083052643
πE2T3 = 0.1216150006037918
# πE2T4 = 1- πE2T1- πE2T2- πE2T3


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
π1T1 = 0.805
π1T2 = 0.835
π1T3 = 0.93
π1T4 = 0.93

δ = 0.7937395498108646 ;      # discount factor

#= New parameters in the model =#
α11 = -log(6.2705530131153148e6)  # if age<=18
α12 = log(1.38e7)                # if educ >=13
α13 = -log(8.22e6)                # if age>=30

α30study = -log(1.12e7)


# tc1T1 = log(exp(tc1T1)*0.5)




params=[ω1T1, ω1T2, ω1T3, ω1T4, α11, α12, α13 ,
        ω2T1, ω2T2, ω2T3, ω2T4,
        α21, tc1T1, tc2, α22, α23, α25, α30study,
        α3, ω3T1, ω3T2, ω3T3, ω3T4, α31, α32, α33, α34, α35,
            ω4T1, ω4T2, ω4T3, ω4T4, α41, α42, α43, α44, α45,
        α50, α51, α52,
        σ1, σ2, σ3, σ4, σ34 ,σ5,
        πE1T1, πE1T2, πE1T3,
        πE2T1, πE2T2, πE2T3,
        π1T1, π1T2, π1T3, π1T4  ] ;


# params = readdlm("C:/Users/claudioq/Dropbox/Labor/Codes/parameters.csv")
params = readdlm("/home/sabouri/Labor/CodeOutput/parameters.csv")


print("\nEstimation started:")
start = Dates.unix2datetime(time())

result, moment, momentData = estimation(params, choiceMomentStdBoot, wageMomentStdBoot) ;

finish = convert(Int, Dates.value(Dates.unix2datetime(time())- start))/1000;
print("\nTtotal Elapsed Time: ", finish, " seconds. \n")

# EmaxGroup1T1,EmaxGroup1T2,EmaxGroup1T3,EmaxGroup1T4,EmaxGroup2T1,EmaxGroup2T2,EmaxGroup2T3,EmaxGroup2T4 = Emax ;

# @code_warntype estimation(params)


# writedlm( "/home/sabouri/thesis/moments/data/choiceMoment.csv",  choiceMoment, ',');
# writedlm( "/home/sabouri/thesis/moments/data/wageMoment.csv",  wageMoment, ',');











## ##############################################################################
#= Optimization =#
if ENV["USER"] == "sabouri"

println("\n \n \n \n")
println("optimization started at = " ,Dates.format(now(), "HH:MM"))

#=**********************************************=#
# #= NLopt.jl =#
# #= specifying algorithm and number of parameters =#
# opt = NLopt.Opt(:LN_NELDERMEAD, 28)
#
# #= specifying objective functiom =#
# opt.min_objective = estimation
#
# #= specifying upper and lower bounds of parameters =#
# opt.lower_bounds = paramsLower
# opt.upper_bounds = paramsUpper
#
# #= optimization stupping creiteria =#
# opt.ftol_rel = 0.01
# # apt.stopval = 10.0
# # apt.maxeval = 2000
#
# #= start the optimizatiom =#
# (optf,optx,ret) = NLopt.optimize(opt, params)
#
# #= writting the output =#
# numevals = apt.numevals
# println("got $optf at $optx after $numevals iterations (returned $ret)")


#= ********************************************** =#
#= Otpim.jl =#
#= older optimization code with Optim package =#
optimization = Optim.optimize(
                x -> estimation(x, choiceMomentStdBoot, wageMomentStdBoot)[1]
                ,params
                ,NelderMead()
                ,Optim.Options(
                 f_tol = 0.05
                ,g_tol = 0.05
                ,allow_f_increases= true
                ,iterations = 2000
                ))

println(optimization)
println(Optim.minimizer(optimization))


#=**********************************************=#
# #= BlackBoxOptim.jl =#
# bounds = [(i*1.0,i*1.0) for i in 1:22]
# # for i in 1:28
# #     bounds[i] = (paramsLower[i], paramsUpper[i])
# # end
# # for i in 1:33
# #     if params[i] > 0
# #         bounds[i] = (params[i]*0.6, params[i]*1.4)
# #     else
# #         bounds[i] = (params[i]*1.4, params[i]*0.6)
# #     end
# # end
# # bounds[32] = (0,1)
# # bounds[33] = (0,1)
# for i in 1:16
#     bounds[i] = (params[i]*0.94, params[i]*1.06)
# end
# for i in 17:22
#     bounds[i] = (params[i]*0.85, params[i]*1.2)
# end
#
#
#
# optimization= BlackBoxOptim.bboptimize(x -> estimation(x, choiceMomentData, wageMomentData) ;
#                          SearchRange = bounds,
#                          method = :adaptive_de_rand_1_bin_radiuslimited ,
#                          MaxTime = 3*24*60*60 )
#
# println(best_candidate(optimization))
#
#
#
# #= showing when optimization finished =#
# println("stop at = " ,Dates.format(now(), "HH:MM"))






################################################################################
# # #= estimating standard error's of the model parameters =#
# #
# # # Pkg.add("ForwardDiff")
# # # using ForwardDiff
# # #
# # # @everywhere  g = x -> ForwardDiff.jacobian(estimation, x); # g = ∇f
# # #
# # # Jacobian =  g(params)
#
# params = Optim.minimizer(optimization)
#
# Jacobian = 0.0 .* Array{Float64,2}(undef,(size(moment,1),size(params,1)) )
# Effect = 0.0 .* Array{Float64,2}(undef,(size(params,1),2) )
#
# input1 = copy(params)
# input2 = copy(params)
#
# for i in 1:size(params,1)
#     print(i,"\n")
#     input1 = copy(params)
#     input1[i] = params[i]*0.990
#     result, moment1, momentData = estimation(input1, choiceMomentData, wageMomentData)
#     Effect[i,1] = result
#
#     input2 = copy(params)
#     input2[i] = params[i]*1.010
#     result, moment2, momentData = estimation(input2, choiceMomentData, wageMomentData)
#     Effect[i,2] = result
#
#     if params[i] > 0
#         diff = (moment2 - moment1)./(abs(input2[i]-input1[i]))
#     else
#         diff = (moment1 - moment2)./(abs(input2[i]-input1[i]))
#     end
#     Jacobian[:,i] = diff
#     # print(moment2-moment1,"   ",abs(input2[i]-input1[i]))
#     # print("\n",params[i]," ",input2[i]," ",input1[i],"\n")
#
# end
#
# using DataFrames
# jac =DataFrame(Jacobian);
# jac = filter(row -> ! isnan(row.x1), jac);
# jac = Matrix(jac);
#
# # momentData2 =DataFrame(momentData);
# # momentData2 = filter(row -> ! isnan(row.x1), momentData2);
# # momentData2 = Matrix(momentData2);
#
# W = 1 * Matrix(I, size(jac,1), size(jac,1))
# # for i = 1:size(W,1)
# #     W[i,i] = 1/ momentData2[i]
# # end
#
# error = transpose(jac) * W * jac ;





#=***************************************************=#
#= send email after completing the optimization =#
opt = SendOptions(
  isSSL = true,
  username = "juliacodeserver@gmail.com",
  passwd = "JuliaCodeServer")
#Provide the message body as RFC5322 within an IO
body = IOBuffer(
  "Date: Fri, 18 Oct 2013 21:44:29 +0100\r\n" *
  "From: You <juliacodeserver@gmail.com>\r\n" *
  "To: ehsansaboori75@gmail.com\r\n" *
  "Subject: Julia Code\r\n" *
  "\r\n" *
  "Julia code completed on the server\r\n")
url = "smtps://smtp.gmail.com:465"
rcpt = ["<ehsansaboori75@gmail.com>", "<z.shamlooo@gmail.com>" ]
from = "<juliacodeserver@gmail.com>"
resp = send(url, rcpt, from, body, opt)








end

##############################################################
# #= ploting the output =#
#
# # Pkg.add("Plots")
# using Plots
# using PyCall
#
# # Pkg.add("PyPlot")
# # using PyPlot
# pyplot()
# Plots.PyPlotBackend()
#
# # name of each array column in a dictionary #
# wageCol = Dict(
#     "age"         => 1,
#     "educated"    => 2,
#     "collar"      => 3,
#     "incomeData"  => 4,
#     "devData"     => 5,
#     "incomeSim"   => 6,
#     "devSim"      => 7
# );
# choiceCol = Dict(
#     "age"       => 1,
#     "educated"  => 2,
#     "homeData"  => 3,
#     "studyData" => 4,
#     "whiteData" => 5,
#     "blueData"  => 6,
#     "milData"   => 7,
#     "homeSim"   => 8,
#     "studySim"  => 9,
#     "whiteSim"  => 10,
#     "blueSim"   => 11,
#     "milSim"    => 12,
# ) ;
#
# choiceMoment[ :, 1:7] =choiceMomentData[: ,1:7] ;
#
# ## delimit
# #*****************************************************
# # plot log average income of individuals in each age
# #   By education category
# #*****************************************************
# # collar= 0
# for collar in [0,1]
#     plot(0,0)
#     for educated in [0,1]
#         plot!(wageMoment[(wageMoment[:,wageCol["collar"]].==collar).&
#                         (wageMoment[:,wageCol["educated"]].==educated),wageCol["age"]],
#             log.(wageMoment[(wageMoment[:,wageCol["collar"]].==collar).&
#                         (wageMoment[:,wageCol["educated"]].==educated),wageCol["incomeData"]]) ,
#             label=(educated==1 ? "data moment educated" : "data moment not educated") ,
#             color = (educated==1 ? "green" : "blue"),
#             alpha= 0.8,
#             marker=:circle,
#             markersize= 4,
#             legendfontsize=6,
#             w= 1)
#
#         plot!(wageMoment[(wageMoment[:,wageCol["collar"]].==collar).&
#                         (wageMoment[:,wageCol["educated"]].==educated),wageCol["age"]],
#             log.(wageMoment[(wageMoment[:,wageCol["collar"]].==collar).&
#                         (wageMoment[:,wageCol["educated"]].==educated),wageCol["incomeSim"]]) ,
#             label=(educated==1 ? "model moment educated" : "model moment not educated") ,
#             color = (educated==1 ? "green" : "blue"),
#             alpha= 0.8,
#             marker=(:rect),#, Plots.stroke(1, :black)),
#             markersize= 4,
#             w= 1,
#             line = (:dash, 1.2)
#             )
#     end
#
#     if collar == 1
#         collar= "blue"
#     elseif collar == 0
#         collar= "white"
#     end
#     title!(collar*"-collar mean income")
#     xlabel!("age")
#     ylabel!("Mean income (Rial)")
#     ylims!(16 , 19.5)
#     xticks!(16:50)
#     xgrid!(false)
#     # savefig("/home/ehsan/Dropbox/Labor/Codes/ModelSimulation/Results/3.1 "*collar*"CollarMeanWage.png")
#     savefig("/home/sabouri/thesis/Results/3.1 "*collar*"CollarMeanWage.png")
# end
#
# ## delimit
# #*****************************************************
# # plot logarithm's income standard deviation
# #   of individuals in each age
# #   By education category
# #*****************************************************
# # collar= 0
# for collar in [0,1]
#     plot(0,0)
#     for educated in [0,1]
#         plot!(wageMoment[(wageMoment[:,wageCol["collar"]].==collar).&
#                         (wageMoment[:,wageCol["educated"]].==educated),wageCol["age"]],
#             log.(wageMoment[(wageMoment[:,wageCol["collar"]].==collar).&
#                         (wageMoment[:,wageCol["educated"]].==educated),wageCol["devData"]]) ,
#             label=(educated==1 ? "data moment educated" : "data moment not educated") ,
#             color = (educated==1 ? "green" : "blue"),
#             alpha= 0.8,
#             marker=:circle,
#             markersize= 4,
#             legendfontsize=6,
#             w= 1)
#
#         plot!(wageMoment[(wageMoment[:,wageCol["collar"]].==collar).&
#                         (wageMoment[:,wageCol["educated"]].==educated),wageCol["age"]],
#             log.(wageMoment[(wageMoment[:,wageCol["collar"]].==collar).&
#                         (wageMoment[:,wageCol["educated"]].==educated),wageCol["devSim"]]) ,
#             label=(educated==1 ? "model moment educated" : "model moment not educated") ,
#             color = (educated==1 ? "green" : "blue"),
#             alpha= 0.8,
#             marker=:rect,
#             markersize= 4,
#             w= 1,
#             line = (:dash, 1.2)
#             )
#     end
#
#     if collar == 1
#         collar= "blue"
#     elseif collar == 0
#         collar= "white"
#     end
#     title!(collar*"-collar income standard deviation")
#     xlabel!("age")
#     ylabel!("Mean income (Rial)")
#     ylims!(16, 19.5)
#     xticks!(16:50)
#     xgrid!(false)
#
#     # savefig("/home/ehsan/Dropbox/Labor/Codes/ModelSimulation/Results/3.2 "*collar*"CollarDeviationWage.png")
#     savefig("/home/sabouri/thesis/Results/3.2 "*collar*"CollarDeviationWage.png")
#
# end
#
#
# ## delimit
# #*****************************************************
# # plot each alternative share
# #   of individuals in each age
# #   By education category
# #*****************************************************
# a=["home", "study", "white", "blue" , "mil"] ;
# for i = 1:5
# # i= 2
# # educated = 0
#     alternative= a[i]
#     plot(0,0)
#     for educated in [-1,0,1]
#         plot!(choiceMoment[choiceMoment[:,choiceCol["educated"]].==educated
#                 ,choiceCol["age"]],
#             choiceMoment[choiceMoment[:,choiceCol["educated"]].==educated
#                 ,choiceCol[alternative*"Data"]],
#             label= "", #"data moment $educated"
#             color= (educated == 1 ? "green" : educated==0 ? "blue" : "black") ,
#             alpha= 0.7,
#             marker=:circle ,
#             markersize= 4,
#             legendfontsize=7,
#             w= 1)
#
#         plot!(choiceMoment[choiceMoment[:,choiceCol["educated"]].==educated
#             ,choiceCol["age"]],
#             choiceMoment[choiceMoment[:,choiceCol["educated"]].==educated
#                 ,choiceCol[alternative*"Sim"]],
#             label= "model moment  $educated",
#             color = (educated == 1 ? "green" : educated==0 ? "blue" : "black") ,
#             alpha= 0.7 ,
#             marker=:rect ,
#             markersize= 4,
#             w= 1,
#             line = (:dash, 1.2))
#     end
#     title!(alternative*" share")
#     xlabel!("age")
#     ylabel!("percentage share of alternative")
#     xticks!(16:50)
#     xgrid!(false)
#
#     # savefig("/home/ehsan/Dropbox/Labor/Codes/ModelSimulation/Results/2."*string(i)*" "*alternative*".png")
#     savefig("/home/sabouri/thesis/Results/2."*string(i)*" "*alternative*".png")
#
# end
# ## delimit cr






################################################################################
# #*************************************************************
# #*************************************************************
# # calculating moment from simulation
# # code for reading in Linux operating system #
# wageMomentData= readdlm("/home/ehsan/Dropbox/Labor/Codes/Moments/wageMoment.csv",',')       ;
# choiceMomentData = readdlm("/home/ehsan/Dropbox/Labor/Codes/Moments/choiceMoment.csv",',')  ;
#
# # calculating moment from simulation
# # we build some arrays to put data moment and model moment aside
#
# # generating an Array named wageMoment
# # to store average income of the simulated moments
# # also we embed data moment in this array
# # wageMomentData is the given wage moment from data
# wageMoment= Array{Float64,2}(undef, (size(wageMomentData,1),5))
# wageCol = Dict(
#     "age"         => 1,
#     "educated"    => 2,
#     "collar"      => 3,
#     "incomeData"  => 4,
#     "incomeSim"   => 5,
# )
# # wageMoment[:,1:4]= wageMomentData[:,1:4];
# wageMoment[:,1]= wageMomentData[:,1]
# wageMoment[:,3:4]= wageMomentData[:,2:3]
#
#
# # generating an Array named choiceMoment
# # to store simulated share of alternatives
# # also we embed data moment in this array
# # dhoiceMomentData is the given alternative share moment from data
# choiceMoment= Array{Float64,2}(undef, (size(choiceMomentData,1),12) )
# choiceCol = Dict(
#     "age"       => 1,
#     "educated"  => 2,
#     "homeData"  => 3,
#     "studyData" => 4,
#     "whiteData" => 5,
#     "blueData"  => 6,
#     "milData"   => 7,
#     "homeSim"   => 8,
#     "studySim"  => 9,
#     "whiteSim"  => 10,
#     "blueSim"   => 11,
#     "milSim"    => 12,
# )
# # choiceMoment[ :, 1:7] =choiceMomentData[: ,1:7]
# choiceMoment[ :, 1] =choiceMomentData[: ,1]
# choiceMoment[ :, 3:7] =choiceMomentData[: ,2:6]
#
#
# # removing share below 4 percent for two reason:
# # it is not informative about distribution of choices
# # also increases the error of coumputation daramatically large if they remain
# for i in 3:6
#     choiceMoment[(choiceMoment[:,i].<0.01) ,i] .= NaN
# end
#
#
#
#
# # each column of simulated data is as follows:
# simCol = Dict(
#     "age"      => 1,
#     "educ"     => 2,
#     "x3"       => 3,
#     "x4"       => 4,
#     "choice"   => 5,
#     "income"   => 6,
#     "educated" => 7,
#     "x5"       => 8,
# )
# ageInterval= unique(choiceMoment[:,choiceCol["age"]])
# ageMax= maximum(ageInterval)
#
# for age in ageInterval
#
#     ## mean income for each occupation moment condition ##
#
#     # for educated in unique(wageMoment[ wageMoment[:,wageCol["age"]].== age , wageCol["educated"] ])
#     for collar in unique(wageMoment[ wageMoment[:,wageCol["age"]].== age , wageCol["collar"] ])
#
#         # mapping each collar code to choice alternative in the model
#         # in the file working with data, we defined:
#         # colar 0 : white-collar occupation
#         # colar 1 : blure-collar occupation
#         # colar 2 : compulsory military service
#         if collar == 0
#             choice=3
#         elseif collar==1
#             choice=4
#         elseif collar==2
#             choice=-1
#         end
#
#         flag = sim[ (sim[:,simCol["age"]].==age).&
#                     (sim[:,simCol["choice"]].== choice) , simCol["income"]]
#
#         wageMoment[ (wageMoment[:,wageCol["age"]].==age).&
#                     (wageMoment[:,wageCol["collar"]].==collar)
#                     , wageCol["incomeSim"]] .= mean(filter(!isnan, flag ))
#
#     end #for collar
#
#
#
#     ## share of each alternative moment conditions ##
#
#     flag2= [count(x->x==i , sim[(sim[:,simCol["age"]].==age), simCol["choice"]]) for i in 1:5]
#     choiceMoment[(choiceMoment[:,choiceCol["age"]].==age), 8:12] = flag2/sum(flag2)
#
# end#for age
#
#
#
#
# # delimit
# a=["home", "study", "white", "blue" , "mil"] ;
# for i = 1:5
#     alternative= a[i]
#
#     plot(choiceMoment[:,choiceCol["age"]],
#         choiceMoment[:,choiceCol[alternative*"Data"]],
#         label= "data moment",
#         marker=:circle ,
#         color = "green",
#         alpha = 0.7,
#         markersize= 4,
#         w=1)
#
#     plot!(choiceMoment[:,choiceCol["age"]],
#         choiceMoment[:,choiceCol[alternative*"Sim"]],
#         label= "model moment",
#         marker=:rect ,
#         color = "blue",
#         alpha = 0.7,
#         markersize= 4,
#         w=1,
#         line = (:dash, 1.2))
#     title!(alternative*" share")
#     xlabel!("age")
#     ylabel!("percentage share of alternative")
#     xticks!(16:50)
#     xgrid!(false)
#     savefig("/home/ehsan/Dropbox/Labor/Codes/ModelSimulation/Results/1."*string(i)*" "*alternative*".png")end
# ## delimit cr
#






















################################################
## This part is for irrelevand code
#


#
#
# function f(x)
#     # if length(grad) > 0
#     #     grad[1] = 2*x[1] -2
#     #     grad[2] = 600*x[2]^5
#     # end
#     return x[1]^2 - 2*x[1] + 100*x[2]^6 #+ y
# end
# #
# # Optim.optimize(x -> f(x), [-1.0,1.0], [-1.0,1.0])
# #
# # Optim.optimize(x-> x^2, -2.0, 1.0)
#
#
# oopt = NLopt.Opt(:LD_MMA, 2)
#
# oopt.min_objective = f
#
# oopt.lower_bounds = [-100,-100]
# oopt.upper_bounds = [100,100]
# oopt.ftol_rel = 1e-4
# oopt.stopval = -5.0
#
# param = [10,1]
# (optf,optx,ret) = NLopt.optimize(oopt, param)
#
# f(optx)
#
# optx
# ret
#
# numevals = oopt.numevals
# println("got $optf at $optx after $numevals iterations (returned $ret)"
#
#
#
#
#
# function rosenbrock2d(x)
#   return (1.0 - x[1])^2 + 100.0 * (x[2] - x[1]^2)^2
# end
#
# Optim.optimize(x -> rosenbrock2d(x)
#             ,[2.0,1.0]
#             ,NelderMead()
#             ,Optim.Options(
#             f_tol= 1
#             ,g_tol = 1
#             ,allow_f_increases= true
#             ,iterations = 100000
#             ))
#
#
# res = bboptimize(x->rosenbrock2d(x,0); SearchRange = (-5.0, 5.0), NumDimensions = 2)
#
# best_candidate(res)
# best_fitness(res)
#
#
#
# bboptimize(rosenbrock2d; SearchRange = [(-5.0, 5.0), (-2.0, 2.0)])
#
#
# a = [-5, -2]
# b = [5, 2]
#
# bboptimize(rosenbrock2d; SearchRange = bounds )
#
# bounds = [(i*1.0,i*1.0) for i in 1:2]
# for i in 1:2
#     bounds[i] = (a[i], b[i])
# end








## GPU Programming with Julia

# # Lower bound for the capital grid:
# lb = 0.001;
# # Upper bound for the capital grid:
# ub = 10;
# # Number of grid points:
# grid_size = 1000;
# # Create an evenly spaced capital grid:
# grid_w = Array{Float32}(collect(range(lb, ub, length=grid_size)));
#
# alpha = 0.5;
# beta = 0.7;
#
#
#
# function cpu_vfi(grid_::Array{Float32}, alpha::Float64, beta::Float64, maxiter_, prec_)
#
#     SIZE_GRID = size(grid_,1);
#
#     V = Array{Float32}(ones(SIZE_GRID, 1));
#
#     for it = 1:maxiter_
#
#         for i = 1:SIZE_GRID
#             tmp = grid_[i].^alpha .- grid_;
#             tmp_max = -Inf;
#             for (j, point_j) in enumerate(tmp)
#                 if point_j > 0
#                     tmp_comp = log.(point_j) .+ beta.*V[j];
#                     tmp_max = max(tmp_comp, tmp_max);
#                 end
#             end
#             V[i] = tmp_max;
#         end
#
#     end
#
#     return V
#
# end
#
# cpu_time = @elapsed CPU_OUT = cpu_vfi(grid_w, alpha, beta, 100, 0.0001);
#
# using CUDA
# using GPUArrays
#
#
# function gpu_vfi(grid_::Array{Float32}, alpha::Float32, beta::Float32, maxiter_, prec_)
#
#     SIZE_GRID = size(grid_, 1);
#
#     # V = ones(CLArray{Float32}, SIZE_GRID, 1);
#     V = CUDA.fill(1.0f0, SIZE_GRID)
#
#     # grid = CLArray(Array{Float32}(grid_));
#     grid = CuArray(Array{Float32}(grid_));
#
#     for iter_ in 1:maxiter_
#
#         # Write kernel for GPU manually:
#         gpu_call(grid, (grid, V, Float32(alpha), Float32(beta), UInt32(SIZE_GRID))) do state, grid, V, alpha, beta, SIZE_GRID
#             idx = linear_index(grid)
#             tmp_max = Float32(-Inf);
#             @inbounds begin
#                 for i = 1:SIZE_GRID
#                     tmp_i = log(grid[idx]^alpha - grid[i]) + beta*V[i];
#                     if tmp_i > tmp_max
#                         tmp_max = tmp_i;
#                     end
#                 end
#                 V[idx] = tmp_max;
#             end
#             return
#         end
#
#     end
#
#     return Array{Float32}(V)
#
# end
