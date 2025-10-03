
#=***************************************************

    The replication code for the follwoing research paper:

     "
        The Effect of Compulsory Military Service on Education and Income of Men in Iran;
        A Structural Model Estimation
     "

    Authors:
        Ehsan Sabouri Kenari
        Mohammad Hoseini

    Contact us at:
        ehsan.sabouri@iies.su.se

****************************************************=#




# include the required libraries 
include("S1_2_libraries.jl")

# include the model parameters
include("S1_3_model_parameters.jl")

# include data moments needed for the estimation
include("S1_4_moments.jl")




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

#=
    The codes needed for running on the server
    and also extracting the simulation results from Server
    to my ubuntu operating system.
=#

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

import Pkg
# Pkg.add("Distributions")
# Pkg.add("StatsBase")
# Pkg.add("Compat")
# Pkg.add("Dates")

using Random
# using Statistics
# using BenchmarkTools
# using Profilex
using Distributions
# using LinearAlgebra
using StatsBase
# using Cubature
using DelimitedFiles
# using NamedArrays
using Dates

#= Optimization packages =#
# using LeastSquaresOptim
# using Optim
# using NLopt
# using BlackBoxOptim

# using Compat.Dates
# using SharedArrays
using LinearAlgebra
# using Test
# using SMTPClient # for sending email
# using Distributed

using CUDA

################################################################################
#=
    Define a function to calculate the mean of the maximum over some vectors.
=#
# @everywhere function MeanMaximum(array)
#     length_each_vector = size(array[1])[1]
#     number_of_vector = size(array)[1]
#
#     s = 0.0
#     @simd for row in 1:length_each_vector
#         max = 0.0
#         for vector in 1:number_of_vector
#             if array[vector][row] > max
#                 max = array[vector][row]
#             end
#         end
#         s += max
#     end
#     value= s/length_each_vector
#     return value
# end


#= contemporaneous utility function =#

#= utility when choice is stay home =#
function util1GPU(p::NamedTuple, age, educ, LastChoice, ε1; type=1)
    # util=  p.ω1[type] + p.α11*(age <= 19) + p.α12*(educ>=13) + p.α13*(age>35)+ ε1 - p.α14*(age-26)*(age>=26)*(age<=35) #+ 1.0e7*(LastChoice==5)
    util=  p.ω1[type] + p.α11*(age<18)*(19-age) + p.α12*(educ>=13) - p.α13*( (age-22)*(age>=22) - (age-50)*(age>=50) ) + ε1 + p.α14*(age==19)*(educ>=8)#+ 1.0e7*(LastChoice==5)
    return util
end

#= utility when choice is study =#
function util2GPU(p::NamedTuple, LastChoice, educ, ε2, age; type=1)
    util= (p.ω2[type] - p.α21*(LastChoice != 2)- p.tc1*(educ>12)- p.tc2*(educ>16) + p.α30study*(age>=30) ) + ε2
    return util
end


#= utility when choice is whitel-collar occupation =#
function wageWhiteCollar(p::NamedTuple, educ, x3, x4, LastChoice, ε3; type=1)
    wage = ( exp((p.ω3[type]+ p.α31*educ+ p.α32*x3+ p.α33*x4+ p.α34*(x3^2)+ p.α35*(x4^2))- (p.α36- p.α37*(educ>=16))*(x3==0)
        - p.α38*(LastChoice != 3)
        + p.α22*(educ>=12)+ p.α23*(educ>=16) + ε3) ) ;
    return wage
end

function util3GPU(p::NamedTuple, x3, x4, LastChoice, educ, ε3; type=1)
    util= (wageWhiteCollar(p, educ, x3, x4, LastChoice, ε3; type=type) + p.α3)
    return util
end

#= utility when choice is blue-collar occupation =#
function wageBlueCollar(p::NamedTuple, educ, x3, x4, LastChoice, ε4; type=1)
    wage = ( exp((p.ω4[type]+ p.α41*educ+ p.α42*x3+ p.α43*x4+ p.α44*(x3^2)+ p.α45*(x4^2))- (p.α46- p.α47*(educ>=16))*(x4==0)
    - p.α48*(LastChoice != 4)
    + p.α24*(educ>=12)+ p.α25*(educ>=16)+ ε4 ) ) ;
    return wage
end

function util4GPU(p::NamedTuple, x3, x4, LastChoice, educ, ε4; type=1)
    util= (wageBlueCollar(p, educ, x3, x4, LastChoice, ε4; type=type) + p.α4)
    return util
end

#= utility when choice is compulsory military service =#
function util5GPU(p::NamedTuple, educ, ε5)
    util= p.α50 + p.α51*(educ>12) + p.α52*(educ>16) + ε5 #+ 7.7071807618222445e7*(educ>20)
    return util
end


################################################################################
#=
    conscription group 2 value function and solve Emax function
    group 2: Not obligated to attend conscription
    value function: given state vector at an age, it denotes the maxiual value
    at age a over all possible career decisions.
=#



function EmaxGroup2Index(age, educ, LastChoice, x3, x4, type)

    typeCount              = 3
    ageStateCount          = 49
    educStateCount         = 23
    LastChoiceStateCount   = 4
    x3StateCount           = 31
    x4StateCount           = 31

    enumerator = (
        (x4+1) +
        (x3)*           x4StateCount +
        (LastChoice-1)* x4StateCount* x3StateCount +
        (educ)*         x4StateCount* x3StateCount* LastChoiceStateCount +
        (age-17)*       x4StateCount* x3StateCount* LastChoiceStateCount* educStateCount +
        (type-1)*       x4StateCount* x3StateCount* LastChoiceStateCount* educStateCount * ageStateCount
    )
    return enumerator
end




#= value function for type 2: Obligated to attent conscription =#
function valueFunctionGroup2!(
    p::NamedTuple,
    epssolve,
    age,
    Emax)


    #***********************************#
    enum = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    stride = blockDim().x * gridDim().x

    typeCount              = 3
    ageStateCount          = 49
    educStateCount         = 23
    LastChoiceStateCount   = 4
    x3StateCount           = 31
    x4StateCount           = 31


    educ = div(enum-1, LastChoiceStateCount*x3StateCount*x4StateCount*typeCount)
    rem  = mod(enum-1, LastChoiceStateCount*x3StateCount*x4StateCount*typeCount)

    LastChoice   = div(rem, x3StateCount*x4StateCount*typeCount) + 1
    rem          = mod(rem, x3StateCount*x4StateCount*typeCount)

    x3   = div(rem, x4StateCount*typeCount)
    rem  = mod(rem, x4StateCount*typeCount)

    x4   = div(rem, typeCount)
    rem  = mod(rem, typeCount)

    type = rem + 1

    EmaxIndex = EmaxGroup2Index(age, educ, LastChoice, x3, x4, type)

    if (educ + x3 + x4 + 5) > age
        return nothing
    end

    if enum > (educStateCount* LastChoiceStateCount* x3StateCount* x4StateCount * typeCount)
        return nothing
    end

    # if EmaxIndex > (ageStateCount* educStateCount* LastChoiceStateCount* x3StateCount* x4StateCount * typeCount)
    #     return nothing
    # end



    #***********************************#
    # function MeanMonteCarlo(x)
    #     a= [1,x,x^2]
    #     return maximum(a)
    # end
    # a = MeanMonteCarlo(1)



    value= -1 # this is for when no if conditon binds
    if age == 65

        if educ < 22
            s = 0.0
            for row in 1:p.MonteCarloCount
                ε1 = epssolve[1,row]
                VF1 = util1GPU(p, age, educ, LastChoice, ε1; type=type)
                ε2 = epssolve[2,row]
                VF2 = util2GPU(p, LastChoice, educ+1, ε2, age; type=type)
                ε3 = epssolve[3,row]
                VF3 = util3GPU(p, x3, x4, LastChoice, educ, ε3; type=type)
                ε4 = epssolve[4,row]
                VF4 = util4GPU(p, x3, x4, LastChoice, educ, ε4; type=type)

                s += max(VF1, VF2, VF3, VF4)
            end
            value = s/p.MonteCarloCount
        else
            s = 0.0
            for row in 1:p.MonteCarloCount
                ε1 = epssolve[1,row]
                VF1 = util1GPU(p, age, educ, LastChoice, ε1; type=type)
                ε3 = epssolve[3,row]
                VF3 = util3GPU(p, x3, x4, LastChoice, educ, ε3; type=type)
                ε4 = epssolve[4,row]
                VF4 = util4GPU(p, x3, x4, LastChoice, educ, ε4; type=type)

                s += max(VF1, VF3, VF4)
            end
            value = s/p.MonteCarloCount
        end
    else

        enum1 = EmaxGroup2Index(age+1, educ, 1, x3, x4, type)
        enum2 = EmaxGroup2Index(age+1, (educ+1*(educ!=22)), 2, x3, x4, type)
        enum3 = EmaxGroup2Index(age+1, educ, 3, (x3+1*(x3!=p.x3Max)), x4, type)
        enum4 = EmaxGroup2Index(age+1, educ, 4, x3, (x4+1*(x4!=p.x4Max)), type)

        EmaxNext1 = Emax[enum1]

        EmaxNext2 = Emax[enum2]

        EmaxNext3 = Emax[enum3]

        EmaxNext4 = Emax[enum4]
        # xx  = [1,2,4]

        if educ < 22
            s = 0.0
            for row in 1:p.MonteCarloCount
                ε1 = epssolve[1,row]
                VF1 = util1GPU(p, age, educ, LastChoice, ε1; type=type)
                VF1 = VF1 + p.δ * EmaxNext1
                ε2 = epssolve[2,row]
                VF2 = util2GPU(p, LastChoice, educ+1, ε2, age; type=type)
                VF2 = VF2 + p.δ * EmaxNext2
                ε3 = epssolve[3,row]
                VF3 = util3GPU(p, x3, x4, LastChoice, educ, ε3; type=type)
                VF3 = VF3 + p.δ * EmaxNext3
                ε4 = epssolve[4,row]
                VF4 = util4GPU(p, x3, x4, LastChoice, educ, ε4; type=type)
                VF4 = VF4 + p.δ * EmaxNext4

                s += max(VF1, VF2, VF3, VF4)
            end
            value = s/p.MonteCarloCount
        else
            s = 0.0
            for row in 1:p.MonteCarloCount
                ε1 = epssolve[1,row]
                VF1 = util1GPU(p, age, educ, LastChoice, ε1; type=type)
                VF1 = VF1 + p.δ * EmaxNext1
                ε3 = epssolve[3,row]
                VF3 = util3GPU(p, x3, x4, LastChoice, educ, ε3; type=type)
                VF3 = VF3 + p.δ * EmaxNext3
                ε4 = epssolve[4,row]
                VF4 = util4GPU(p, x3, x4, LastChoice, educ, ε4; type=type)
                VF4 = VF4 + p.δ * EmaxNext4

                s += max(VF1, VF3, VF4)
            end
            value = s/p.MonteCarloCount
        end

    end


    # if EmaxIndex <= (ageStateCount* educStateCount* LastChoiceStateCount* x3StateCount* x4StateCount * typeCount)
    Emax[EmaxIndex] = value
    # end

    return nothing
end


#
# a = [1,2,3,4,5,6]
# maximum(a[[1,4]])



#= solve Emax for conscription group 2: Not obligated to attent conscription =#
function solveGroup2AllType(p::NamedTuple, epssolve)


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

    ageState  = 65 :-1 :17     # age age of the individual
    educState = 0 :1 :22       # educ number of completed education
    LastChoiceState   = 1:4    # LastChoice : alternative chosen in the last period
    x3State   = 0 :1 : p.x3Max   # x3 experience in white-collar
    x4State   = 0 :1 : p.x4Max   # x4 experience in blue-collar

    ageStateCount  = length(ageState)
    educStateCount = length(educState)
    LastChoiceStateCount   = length(LastChoiceState)
    x3StateCount   = length(x3State)
    x4StateCount   = length(x4State)
    typeCount      = 3


    stateSpaceSize = ageStateCount* educStateCount* LastChoiceStateCount* x3StateCount* x4StateCount * typeCount

    Emax = CUDA.fill(1.0, (stateSpaceSize, 1))
    epssolve = CuArray(epssolve)


    numblocks = ceil(Int, educStateCount*LastChoiceStateCount*x3StateCount*x4StateCount*typeCount/256)


    for age in ageState


        @cuda threads=256 blocks=numblocks valueFunctionGroup2!(p,
                                                            epssolve,
                                                            age,
                                                            Emax)


        synchronize()

    end #age


    return Array(Emax)

end


# #= test section =#
# #= here we check whether Emax function is working perfect or not. =#
# include("/home/sabouri/Dropbox/Labor/Codes/GitRepository/modelParameters.jl")
# epsSolveMean=[0.0, 0.0, 0.0, 0.0] ;
# epsSolveσ=[ σ1   0.0  0.0   0.0 ;
#             0.0  σ2   0.0   0.0 ;
#             0.0  0.0  σ3    σ34 ;
#             0.0  0.0  σ34   σ4  ] ;
#
# M = 200 ;
# epssolve=rand(MersenneTwister(1234),MvNormal(epsSolveMean, epsSolveσ) , M) ;
#
# for i in 1:2
#     print("Emax Group 2 calculation: \n")
#     start = Dates.unix2datetime(time())
#
#     EmaxGroup2GPU = solveGroup2AllType(14.0,14.0,14.0,14.0, α11, α12, α13,
#                     14.0,14.0,14.0,14.0, α21, tc1T1, tc2, α22, α23, 0, α25, α30study,
#                     α3, 14.0,14.0,14.0,14.0, α31, α32, α33, α34, α35, 0, 0, 0,
#                     α4, 14.0,14.0,14.0,14.0, α41, α42, α43, α44, α45, 0, 0, 0,
#                     0.92,
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

function EmaxGroup1Index(age, educ, LastChoice, x3, x4, x5, type, homeSinceSchool)

    typeCount              = 3
    ageStateCount          = 49
    educStateCount         = 23
    LastChoiceStateCount   = 5
    x3StateCount           = 31
    x4StateCount           = 31
    x5StateCount           = 3
    # homeSinceSchoolCount   = 2 #4 + 1

    enumerator = (
        (x5+1) +
        (x4)             * x5StateCount +
        (x3)             * x5StateCount* x4StateCount +
        (LastChoice-1)   * x5StateCount* x4StateCount* x3StateCount +
        (educ)           * x5StateCount* x4StateCount* x3StateCount* LastChoiceStateCount +
        (age-17)         * x5StateCount* x4StateCount* x3StateCount* LastChoiceStateCount* educStateCount +
        (type-1)         * x5StateCount* x4StateCount* x3StateCount* LastChoiceStateCount* educStateCount* ageStateCount +
        (homeSinceSchool)* x5StateCount* x4StateCount* x3StateCount* LastChoiceStateCount* educStateCount* ageStateCount* typeCount
    )
    return enumerator
end






#= value function for conscription goup 2: obligated to attend conscription =#
function valueFunctionGroup1!(p::NamedTuple,
                            epssolve,
                            age,
                            Emax)

    # print(p.homeSinceSchoolMax," sdfds sdfds dfs ")
    #***********************************#
    enum = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    stride = blockDim().x * gridDim().x

    typeCount              = 3
    ageStateCount          = 49
    educStateCount         = 23
    LastChoiceStateCount   = 5
    x3StateCount           = 31
    x4StateCount           = 31
    x5StateCount           = 3
    homeSinceSchoolCount   = p.homeSinceSchoolMax+1

    educ = div(enum-1, LastChoiceStateCount*x5StateCount*x3StateCount*x4StateCount*typeCount*homeSinceSchoolCount)
    rem  = mod(enum-1, LastChoiceStateCount*x5StateCount*x3StateCount*x4StateCount*typeCount*homeSinceSchoolCount)

    LastChoice   = div(rem, x5StateCount*x3StateCount*x4StateCount*typeCount*homeSinceSchoolCount) + 1
    rem          = mod(rem, x5StateCount*x3StateCount*x4StateCount*typeCount*homeSinceSchoolCount)

    x3   = div(rem, x5StateCount*x4StateCount*typeCount*homeSinceSchoolCount)
    rem  = mod(rem, x5StateCount*x4StateCount*typeCount*homeSinceSchoolCount)

    x4   = div(rem, x5StateCount*typeCount*homeSinceSchoolCount)
    rem  = mod(rem, x5StateCount*typeCount*homeSinceSchoolCount)

    x5   = div(rem, typeCount*homeSinceSchoolCount)
    rem  = mod(rem, typeCount*homeSinceSchoolCount)

    type = div(rem, homeSinceSchoolCount) + 1
    rem  = mod(rem, homeSinceSchoolCount)

    homeSinceSchool = rem

    EmaxIndex = EmaxGroup1Index(age, educ, LastChoice, x3, x4, x5, type, homeSinceSchool)

    if (educ + x3 + x4 + x5 + 5) > age
        return nothing
    end

    if enum > (educStateCount* LastChoiceStateCount* x3StateCount* x4StateCount* x5StateCount* typeCount* homeSinceSchoolCount)
        # @cuprintln(EmaxIndex, "  aaa\n")
        return nothing
    end

    # if EmaxIndex > (ageStateCount* educStateCount* LastChoiceStateCount* x3StateCount* x4StateCount* x5StateCount * typeCount*homeSinceSchoolCount)
    #     @cuprintln(EmaxIndex, "  bbb\n")
    #     return nothing
    # end
    #
    # if EmaxIndex < 1
    #     @cuprintln(EmaxIndex, "  ccc\n")
    #     return nothing
    # end


    #***********************************#

    # MonteCarloCount = size(epssolve, 2)

    value= -1 # this is for when no if conditon binds
    if age == 65
        if educ < 22
            s = 0.0
            for row in 1:p.MonteCarloCount
                ε1 = epssolve[1,row]
                VF1 = util1GPU(p, age, educ, LastChoice, ε1; type=type)
                ε2 = epssolve[2,row]
                VF2 = util2GPU(p, LastChoice, educ+1, ε2, age; type=type)
                ε3 = epssolve[3,row]
                VF3 = util3GPU(p, x3, x4, LastChoice, educ, ε3; type=type)
                ε4 = epssolve[4,row]
                VF4 = util4GPU(p, x3, x4, LastChoice, educ, ε4; type=type)

                s += max(VF1, VF2, VF3, VF4)
            end
            value = s/p.MonteCarloCount
        else
            s = 0.0
            for row in 1:p.MonteCarloCount
                ε1 = epssolve[1,row]
                VF1 = util1GPU(p, age, educ, LastChoice, ε1; type=type)
                ε3 = epssolve[3,row]
                VF3 = util3GPU(p, x3, x4, LastChoice, educ, ε3; type=type)
                ε4 = epssolve[4,row]
                VF4 = util4GPU(p, x3, x4, LastChoice, educ, ε4; type=type)

                s += max(VF1, VF3, VF4)
            end
            value = s/p.MonteCarloCount
        end

    else

        enum1 = EmaxGroup1Index(age+1, educ, 1, x3, x4, x5, type, homeSinceSchool + 1*(age>=19)*(homeSinceSchool<p.homeSinceSchoolMax) )
        enum2 = EmaxGroup1Index(age+1, (educ+1*(educ!=22)), 2, x3, x4, x5, type, homeSinceSchool)
        enum3 = EmaxGroup1Index(age+1, educ, 3, (x3+1*(x3!=p.x3Max)), x4, x5, type, homeSinceSchool)
        enum4 = EmaxGroup1Index(age+1, educ, 4, x3, (x4+1*(x4!=p.x4Max)), x5, type, homeSinceSchool)
        enum5 = EmaxGroup1Index(age+1, educ, 5, x3, x4, (x5+1*(x5!=2)), type, homeSinceSchool)

        EmaxNext1 = Emax[enum1]
        EmaxNext2 = Emax[enum2]
        EmaxNext3 = Emax[enum3]
        EmaxNext4 = Emax[enum4]
        EmaxNext5 = Emax[enum5]


        #####
        if age > 18
            if     x5 == 2
                if educ < 22
                    s = 0.0
                    for row in 1:p.MonteCarloCount
                        ε1 = epssolve[1,row]
                        VF1 = util1GPU(p, age, educ, LastChoice, ε1; type=type)
                        VF1 = VF1 + p.δ * EmaxNext1
                        ε2 = epssolve[2,row]
                        VF2 = util2GPU(p, LastChoice, educ+1, ε2, age; type=type)
                        VF2 = VF2 + p.δ * EmaxNext2
                        ε3 = epssolve[3,row]
                        VF3 = util3GPU(p, x3, x4, LastChoice, educ, ε3; type=type)
                        VF3 = VF3 + p.δ * EmaxNext3
                        ε4 = epssolve[4,row]
                        VF4 = util4GPU(p, x3, x4, LastChoice, educ, ε4; type=type)
                        VF4 = VF4 + p.δ * EmaxNext4

                        s += max(VF1, VF2, VF3, VF4)
                    end
                    value = s/p.MonteCarloCount
                else
                    s = 0.0
                    for row in 1:p.MonteCarloCount
                        ε1 = epssolve[1,row]
                        VF1 = util1GPU(p, age, educ, LastChoice, ε1; type=type)
                        VF1 = VF1 + p.δ * EmaxNext1
                        ε3 = epssolve[3,row]
                        VF3 = util3GPU(p, x3, x4, LastChoice, educ, ε3; type=type)
                        VF3 = VF3 + p.δ * EmaxNext3
                        ε4 = epssolve[4,row]
                        VF4 = util4GPU(p, x3, x4, LastChoice, educ, ε4; type=type)
                        VF4 = VF4 + p.δ * EmaxNext4

                        s += max(VF1, VF3, VF4)
                    end
                    value = s/p.MonteCarloCount
                end
            # elseif x5 == 1
            #     s = 0.0
            #     for row in 1:p.MonteCarloCount
            #         ε5 = epssolve[5,row]
            #         VF5 = util5GPU(p, educ, ε5)
            #         VF5 = VF5 + p.δ * EmaxNext5
            #
            #         s += max(VF5)
            #     end
            #     value = s/p.MonteCarloCount
            else#if x5 == 0
                if educ == 22
                    if homeSinceSchool==p.homeSinceSchoolMax
                        s = 0.0
                        for row in 1:p.MonteCarloCount
                            ε5 = epssolve[5,row]
                            VF5 = util5GPU(p, educ, ε5)
                            VF5 = VF5 + p.δ * EmaxNext5

                            s += max(VF5)
                        end
                        value = s/p.MonteCarloCount
                    else
                        s = 0.0
                        for row in 1:p.MonteCarloCount
                            ε1 = epssolve[1,row]
                            VF1 = util1GPU(p, age, educ, LastChoice, ε1; type=type)
                            VF1 = VF1 + p.δ * EmaxNext1
                            ε5 = epssolve[5,row]
                            VF5 = util5GPU(p, educ, ε5)
                            VF5 = VF5 + p.δ * EmaxNext5

                            s += max(VF1, VF5)
                        end
                        value = s/p.MonteCarloCount
                    end
                else
                    if     homeSinceSchool<p.homeSinceSchoolMax
                        s = 0.0
                        for row in 1:p.MonteCarloCount
                            ε1 = epssolve[1,row]
                            VF1 = util1GPU(p, age, educ, LastChoice, ε1; type=type)
                            VF1 = VF1 + p.δ * EmaxNext1
                            ε5 = epssolve[5,row]
                            VF5 = util5GPU(p, educ, ε5)
                            VF5 = VF5 + p.δ * EmaxNext5
                            ε2 = epssolve[2,row]
                            VF2 = util2GPU(p, LastChoice, educ+1, ε2, age; type=type)
                            VF2 = VF2 + p.δ * EmaxNext2
                            s += max(VF1, VF2, VF5)
                        end
                        value = s/p.MonteCarloCount
                    else
                        s = 0.0
                        for row in 1:p.MonteCarloCount
                            ε5 = epssolve[5,row]
                            VF5 = util5GPU(p, educ, ε5)
                            VF5 = VF5 + p.δ * EmaxNext5
                            ε2 = epssolve[2,row]
                            VF2 = util2GPU(p, LastChoice, educ+1, ε2, age; type=type)
                            VF2 = VF2 + p.δ * EmaxNext2
                            s += max(VF5, VF2)
                        end
                        value = s/p.MonteCarloCount
                    end
                end
            end

        elseif age <= 18
            s = 0.0
            for row in 1:p.MonteCarloCount
                ε1 = epssolve[1,row]
                VF1 = util1GPU(p, age, educ, LastChoice, ε1; type=type)
                VF1 = VF1 + p.δ * EmaxNext1
                ε2 = epssolve[2,row]
                VF2 = util2GPU(p, LastChoice, educ+1, ε2, age; type=type)
                VF2 = VF2 + p.δ * EmaxNext2
                ε3 = epssolve[3,row]
                VF3 = util3GPU(p, x3, x4, LastChoice, educ, ε3; type=type)
                VF3 = VF3 + p.δ * EmaxNext3
                ε4 = epssolve[4,row]
                VF4 = util4GPU(p, x3, x4, LastChoice, educ, ε4; type=type)
                VF4 = VF4 + p.δ * EmaxNext4


                s += max(VF1, VF2, VF3, VF4)
            end
            value = s/p.MonteCarloCount
        end

    end

    Emax[EmaxIndex] = value
    return nothing
end




#= Solve Emax for conscription goup 1: obligated to attent conscription =#
function solveGroup1AllType(p::NamedTuple, epssolve)

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
    # State space size= 49*23*2*31*31*3=         6,498,282
    =#

    ageState  = 65 :-1 :17      # age age of the individual
    educState = 0 :1 :22        # educ number of completed education
    LastChoiceState   = 1:5     # LastChoice : alternative chosen in the last period
    x3State   = 0 :1 : 30#p.x3Max    # x3 experience in white-collar
    x4State   = 0 :1 : 30#p.x4Max    # x4 experience in blue-collar
    x5State   = [0,1,2]         # x5 indicate the years attending conscription
    homeSinceSchoolState = 0 :1 : p.homeSinceSchoolMax

    ageStateCount        = size(ageState)[1]
    educStateCount       = size(educState)[1]
    LastChoiceStateCount = size(LastChoiceState)[1]
    x3StateCount         = size(x3State)[1]
    x4StateCount         = size(x4State)[1]
    x5StateCount         = size(x5State)[1]
    typeCount            = 3
    homeSinceSchoolCount = size(homeSinceSchoolState)[1]

    stateSpaceSize = ageStateCount*educStateCount*LastChoiceStateCount*x3StateCount*x4StateCount*x5StateCount*typeCount*homeSinceSchoolCount

    Emax = CUDA.fill(1.0, (stateSpaceSize, 1))
    epssolve = CuArray(epssolve)


    numblocks = ceil(Int, educStateCount*LastChoiceStateCount*x3StateCount*x4StateCount*x5StateCount*typeCount*homeSinceSchoolCount/ 256)



    for age in ageState

        @cuda threads=256 blocks=numblocks valueFunctionGroup1!(p,
                                                            epssolve,
                                                            age,
                                                            Emax)
        # synchronize()
    end#age

    return Array(Emax)

end#



# p = (
#     ω1 = (
#         6.494981446570038e7,
#         4.071547679686325e7,
#         3.3510846922447067e7,
#         2.2812544086896487e7,
#     ),
#     α11 = -3.6198218054661066e6,
#     α12 = 1.6350327793562492e7,
#     α13 = 578137.0016164911,
#     ω2 = (
#         4.682169461357033e7,
#         3.454377812455817e7,
#         1.0050479426923603e8,
#         1.7620503972486955e8,
#     ),
#     α21 = 4.970335384466663e7,
#     tc1 = 6.639993948943794e7,
#     tc2 = 5.947449454495121e7,
#     α22 = 0.004063915354641,
#     α23 = 0.120589170776119,
#     α24 = 0.004063915354641,
#     α25 = 0.195642963405802,
#     α30study = -1.75184913592058e7,
#     α3 = 8.987225720448703e6,
#     ω3 = (
#         15.3190224863048,
#         14.6600146092114,
#         14.7572471679754,
#         15.4257130787502,
#     ),
#     α31 = 0.131968749103811,
#     α32 = 0.069661681577344,
#     α33 = 0.00807438755079,
#     α34 = -0.001717715506452,
#     α35 = -0.000676942380882,
#     α36 = -0.096846452570373,
#     α37 = 0.177014442769366,
#     α38 = -0.030979085160465,
#     α4 = 0.0,
#     ω4 = (
#         17.1357273111579,
#         16.6192133612883,
#         16.4725358085427,
#         17.1452068733274,
#     ),
#     α41 = 0.033686638581205,
#     α42 = 0.013498871415515,
#     α43 = 0.089851927810506,
#     α44 = -0.004040294642764,
#     α45 = -0.001894047990191,
#     α46 = 0.353411632358943,
#     α47 = 0.12684934370643,
#     α48 = -0.026040022906069,
#     α50 = 2.721455614761497e6,
#     α51 = 1.1251069905457921e7,
#     α52 = 1.1801725341969458e7,
#     σ1 = 5.1510607965574506e14,
#     σ2 = 5.272558880654142e13,
#     σ3 = 0.427370676099242,
#     σ4 = 0.200546301781832,
#     σ34 = -0.244851477711262,
#     σ5 = 6.669157753534134e14,
#     δ = 0.92,
#     x3Max = 30,
#     x4Max = 30,
#     MonteCarloCount = 200,
#     homeSinceSchoolMax = 4,
# )
#
#
# #= test section =#
# #= here we check whether Emax function is working perfect or not. =#
# include("/home/sabouri/Dropbox/Labor/Codes/GitRepository/modelParameters.jl")
# epsSolveMeanGroup1= [0.0, 0.0, 0.0, 0.0, 0.0] ;
# epsSolveσGroup1=[σ1   0.0  0.0  0.0  0.0 ;
#                 0.0  σ2   0.0  0.0  0.0 ;
#                 0.0  0.0  σ3   σ34  0.0 ;
#                 0.0  0.0  σ34  σ4   0.0
#                 0.0  0.0  0.0  0.0  σ5  ] ;
# M=200;
# epssolveGroup1= rand(MersenneTwister(1234),MvNormal(epsSolveMeanGroup1, epsSolveσGroup1) , M) ;
#
#
# for i in 1:1
#     print("Emax Group 1 calculation: \n")
#     start = Dates.unix2datetime(time())
#
#     EmaxGroup1= solveGroup1AllType(p, epssolveGroup1);
#
#     finish = convert(Int, Dates.value(Dates.unix2datetime(time())- start))/1000;
#     print("TOTAL ELAPSED TIME: ", finish, " seconds. \n",EmaxGroup1[1],"   ",EmaxGroup1[2])
# end




# ################################################################################
# function solveAllGroupAllType(p::NamedTuple, epssolveGroup1, epssolveGroup2)
#
#
#     ageState  = 65 :-1 :17      # age age of the individual
#     educState = 0 :1 :22        # educ number of completed education
#     LastChoiceState   = 1:5     # LastChoice : alternative chosen in the last period
#     x3State   = 0 :1 : p.x3Max    # x3 experience in white-collar
#     x4State   = 0 :1 : p.x4Max    # x4 experience in blue-collar
#     x5State   = [0,1,2]         # x5 indicate the years attending conscription
#
#
#     ageStateCount        = size(ageState)[1]
#     educStateCount       = size(educState)[1]
#     LastChoiceStateCount = size(LastChoiceState)[1]
#     x3StateCount         = size(x3State)[1]
#     x4StateCount         = size(x4State)[1]
#     x5StateCount         = size(x5State)[1]
#     typeCount            = 4
#
#
#     stateSpaceSizeGroup1 = ageStateCount* educStateCount* LastChoiceStateCount* x3StateCount* x4StateCount* x5StateCount * typeCount
#     EmaxGroup1 = CUDA.fill(1.0, (stateSpaceSizeGroup1, 1))
#     epssolveGroup1 = CuArray(epssolveGroup1)
#     numblocksGroup1 = ceil(Int, educStateCount*LastChoiceStateCount*x3StateCount*x4StateCount*x5StateCount*typeCount/256)
#
#
#
#     stateSpaceSizeGroup2 = ageStateCount* educStateCount* LastChoiceStateCount* x3StateCount* x4StateCount * typeCount
#     EmaxGroup2 = CUDA.fill(1.0, (stateSpaceSizeGroup2, 1))
#     epssolveGroup2 = CuArray(epssolveGroup2)
#     numblocksGroup2 = ceil(Int, educStateCount*LastChoiceStateCount*x3StateCount*x4StateCount*typeCount/256)
#
#
#
#     for age in ageState
#
#         @cuda threads=256 blocks=numblocksGroup1 valueFunctionGroup1!(p,
#                                                             epssolveGroup1,
#                                                             age,
#                                                             EmaxGroup1)
#
#
#         @cuda threads=256 blocks=numblocksGroup2 valueFunctionGroup2!(p,
#                                                             epssolveGroup2,
#                                                             age,
#                                                             EmaxGroup2)
#
#         synchronize()
#     end#age
#
#
#     return Array(EmaxGroup1), Array(EmaxGroup2)
#
# end#












################################################################################
#= simulate conscription goup 1 =#

function simulateGroup2(p::NamedTuple, N, Emax, weights; Seed=1234, type=1)


    #= Pre-allocating each person-year's state=#
    sim = Array{Float64, 2}(undef, (N*50, length(p.simCol)))
    sim[:,p.simCol["x5"]] .= NaN

    #= education distribution in age 16 of people: =#
    educLevel = [0    ,5    ,8    ,10  ]
    # weights =   [0.02 ,0.20 ,0.24 ,0.54]
    #= drawing educ level exogenously form this distribution =#
    a = sample(MersenneTwister(Seed),educLevel, Weights(weights), N)


    epsSolveMean= [0.0, 0.0, 0.0, 0.0]
    epsSolveσ=[p.σ1  0.0    0.0    0.0  ;
               0.0   p.σ2   0.0    0.0  ;
               0.0   0.0   p.σ3   p.σ34 ;
               0.0   0.0   p.σ34  p.σ4  ]
    epsestimation = rand(MersenneTwister(Seed), MvNormal(epsSolveMean, epsSolveσ), 50 * N)

    for id in 1:N

        for age in 16:65

            index= 50*(id-1)+ age-15
            sim[index, p.simCol["age"]]= age

            if age==16
                x3         = 0
                x4         = 0
                educ       = a[id]
                LastChoice = 2
            else
                x3   = convert(Int,sim[index-1,p.simCol["x3"]])
                x4   = convert(Int,sim[index-1,p.simCol["x4"]])
                educ = convert(Int,sim[index-1,p.simCol["educ"]])
                LastChoice  = convert(Int,sim[index-1,p.simCol["choice"]])
            end

            #= four shocks to person i in age 'age': =#
            ε1,ε2,ε3,ε4= epsestimation[ : , index]

            #= comtemporaneous utility from each decision : =#
            u1= util1GPU(p, age, educ, LastChoice, ε1; type=type)
            u2= util2GPU(p, LastChoice, educ+1, ε2, age; type=type)
            u3= util3GPU(p, x3, x4, LastChoice, educ, ε3; type=type)
            u4= util4GPU(p, x3, x4, LastChoice, educ, ε4; type=type)

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

                enumerator = EmaxGroup2Index(age+1, educ, 1, x3, x4, type)
                u1= u1 +p.δ*Emax[enumerator]

                enumerator = EmaxGroup2Index(age+1, (educ+1*(educ< 22)), 2, x3, x4, type)
                u2= u2 +p.δ*Emax[enumerator]

                enumerator = EmaxGroup2Index(age+1, educ, 3, (x3+1*(x3< p.x3Max)), x4, type)
                u3= u3 +p.δ*Emax[enumerator]

                enumerator = EmaxGroup2Index(age+1, educ, 4, x3, (x4+1*(x4< p.x4Max)), type)
                u4= u4 +p.δ*Emax[enumerator]

                if educ < 22
                    utility= [u1, u2, u3, u4]
                elseif educ==22
                    utility= [u1, -1e20, u3, u4]
                end
                choice= argmax(utility)
                maxUtility = maximum(utility)
            end

            #= writing 'choice' in results =#
            sim[index, p.simCol["choice"]] = choice
            if age>16
                sim[index-1, p.simCol["choice_next"]] = choice
            end

            #= specifying subsequent period state based on 'choice' of this period =#
            if     choice==1
                sim[index, p.simCol["income"]]= NaN
                sim[index, p.simCol["x3"]]    = x3
                sim[index, p.simCol["x4"]]    = x4
                sim[index, p.simCol["educ"]]  = educ
            elseif choice==2
                sim[index, p.simCol["income"]]= NaN
                sim[index, p.simCol["x3"]]    = x3
                sim[index, p.simCol["x4"]]    = x4
                sim[index, p.simCol["educ"]]  = educ+ 1
            elseif choice==3
                sim[index, p.simCol["income"]]= wageWhiteCollar(p, educ, x3, x4, LastChoice, ε3; type=type)
                sim[index, p.simCol["x3"]]  = x3 +1*(x3<p.x3Max)
                sim[index, p.simCol["x4"]]  = x4
                sim[index, p.simCol["educ"]]= educ

            elseif choice==4
                sim[index, p.simCol["income"]]= wageBlueCollar(p, educ, x3, x4, LastChoice, ε4; type=type)
                sim[index, p.simCol["x3"]]  = x3
                sim[index, p.simCol["x4"]]  = x4 +1*(x4<p.x4Max)
                sim[index, p.simCol["educ"]]= educ
            end
            sim[index, p.simCol["Emax"]] = maxUtility #utility[choice]

            #=
            specifying if persion is educated or not (educ > 12 or not)
            this helps in calculating moment conditions from simulated data
            =#
            if choice == 2
                educ = educ + 1
            end
            if age < 22
                sim[index, p.simCol["educated"]] = -1
            else
                if educ > 12
                    sim[index, p.simCol["educated"]] = 1
                elseif educ <= 12
                    sim[index, p.simCol["educated"]] = 0
                end
            end

        end

    end

    return sim
end#simulate-1e20


################################################################################
#= simulate conscription goup 2 =#

function simulateGroup1(p::NamedTuple, N, Emax, weights; Seed=1234, type=1)

    #= Pre-allocating each person-year's state =#
    sim = Array{Float64, 2}(undef, (N*50, length(p.simCol)))

    #= education distribution in age 16 of people: =#
    educLevel = [0    ,5    ,8    ,10  ]
    # weights =   [0.02 ,0.20 ,0.24 ,0.54]
    # drawing educ level exogenously form this distribution
    a=sample(MersenneTwister(Seed),educLevel, Weights(weights), N)


    epsSolveMean= [0.0, 0.0, 0.0, 0.0, 0.0]
    epsSolveσ=[p.σ1   0.0    0.0    0.0    0.0  ;
               0.0    p.σ2   0.0    0.0    0.0  ;
               0.0    0.0    p.σ3   p.σ34  0.0  ;
               0.0    0.0    p.σ34  p.σ4   0.0  ;
               0.0    0.0    0.0    0.0    p.σ5 ]
    epsestimation=rand(MersenneTwister(Seed),MvNormal(epsSolveMean, epsSolveσ) , 50*N)

    for id in 1:N

        for age in 16:65

            index= 50*(id-1)+ age-15
            sim[index, p.simCol["age"]]= age

            if age==16
                x3           = 0
                x4           = 0
                educ         = a[id]
                LastChoice   = 2
                x5           = 0
                homeSinceSchool = 0
            else
                x3   = convert(Int,sim[index-1,p.simCol["x3"]])
                x4   = convert(Int,sim[index-1,p.simCol["x4"]])
                educ = convert(Int,sim[index-1,p.simCol["educ"]])
                LastChoice  = convert(Int,sim[index-1,p.simCol["choice"]])
                x5   = convert(Int, sim[index-1,p.simCol["x5"]])
                homeSinceSchool = convert(Int, sim[index-1,p.simCol["homeSinceSchool"]])
            end

            #= four shocks to person i in age 'age': =#
            ε1, ε2, ε3, ε4, ε5 = epsestimation[:, index]

            #= comtemporaneous utility from each decision : =#
            u1= util1GPU(p, age, educ, LastChoice, ε1; type=type)
            u2= util2GPU(p, LastChoice, educ+1, ε2, age; type=type)
            u3= util3GPU(p, x3, x4, LastChoice, educ, ε3; type=type)
            u4= util4GPU(p, x3, x4, LastChoice, educ, ε4; type=type)
            u5= util5GPU(p, educ, ε5)

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


                enumerator = EmaxGroup1Index(age+1, educ, 1, x3, x4, x5, type, homeSinceSchool+ 1*(age>=19)*(homeSinceSchool<p.homeSinceSchoolMax))
                u1= u1 +p.δ*Emax[enumerator]

                enumerator = EmaxGroup1Index(age+1, (educ+1*(educ< 22)), 2, x3, x4, x5, type, homeSinceSchool)
                u2= u2 +p.δ*Emax[enumerator]

                enumerator = EmaxGroup1Index(age+1, educ, 3, (x3+1*(x3< p.x3Max)), x4, x5, type, homeSinceSchool)
                u3= u3 +p.δ*Emax[enumerator]

                enumerator = EmaxGroup1Index(age+1, educ, 4, x3, (x4+1*(x4< p.x4Max)), x5, type, homeSinceSchool)
                u4= u4 +p.δ*Emax[enumerator]

                enumerator = EmaxGroup1Index(age+1, educ, 5, x3, x4, x5+1*(x5!=2), type, homeSinceSchool)
                u5= u5 +p.δ*Emax[enumerator]


                if age > 18
                    if x5 == 2
                        if educ < 22
                            utility= [u1, u2, u3, u4, -1e20]
                        else
                            utility= [u1, -1e20, u3, u4, -1e20]
                        end#if educ
                    # elseif x5 == 1
                    #         utility= [-1e20, -1e20, -1e20, -1e20, u5]
                    else #if x5 == 0
                        if educ == 22
                            if homeSinceSchool<p.homeSinceSchoolMax
                                utility= [u1, -1e20, -1e20, -1e20, u5]
                            else
                                utility= [-1e20, -1e20, -1e20, -1e20, u5]
                            end
                        else
                            if homeSinceSchool<p.homeSinceSchoolMax
                                utility= [u1, u2, -1e20, -1e20, u5]
                            else
                                utility= [-1e20, u2, -1e20, -1e20, u5]
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
            sim[index, p.simCol["choice"]] = choice
            if age>16
                sim[index-1, p.simCol["choice_next"]] = choice
            end

            #= specifying subsequent period state based on 'choice' of this period =#
            if     choice==1
                sim[index, p.simCol["income"]]= NaN
                sim[index, p.simCol["x3"]]    = x3
                sim[index, p.simCol["x4"]]    = x4
                sim[index, p.simCol["educ"]]  = educ
                sim[index, p.simCol["x5"]]    = x5
                sim[index, p.simCol["homeSinceSchool"]] = homeSinceSchool+ 1*(age>=19)*(homeSinceSchool<p.homeSinceSchoolMax)
            elseif choice==2
                sim[index, p.simCol["income"]]= NaN
                sim[index, p.simCol["x3"]]    = x3
                sim[index, p.simCol["x4"]]    = x4
                sim[index, p.simCol["educ"]]  = educ+ 1
                sim[index, p.simCol["x5"]]    = x5
                sim[index, p.simCol["homeSinceSchool"]] = homeSinceSchool
            elseif choice==3
                sim[index, p.simCol["income"]]= wageWhiteCollar(p, educ, x3, x4, LastChoice, ε3; type=type)
                sim[index, p.simCol["x3"]]  = x3 +1*(x3<p.x3Max)
                sim[index, p.simCol["x4"]]  = x4
                sim[index, p.simCol["educ"]]= educ
                sim[index, p.simCol["x5"]]    = x5
                sim[index, p.simCol["homeSinceSchool"]] = homeSinceSchool

            elseif choice==4
                sim[index, p.simCol["income"]]= wageBlueCollar(p, educ, x3, x4, LastChoice, ε4; type=type)
                sim[index, p.simCol["x3"]]  = x3
                sim[index, p.simCol["x4"]]  = x4 +1*(x4<p.x4Max)
                sim[index, p.simCol["educ"]]= educ
                sim[index, p.simCol["x5"]]    = x5
                sim[index, p.simCol["homeSinceSchool"]] = homeSinceSchool

            elseif choice==5
                sim[index, p.simCol["income"]]= 0.08 * mean([wageWhiteCollar(p, educ, x3, x4, LastChoice, ε3; type=type), wageBlueCollar(p, educ, x3, x4, LastChoice, ε4; type=type)])
                sim[index, p.simCol["x3"]]    = x3
                sim[index, p.simCol["x4"]]    = x4
                sim[index, p.simCol["educ"]]  = educ
                sim[index, p.simCol["x5"]]    = x5 + 1
                sim[index, p.simCol["homeSinceSchool"]] = homeSinceSchool

            end
            sim[index, p.simCol["Emax"]] = maxUtility #utility[choice]

            #=
             specifying if persion is educated or not (educ > 12 or not)
             this helps in calculating moment conditions from simulated data
            =#
            if choice == 2
                educ = educ + 1
            end
            if age < 22
                sim[index, p.simCol["educated"]] = -1
            else
                if educ>13
                    sim[index, p.simCol["educated"]] = 1
                elseif educ<=13
                    sim[index, p.simCol["educated"]] = 0
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
function SMMCalculate(choiceMoment, wageMoment, educatedShare, transMoment,
    wageCol, choiceCol, educatedCol, transCol,
    contributions)


    wageWhiteError  = 0.0
    wageBlueError   = 0.0
    devWhiteError   = 0.0
    devBlueError    = 0.0
    homeError       = 0.0
    studyError      = 0.0
    whiteError      = 0.0
    blueError       = 0.0
    milError        = 0.0
    educatedError   = 0.0
    transError      = 0.0


    for i in 1:size(wageMoment, 1)

        # percentage error of mean income moment
        error = (
            (
                wageMoment[i, wageCol["incomeData"]] -wageMoment[i, wageCol["incomeSim"]]
            ) / wageMoment[i, wageCol["incomeStdBoot"]]
        )

        #=
        If error is NaN, it means no one is working in that occupation, thus we
        make this error bigger to force the optimization algorithm to avoid
        from this area of paramater's domain.
        =#

#         if isinf(error)|isnan(error)
#             print("wageWhiteError or wageBlueError error at age = ", wageMoment[i, wageCol["age"]] ,"\n")
#         end
        if isinf(error)|isnan(error)
            error = 10.0
        end

        if wageMoment[i,wageCol["collar"]] == 0.0
            wageWhiteError += error^2
        end

        if wageMoment[i,wageCol["collar"]] == 1.0
            wageBlueError += error^2
        end

        contributions = [contributions; error]

        #= percentage error of mean income standard deviation =#
        error = (
            (
                wageMoment[i, wageCol["devData"]] -wageMoment[i, wageCol["devSim"]]
            ) / wageMoment[i, wageCol["devStdBoot"]]
        )

#         if isinf(error)|isnan(error)
#             print("devWhiteError or devBlueError age = ", wageMoment[i, wageCol["age"]] ,"\n")
#         end

        if isinf(error)|isnan(error)
            error = 10.0
        end

        if wageMoment[i,wageCol["collar"]] == 0.0
            devWhiteError += error^2
        end

        if wageMoment[i,wageCol["collar"]] == 1.0
            devBlueError += error^2
        end

        contributions = [contributions; error]

    end



    for i in 1:size(choiceMoment,1)

        #= choice 1: home production =#
        error = (
            (
                choiceMoment[i, choiceCol["homeData"]] -
                choiceMoment[i, choiceCol["homeSim" ]]
            ) / choiceMoment[i, choiceCol["homeStdBoot"]]
        )

#         if isinf(error)
#             print("homeError age = ", wageMoment[i, wageCol["age"]] ,"\n")
#         end

        if isinf(error)|isnan(error)
            error =  0.0
        end


        contributions = [contributions; error]


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

#         if isinf(error)
#             print("studyError age = ", wageMoment[i, wageCol["age"]] ,"\n")
#         end
        if isinf(error)|isnan(error)
            error =  0.0
        end

        studyError += error^2

        contributions = [contributions; error]

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

#         if isinf(error)
#             print("whiteError age = ", wageMoment[i, wageCol["age"]] ,"\n")
#         end
        if isinf(error)|isnan(error)
            error =  0.0
        end

        contributions = [contributions; error]

        whiteError += error^2

        #= choice 4: blue-collar occupation =#
        error = (
            (
                choiceMoment[i, choiceCol["blueData"]] -
                choiceMoment[i, choiceCol["blueSim" ]]
            ) / choiceMoment[i, choiceCol["blueStdBoot"]]
        )
#         if isinf(error)
#             print("blueError age = ", wageMoment[i, wageCol["age"]] ,"\n")
#         end
        if isinf(error)|isnan(error)
            error =  0.0
        end

        contributions = [contributions; error]

        blueError += error^2

        #= choice 5: compulsory military service =#
        error = (
            (
                choiceMoment[i, choiceCol["milData"]] -
                choiceMoment[i, choiceCol["milSim" ]]
            ) / choiceMoment[i, choiceCol["milStdBoot"]]
        )
#         if isinf(error)
#             print("milError age = ", wageMoment[i, wageCol["age"]] ,"\n")
#         end
        if isinf(error)|isnan(error)
            error =  0.0
        end

        if (choiceMoment[i, choiceCol["age"]] > 18) & (choiceMoment[i, choiceCol["age"]] <= 32)
            milError += error^2
            contributions = [contributions; error]
        end


    end


    #= loop to calculated the SMM error of
    the educated share moments =#

    for i in 1:size(educatedShare,1)

        error = (
            (
                educatedShare[i, educatedCol["educatedData"]] -
                educatedShare[i, educatedCol["educatedSim" ]]
            ) / educatedShare[i, educatedCol["educatedStdBoot"]]
        )
#         if isinf(error)|isnan(error)
#             print("educatedError age = ", wageMoment[i, wageCol["age"]] ,"\n")
#         end
        if isnan(error)|isnan(error)
            error =  0.0
        end
        contributions = [contributions; error]
        educatedError = educatedError + error^2

    end


    # for i in 1:size(transMoment, 1)
    #
    #     # percentage error of mean income moment
    #     error = (
    #         (
    #             transMoment[i, transCol["transData"]] -transMoment[i, transCol["transSim"]]
    #         ) / transMoment[i, transCol["transStdBoot"]]
    #     )
    #
    #     if isinf(error) | isnan(error)
    #         error = 0.0
    #     end
    #
    #     contributions = [contributions; error]
    #     transError = transError + error^2
    #
    # end


#     #= Printing each error seperately =#
#     print("\n wageWhiteError  = ", wageWhiteError )
#     print("\n wageBlueError   = ", wageBlueError  )
#     print("\n homeError       = ", homeError      )
#     print("\n studyError      = ", studyError     )
#     print("\n whiteError      = ", whiteError     )
#     print("\n blueError       = ", blueError      )
#     print("\n milError        = ", milError       )
#     print("\n devWhiteError   = ", devWhiteError  )
#     print("\n devBlueError    = ", devBlueError   )
#     print("\n educatedError   = ", educatedError  )


    #=
    Shift the error term when estimation is going to areas of parameters
    where no one employ in white-collar or blue-occupation
    =#

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
        devBlueError +
        educatedError #+
        # transError
    )
    print("\n error without constraint   = ", round(SMMError)  )

    return SMMError, contributions
end



#= ************************************************************* =#
function constraintError(sim, simCol, contributions)

    whiteConstraintError = 0.0
    studyConstraintError = 0.0
    blueConstraintError  = 0.0
    homeConstraintError  = 0.0
    whiteWageConstraintError = 0.0
    blueWageConstraintError = 0.0

    for age in 50:63

        flag2 = [
            count(
                x -> x == i,
                sim[(sim[:, simCol["age"]].==age)
                , simCol["choice"]],
                ) for i = 1:5
        ]

        flag2 = flag2 / sum(flag2)

        error = ((flag2[3]-0.165)/0.001)
        if isinf(error)|isnan(error)
            error = 10
#             print("10\n")
        end
        contributions = [contributions; error]
        whiteConstraintError = whiteConstraintError + error^2

        error = (flag2[2]-0)/0.003
        if isinf(error)|isnan(error)
            error = 10
#             print("11\n")
        end
        contributions = [contributions; error]
        studyConstraintError = studyConstraintError + error^2



        error = (flag2[4]-0.73)/0.002
        if isinf(error)|isnan(error)
            error = 10
#             print("11\n")
        end
        contributions = [contributions; error]
        blueConstraintError = blueConstraintError + error^2

    end


    for age in 40:55

        # flagChoice = [
        #     count(
        #         x -> x == i,
        #         sim[(sim[:, simCol["age"]].==age)
        #         , simCol["choice"]],
        #         ) for i = 1:5
        # ]
        # flagChoice = flagChoice / sum(flagChoice)

        flag = sim[ (sim[:,simCol["age"]].==age).&
                    (sim[:,simCol["choice"]].== 3), simCol["income"]]

        LogWageMean = 19.2*(age<=40) + 19.3*(age>40)*(age<=45) + 19.4*(age>45)*(age<=50) + 19.4*(age>50)*(age<=55)
        error = (log(mean(filter(!isnan, flag))) - LogWageMean) / 0.02

        if (isinf(error)|isnan(error)|(error==Inf)|(error==NaN))
            error = 40
        end
        contributions = [contributions; error]
        whiteWageConstraintError = whiteWageConstraintError + error^2
        # end

        flag = sim[ (sim[:,simCol["age"]].==age).&
                    (sim[:,simCol["choice"]].== 4), simCol["income"]]

        LogWageMean = 18.5*(age<=40) + 18.5*(age>40)*(age<=45) + 18.6*(age>45)*(age<=50) + 18.5*(age>50)*(age<=55)
        error = (log(mean(filter(!isnan, flag))) - LogWageMean) / 0.02
        if (isinf(error)|isnan(error)|(error==Inf)|(error==NaN))
            error = 30
        end
        contributions = [contributions; error]
        blueWageConstraintError = blueWageConstraintError + error^2
        # end

    end




    output = whiteConstraintError + studyConstraintError + blueConstraintError + homeConstraintError + whiteWageConstraintError + blueWageConstraintError
    return output, contributions
end




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
    δ = 0.92 #0.7937395498108646 ;

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
    writedlm( "data/simulation/simNew2.csv",  sim, ',');
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





################################################################################
#= Initiating the best result on the disk with a large number =#


#= read data moment files =#

# include("/home/sabouri/Dropbox/Labor/Codes/GitRepository/ThreeTypes/modelParameters.jl")
# include("/content/drive/MyDrive/Projects/Labor/Codes/GitRepository/ThreeTypes/modelParameters.jl")



# contributions = 1.0e50
# writedlm("/home/sabouri/Labor/CodeOutput/contributionsBest.csv", contributions , ',') ;

# result = 1.0e50
# writedlm("/home/sabouri/Labor/CodeOutput/result.csv", result)

# print("\nEstimation started:")
# start = Dates.unix2datetime(time())

result = estimation(Params, choiceMomentStdBoot, wageMomentStdBoot, educatedShareStdBoot, transMomentStdBoot) ;

# finish = convert(Int, Dates.value(Dates.unix2datetime(time())- start))/1000 ;
# print("\nTtotal Elapsed Time: ", finish, " seconds. \n")

# end