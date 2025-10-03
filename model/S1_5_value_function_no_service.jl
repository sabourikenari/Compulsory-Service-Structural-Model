
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


