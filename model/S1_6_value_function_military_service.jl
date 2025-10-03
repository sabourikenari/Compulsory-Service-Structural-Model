


################################################################################
#=
conscription goup 2 value function and solve Emax function
conscription goup 2: obligated to attend conscription
=#

function flatten_index(sizes::NTuple{N,Int}, coords::NTuple{N,Int}) where N
    idx = 1
    stride = 1
    @inbounds for i in 1:N
        idx += (coords[i] - 1) * stride
        stride *= sizes[i]
    end
    return idx
end

function EmaxGroup1Index(age, educ, LastChoice, x3, x4, x5, type, homeSinceSchool)
    sizes  = (3, 31, 31, 5, 23, 49, 3, 5)  # x5, x4, x3, LastChoice, educ, age, type, homeSinceSchool
    coords = (x5+1, x4+1, x3+1, LastChoice, educ+1, age-17+1, type, homeSinceSchool+1)
    return flatten_index(sizes, coords)
end

# function EmaxGroup1Index(age, educ, LastChoice, x3, x4, x5, type, homeSinceSchool)

#     typeCount              = 3
#     ageStateCount          = 49
#     educStateCount         = 23
#     LastChoiceStateCount   = 5
#     x3StateCount           = 31
#     x4StateCount           = 31
#     x5StateCount           = 3
#     # homeSinceSchoolCount   = 2 #4 + 1

#     enumerator = (
#         (x5+1) +
#         (x4)             * x5StateCount +
#         (x3)             * x5StateCount* x4StateCount +
#         (LastChoice-1)   * x5StateCount* x4StateCount* x3StateCount +
#         (educ)           * x5StateCount* x4StateCount* x3StateCount* LastChoiceStateCount +
#         (age-17)         * x5StateCount* x4StateCount* x3StateCount* LastChoiceStateCount* educStateCount +
#         (type-1)         * x5StateCount* x4StateCount* x3StateCount* LastChoiceStateCount* educStateCount* ageStateCount +
#         (homeSinceSchool)* x5StateCount* x4StateCount* x3StateCount* LastChoiceStateCount* educStateCount* ageStateCount* typeCount
#     )
#     return enumerator
# end





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
        return nothing
    end


    #***********************************#

    value= -1 # this is for sanity check for a time when none of the if conditons binds
    # ---------------- Terminal age ---------------- #
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

    # ---------------- Recursive ages ---------------- #
    else

        # Next-state indices for each *choice taken this period*
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
                        
            # if     x5 == 2
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
                valueCompleted = s/p.MonteCarloCount
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
                valueCompleted = s/p.MonteCarloCount
            end
            
            
            # elseif x5 == 0
            if educ == 22
                if homeSinceSchool==p.homeSinceSchoolMax
                    s = 0.0
                    for row in 1:p.MonteCarloCount
                        ε5 = epssolve[5,row]
                        VF5 = util5GPU(p, educ, ε5)
                        VF5 = VF5 + p.δ * EmaxNext5

                        s += max(VF5)
                    end
                    valueNotCompleted = s/p.MonteCarloCount
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
                    valueNotCompleted = s/p.MonteCarloCount
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
                    valueNotCompleted = s/p.MonteCarloCount
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
                    valueNotCompleted = s/p.MonteCarloCount
                end
            end
        
        
            if x5==2
                value = valueCompleted
            
            elseif x5==0 
                value = valueNotCompleted
            
            elseif x5==1
                # probability_get_one_year = 0  
                value = p.probability_get_one_year * valueCompleted + (1 - p.probability_get_one_year) *  valueNotCompleted
            
            end

        

        # age <= 18 → normal set without Military
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


