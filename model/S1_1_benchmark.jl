
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


# include the required libraries 
include("S1_2_libraries.jl")

# include the model parameters
include("S1_3_model_parameters.jl")

# include data moments needed for the estimation
include("S1_4_moments.jl")

# define contemporaneous utility functions
include("S1_4_utility_functions.jl")

# solve value function iteration
include("S1_5_value_function_no_service.jl") 
include("S1_6_value_function_military_service.jl")

# simulate the model
include("S1_7_simulation_no_service.jl")
include("S1_8_simulation_military_service.jl")

# calculate the estimation criterion
include("S1_9_SMM.jl")

# estimation function
include("S1_10_wrapper.jl")






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