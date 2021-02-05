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

#=
    The codes needed for running on the server
    and also extracting the simulation results from Server
    to my ubuntu operating system.
=#



################################################################################
#= Define estimation Function =#

function estimation_pounders(input,
    choiceMomentData, wageMomentData, educatedShareData)


    #=****************************************************=#
    #= parameters =#

    ω1T1, ω1T2, ω1T3, ω1T4, α11, α12, α13,
    ω2T1, ω2T2, ω2T3, ω2T4,
    α21, tc1T1, tc2, α22, α23, α25, α30study,
    α3, ω3T1, ω3T2, ω3T3, ω3T4, α31, α32, α33, α34, α35,
        ω4T1, ω4T2, ω4T3, ω4T4, α41, α42, α43, α44, α45,
    α50, α51, α52,
    σ1, σ2, σ3, σ4, σ34 ,σ5,
    πE1T1exp, πE1T2exp, πE1T3exp,
    πE2T1exp, πE2T2exp, πE2T3exp,
    π1T1exp, π1T2exp, π1T3exp, π1T4exp                 = convert.(Float64, input.value)


    input_estimation=[ω1T1, ω1T2, ω1T3, ω1T4, α11, α12, α13 ,
            ω2T1, ω2T2, ω2T3, ω2T4,
            α21, tc1T1, tc2, α22, α23, α25, α30study,
            α3, ω3T1, ω3T2, ω3T3, ω3T4, α31, α32, α33, α34, α35,
                ω4T1, ω4T2, ω4T3, ω4T4, α41, α42, α43, α44, α45,
            α50, α51, α52,
            σ1, σ2, σ3, σ4, σ34 ,σ5,
            πE1T1exp, πE1T2exp, πE1T3exp,
            πE2T1exp, πE2T2exp, πE2T3exp,
            π1T1exp, π1T2exp, π1T3exp, π1T4exp  ]

    out = estimation(input_estimation,
        choiceMomentData, wageMomentData, educatedShareData)

    return out
end
