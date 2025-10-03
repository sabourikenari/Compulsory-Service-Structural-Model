
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
