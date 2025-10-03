

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