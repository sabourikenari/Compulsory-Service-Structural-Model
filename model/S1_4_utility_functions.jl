
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
