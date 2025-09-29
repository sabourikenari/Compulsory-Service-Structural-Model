* ----------------------------------------------------------------------
* ----------------------------------------------------------------------
* conscription and lifetime earnings

* change working directory
cd "C:\Users\ehsa7798\GoogleDrive\Projects\Labor\Codes"


import delimited ".\simNew.csv", clear

rename (v1-v12) (age education x3 x4 choice income educated x5 type Emax choice_next homeSinceSchool)

capture drop conscript
gen conscript = 1
replace conscript = 0 if mi(x5)

gen id = floor((_n - 1) / 50) + 1

replace income = 0 if mi(income)
gcollapse (sum) lifetime_income = income, by(id) fast merge

hashsort id age


matrix result = J(4,3,0)
preserve

    keep if age==65
    gen log_income = log(lifetime_income)

    reghdfe log_income conscript i.type, vce(robust)
    matrix result[1,1] = _b[conscript]
    matrix result[1,2] = _se[conscript]

    reghdfe log_income conscript if type==1, vce(robust)
    matrix result[2,1] = _b[conscript]
    matrix result[2,2] = _se[conscript]

    reghdfe log_income conscript if type==2, vce(robust)
    matrix result[3,1] = _b[conscript]
    matrix result[3,2] = _se[conscript]

    reghdfe log_income conscript if type==3, vce(robust)
    matrix result[4,1] = _b[conscript]
    matrix result[4,2] = _se[conscript]

restore


capture frame drop result
frame create result
frame change result 

svmat result
rename result1 coef
rename result2 se
gen type = _n

replace coef = coef * 100
gen ub = coef + 1.96* se * 100
gen lb = coef - 1.96* se * 100

graph twoway bar coef type, legend(label(1 "Estimate")) barw(0.6) bfcolor(navy%20) bcolor(navy) lwidth(thick) ///
    || rcap ub lb type,  legend(label(2 "95% CI")) color(black%90) lwidth(medium) ///
    ytitle("change (%)") xtitle("population") yline(0, lcolor(gray) lwidth(medthick) lpattern(dash) ) ///
    legend(order(2) pos(2) ring(0) col(3) size(medthin) ) ///
    graphregion(color(white)) name("lifetiemIncome", replace) ///
	xlabel(1 "ATT" 2 "Type One" 3 "Type Two" 4 "Type Three") ///
    ylabel(-10(2)2) xline(1.5, lcolor(gray) lpattern(dash)) ///
	
	// graph export "./Data analysis/Results/D2_1_lifetime_income_change.pdf", replace

frame change default


* ----------------------------------------------------------------------
* conscription and income

matrix result = J(50,3,0)
local enum = 1
quietly forvalues ageRestriction = 24(1)65 {

    ppmlhdfe income conscript if inrange(age, `ageRestriction', `ageRestriction') , vce(robust) absorb(type)

    nlcom (percentage_change: (exp(_b[conscript])-1))
    // matlist r(V)'
    matrix result[`enum',1] = r(b)
    matrix result[`enum',2] = r(V)
    
    matrix result[`enum',3] = `ageRestriction'
    local enum = `enum' + 1
    di `enum'
}

** graph
capture frame drop result
frame create result
frame change result 

svmat result
rename result1 coef
rename result2 se
rename result3 age

drop if age==0

gen ub = coef + 1.96* sqrt(se)
gen lb = coef - 1.96* sqrt(se)
replace coef = coef * 100
replace ub = ub * 100
replace lb = lb * 100

graph twoway line coef age, legend(label(1 "coefficient")) lwidth(thick)  ///
    || rarea ub lb age,  legend(label(2 "95% CI")) color(navy%10) ///
    ytitle("change (%)") xtitle("age")  ///
    legend(order(1 2) pos(11) ring(0) col(2)) ///
    graphregion(color(white)) name("income", replace) ///
	ylabel(-40(10)10) xlabel(25(5)65) xscale(r(23 66)) ///

	// graph export "./Data analysis/Results/D2_1_income_change.pdf", replace


frame change default




* ----------------------------------------------------------------------
* ----------------------------------------------------------------------
* conscription and education

* change working directory
cd "C:\Users\ehsa7798\GoogleDrive\Projects\Labor\Codes"


import delimited ".\simNew.csv", clear

rename (v1-v12) (age education x3 x4 choice income educated x5 type Emax choice_next homeSinceSchool)

capture drop conscript
gen conscript = 1
replace conscript = 0 if mi(x5)

gen id = floor((_n - 1) / 50) + 1

replace income = 0 if mi(income)
gcollapse (sum) lifetime_income = income, by(id) fast merge

hashsort id age

gen study = (choice==2)
preserve 

	keep if inrange(age,20,23)
	reg study conscript i.type i.age

restore

gen blue= (choice==4)
preserve 

	keep if inrange(age,33,35)
	reg blue conscript i.type

restore


* ----------------------------------------------------------------------
import delimited ".\simNew.csv", clear
save ".\simNew.dta", replace

import delimited ".\NoConscription.csv", clear

append using ".\simNew.dta", generate(main)
erase ".\simNew.dta"

// import delimited ".\simNew.csv", clear
// save ".\simNew.dta", replace
// import delimited ".\simNew.csv", clear
// keep if mi(v8)
// append using ".\simNew.dta", generate(main)
// erase ".\simNew.dta"

rename (v1-v12) (age education x3 x4 choice income educated x5 type Emax choice_next homeSinceSchool)

gen study = (choice==2)
preserve 

	keep if inrange(age,20,23)
	sum study
	reg study main

restore