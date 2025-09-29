

* change working directory
// cd "C:\Users\ehsa7798\GoogleDrive\Projects\Labor\Codes"
cd "C:\Users\ehsa7798\My Drive\Projects\Labor\Codes"

* load HEIS data
// use "C:\Academic\HEIS\all_ind.dta", clear
use "C:\Users\ehsa7798\Documents\all_ind.dta", clear

* ----------------------------------------------------------------------
preserve 

    gen birth_y = year - age
	replace birth_y = birth_y + 1921
    gen in_school = (studying==1)

    keep if inrange(birth_y, 1975, 1986)
    keep if inrange(age, 20, 23)

    gcollapse (mean) in_school (semean) std = in_school [weight=weight], by(birth_y gender) fast
	
    gen ub = (in_school + 1.96* std)*100
    gen lb = (in_school - 1.96* std)*100
	replace in_school = in_school * 100
	
    replace birth_y = birth_y+0.06 if gender==1
    replace birth_y = birth_y-0.06 if gender==2

    graph twoway connected in_school birth_y if gender==1, legend(label(1 "male")) color(navy) msymbol(O) lwidth(thin) ///
        || rcap ub lb birth_y if gender==1, color(navy) legend(label(2 "male")) lwidth(medium) ///
        || connected in_school birth if gender==2, legend(label(3 "female")) color(dkgreen) msymbol(S) lwidth(thin) /// 
        || rcap ub lb birth_y if gender==2, color(dkgreen) legend(label(4 "female")) lwidth(medium) ///
        ytitle("Proportion in school") xtitle("birth year") ///
        legend(order(1 3) pos(11) ring(0) col(2)) ///
        xlabel(1975(1)1986) ylabel(10(5)25) yscale(r(9 27)) ///
        graphregion(color(white)) name("educ", replace) ///
		xline(1980.5, lc(gray%13) lwidth(35)) ///
		text(10 1980.5 "Birth Cohorts" "Exemption Reform", placement(c) color(navy%90))
		
	// graph export "./Data analysis/Results/D1_1_Exemption_Reform_education.pdf", replace

restore



preserve

    * define birth cohort
    gen birth_y = year - age
    replace birth_y = birth_y + 1921
    gen in_school = (studying==1)

    * define male dummy
    gen male = (gender==1)
    replace male=. if mi(gender)

    * define exposure cohorts 
    gen cohorts = . 
    replace cohorts = 0 if inlist(birth_y,1979,1980,1981,1982)
    replace cohorts = 1 if inrange(birth_y,1985,1990)
    keep if !mi(cohorts)
	
	
	reghdfe in_school i.cohorts##i.male if inrange(age,20,23) [aweight=weight] , vce(robust) noabsorb

restore



* ----------------------------------------------------------------------
// matrix result = J(20,3,0)

// preserve

//     * define birth cohort
//     gen birth_y = year - age
//     replace birth_y = birth_y + 1921

//     * define male dummy
//     gen male = (gender==1)
//     replace male=. if mi(gender)

//     // * define placebo cohorts 
//     // gen cohorts = . 
//     // replace cohorts = 1 if inrange(birth_y,1985,1987)
//     // replace cohorts = 0 if inrange(birth_y,1988,1991)
//     // keep if !mi(cohorts)

//     * define exposure cohorts 
//     gen cohorts = . 
//     replace cohorts = 0 if inlist(birth_y,1979,1980,1981,1982)
//     replace cohorts = 1 if inrange(birth_y,1985,1988)
// // 	replace cohorts = 1 if inrange(birth_y,1970,1975)
// //     replace cohorts = 1 if inlist(birth_y, 1978,1977,1976,1975)
//     keep if !mi(cohorts)

//     // hist age, discrete by(cohorts) scheme(lean2) xlabel(10(2)40)
//     // gen hours = .
//     // replace hours = hours_w*days_w if !mi(hours_w)
//     // replace hours = hours_s*days_s*52 if mi(hours_w)

//     // gen wage = .
//     // replace wage = wage_w_m*40/hours if !mi(hours_w)
//     // replace wage = income_s_y/hours if mi(hours_w)

//     // gen income = income_s_y + netincome_w_y 
// 	replace netincome_w_y=0 if  netincome_w_y<0
//     replace netincome_w_y = netincome_w_y/cpi_y
//     gen lnIncome = log(netincome_w_y)
//     // gen lnIncome = netincome_w_y/cpi_y
    
//     // gen self_employed = (income_s_y>0)

//     local enum = 1
//     forvalues ageRestriction = 22(2)34 {
//         // keep if inlist(age,`age')

//         reghdfe lnIncome i.cohorts##i.male if inrange(age, `ageRestriction', `ageRestriction'+1) [weight=weight], absorb(province)
//         matrix result[`enum',1] = _b[1.cohorts#1.male]
//         matrix result[`enum',2] = _se[1.cohorts#1.male]
        
//         matrix result[`enum',3] = `ageRestriction'
//         local enum = `enum' + 1
//     }

// restore


// capture frame drop result
// frame create result
// frame change result

// svmat result
// rename result1 coef
// rename result2 se
// rename result3 age

// drop if age==0

// gen ub = coef + 1.96* se
// gen lb = coef - 1.96* se

// // graph twoway connected coef age, legend(label(1 "coefficient")) color(green) msymbol(O) ///
// //     || rcap ub lb age,  legend(label(2 "95% CI")) color(green) ///
// //     ytitle("Coefficient") xtitle("age") ///
// //     legend(order(1 2) pos(11) ring(0) col(2)) ///
// //     xlabel(22(2)34) ///
// //     graphregion(color(white))

// graph twoway bar coef age, legend(label(1 "coefficient")) barw(1.5) bfcolor(navy%20) bcolor(navy) lwidth(thick) ///
//     || rcap ub lb age,  legend(label(2 "95% CI")) color(black%80) lwidth(medthick) ///
//     ytitle("Coefficient") xtitle("age") yline(0, lcolor(khaki) lwidth(thick) lpattern(dash) ) ///
//     legend(order(1 2) pos(11) ring(0) col(2)) ///
//     xlabel(22(2)34) ylabel(-0.4(0.2)0.3) ///
//     graphregion(color(white)) name("inc", replace)



// frame change default




* ----------------------------------------------------------------------
matrix result = J(20,3,0)
preserve

    * define birth cohort
    gen birth_y = year - age
    replace birth_y = birth_y + 1921

    * define male dummy
    gen male = (gender==1)
    replace male=. if mi(gender)

    * define exposure cohorts 
    gen cohorts = . 
    replace cohorts = 0 if inlist(birth_y,1979,1980,1981,1982)
    replace cohorts = 1 if inrange(birth_y,1985,1990)
    // replace cohorts = 1 if inlist(birth_y,1976,1975,1974)
    keep if !mi(cohorts)
	
// 	replace netincome_w_y = netincome_w_y + income_s_y if !mi(income_s_y)
    replace netincome_w_y = 0 if netincome_w_y<0
    replace netincome_w_y = netincome_w_y/cpi_y

    local enum = 1
    forvalues ageRestriction = 24(3)34 {

        poisson netincome_w_y i.cohorts##i.male if inrange(age, `ageRestriction', `ageRestriction'+2) [iweight=weight] , vce(robust)
		// matlist e(b)'

        nlcom (percentage_change: (exp(_b[1.cohorts#1.male])-1))
		// matlist r(V)'
        matrix result[`enum',1] = r(b)
        matrix result[`enum',2] = r(V)
        
        matrix result[`enum',3] = `ageRestriction'
        local enum = `enum' + 1
    }

restore


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


graph twoway bar coef age, legend(label(1 "coefficient")) barw(1.5) bfcolor(navy%20) bcolor(navy) lwidth(thick) ///
    || rcap ub lb age,  legend(label(2 "95% CI")) color(black%80) lwidth(medthick) ///
    ytitle("Coefficient") xtitle("age") yline(0, lcolor(khaki) lwidth(thick) lpattern(dash) ) ///
    legend(order(1 2) pos(11) ring(0) col(2)) ///
    graphregion(color(white)) name("inc", replace) ///
	ylabel(-0.2(0.05)0.05)
//     xlabel(25(3)30) ///

gen type = "data"
save "./data_result.dta",replace


frame change default



* ----------------------------------------------------------------------
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

* main = 0 is no conscription

* ----------------------------------------------------------------------
matrix result = J(30,3,0)
preserve
	
	// drop if !mi(x5)
	replace income = 0 if mi(income)
	
    local enum = 1
    forvalues ageRestriction = 24(3)40 {
	
        ppmlhdfe income main if inrange(age, `ageRestriction', `ageRestriction'+2) //, vce(robust)
		// matlist e(b)'

        nlcom (percentage_change: (exp(_b[main])-1))
		// matlist r(V)'
        matrix result[`enum',1] = r(b)
        matrix result[`enum',2] = r(V)
        
        matrix result[`enum',3] = `ageRestriction'
        local enum = `enum' + 1
    }

restore


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


// graph twoway bar coef age, legend(label(1 "coefficient")) barw(1.5) bfcolor(navy%20) bcolor(navy) lwidth(thick) ///
//     || rcap ub lb age,  legend(label(2 "95% CI")) color(black%80) lwidth(medthick) ///
//     ytitle("Coefficient") xtitle("age") yline(0, lcolor(khaki) lwidth(thick) lpattern(dash) ) ///
//     legend(order(1 2) pos(11) ring(0) col(2)) ///
//     graphregion(color(white)) name("inc", replace) ///
// 	ylabel(-0.2(0.05)0.05)
// //     xlabel(22(2)34) ///


gen type = "model"
append using "./data_result.dta"

replace age = age + 0.53 if type=="model" & age<36
replace age = age - 0.53 if type=="data"
replace coef = coef * 100
replace ub = ub * 100
replace lb = lb * 100

graph twoway bar coef age if type=="model", legend(label(1 "Model")) barw(1) bfcolor(dkorange%20) bcolor(dkorange) lwidth(thick) ///
	|| bar coef age if type=="data", legend(label(2 "Empirical Estimate")) barw(1) bfcolor(navy%20) bcolor(navy) lwidth(thick) ///
    || rcap ub lb age,  legend(label(3 "95% CI")) color(black%60) lwidth(medthin) ///
    ytitle("change (%)") xtitle("age") yline(0, lcolor(gray) lwidth(medthick) lpattern(dash) ) ///
    legend(order(1 2 3) pos(11) ring(0) col(3) size(medthin) ) ///
    graphregion(color(white)) name("inc", replace) ///
	ylabel(-25(5)10) ///
	xlabel(24 "24-26" 27 "27-29" 30 "30-32" 33 "33-35" 36 "36-38" 39 "39-41")
	
// 	graph export "./Data analysis/Results/D1_1_Exemption_Reform_earnings.pdf", replace

frame change default






* ----------------------------------------------------------------------
* occupation

// * change working directory
// cd "C:\Users\ehsa7798\GoogleDrive\Projects\Labor\Codes"
//
// * load HEIS data
// use "C:\Academic\HEIS\all_ind.dta", clear
//
// * Create occupation variable
// replace ISCO_w = ISCO_s if mi(ISCO_w)
// gen occupation = real(substr(string(ISCO_w, "%04.0f"), 1, 1))
//
// gen choice = ""
// replace choice = "study" if (studying==1)
// replace choice = "white-collar" if inlist(occupation, 1, 2, 3)
// replace choice = "blue-collar" if inlist(occupation, 4, 5, 6, 7, 8, 9)
//
//
// preserve
//
//     * define birth cohort
//     gen birth_y = year - age
//     replace birth_y = birth_y + 1921
//
//     * define male dummy
//     gen male = (gender==1)
//     replace male=. if mi(gender)
//
//     * define exposure cohorts 
//     gen cohorts = . 
//     replace cohorts = 0 if inlist(birth_y,1979,1980,1981,1982)
//     replace cohorts = 1 if inrange(birth_y,1985,1990)
//     keep if !mi(cohorts)
// 	keep if inrange(age, 30, .)
//     gen bluecollar = (choice=="blue-collar")
//     reghdfe bluecollar i.cohorts##i.male [aweight=weight] , vce(robust) absorb(age)
//	
// 	gen whitecollar = (choice=="white-collar")
//     reghdfe whitecollar i.cohorts##i.male [aweight=weight] , vce(robust) absorb(age)
//	
// // 	gen study = (choice=="study")
// // 	reghdfe study i.cohorts##i.male [aweight=weight] , vce(robust) absorb(age)
// //
// // 	gen else = (choice=="")
// // 	reghdfe else i.cohorts##i.male [aweight=weight] , vce(robust) absorb(age)
//	
// 	gen nonemployed = !inlist(occupationalst,1,4)
// 	reghdfe nonemployed i.cohorts##i.male [aweight=weight] , vce(robust) absorb(age)
//
//
// restore




