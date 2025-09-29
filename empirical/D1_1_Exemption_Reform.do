

* load HEIS data
use "C:\Users\ehsa7798\Documents\all_ind.dta", clear

** variable definitions
* define birth cohort
gen birth_y = year - age
replace birth_y = birth_y + 1921

* define male dummy
gen male = (gender==1)
replace male=. if mi(gender)

// 	replace netincome_w_y = netincome_w_y + income_s_y if !mi(income_s_y)
replace netincome_w_y = 0 if netincome_w_y<0
replace netincome_w_y = netincome_w_y/cpi_y


* ----------------------------------------------------------------------
** graphical evidence on exemption reform and education
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
		text(10 1980.5 "Birth Cohorts" "Exemption Reform", placement(c) color(navy%90)) scheme(s2color)
		
	// graph export "./Data analysis/Results/D1_1_Exemption_Reform_education.pdf", replace

restore


* ----------------------------------------------------------------------
** compare model and empirical estimate of the effect of conscription on study and occupational choice
// preserve

//     * define birth cohort
//     gen birth_y = year - age
//     replace birth_y = birth_y + 1921
//     gen in_school = (studying==1)

//     * define male dummy
//     gen male = (gender==1)
//     replace male=. if mi(gender)

//     * define exposure cohorts 
//     gen cohorts = . 
//     replace cohorts = 0 if inlist(birth_y,1979,1980,1981,1982)
//     replace cohorts = 1 if inrange(birth_y,1985,1990)
//     keep if !mi(cohorts)
	
	
// 	reghdfe in_school i.cohorts##i.male if inrange(age,20,23) [aweight=weight] , vce(robust) noabsorb

// restore


* ----------------------------------------------------------------------
matrix result = J(30,3,0)
preserve

    * define exposure cohorts 
    gen cohorts = . 
    replace cohorts = 0 if inlist(birth_y,1979,1980,1981,1982)
    replace cohorts = 1 if inrange(birth_y,1985,1990)
    keep if !mi(cohorts)

    eststo clear
	local enum = 1
	forvalues ageRestriction = 24(3)34 {
        
    poisson netincome_w_y i.cohorts##i.male i.year ///
        if inrange(age, `ageRestriction', `ageRestriction'+2) ///
        [iweight=weight], vce(cluster birth_y)
    
    * store regression
    eststo age`ageRestriction', title("`ageRestriction'-`= `ageRestriction' + 2'")
    
    * run nlcom for percentage effect
    quietly nlcom (pct_effect: (exp(_b[1.cohorts#1.male]) - 1)*100)
    
    local peff : display %6.2f r(b)[1,1]
    local pse  : display %6.2f sqrt(r(V)[1,1])
    
    * add coefficient normally
    estadd scalar pct_eff = r(b)[1,1]
    
    * add SE as a string with parentheses
    estadd local pct_se_str = "(" + "`pse'" + ")"

    matrix result[`enum',1] = r(b)[1,1]
    matrix result[`enum',2] = r(V)[1,1]
    
    matrix result[`enum',3] = `ageRestriction'
    local enum = `enum' + 1
	
    ** with additional controls
	
	poisson netincome_w_y i.cohorts##i.male i.year i.urban i.province ///
        if inrange(age, `ageRestriction', `ageRestriction'+2) ///
        [iweight=weight], vce(cluster birth_y)
    
    * store regression
    eststo age2`ageRestriction', title("`ageRestriction'-`= `ageRestriction' + 2'")
    
    * run nlcom for percentage effect
    quietly nlcom (pct_effect: (exp(_b[1.cohorts#1.male]) - 1)*100)
    
    local peff : display %6.2f r(b)[1,1]
    local pse  : display %6.2f sqrt(r(V)[1,1])
    
    * add coefficient normally
    estadd scalar pct_eff = r(b)[1,1]
    
    * add SE as a string with parentheses
    estadd local pct_se_str = "(" + "`pse'" + ")"
	
	
    }


    esttab age* using "./results/empirical/D1_1_table_poisson_main.tex", replace ///
        keep(1.cohorts#1.male) varlabels(1.cohorts#1.male "Conscription * Male") ///
        b(2) se(2) star(* 0.10 ** 0.05 *** 0.01) ///
        stats(pct_eff pct_se_str N, ///
            fmt(2 s 0) ///
            labels("Implied effect (\%)" "SE (percentage effect)" "Observations")) ///
        indicate("year fixed effect = *.year" ///
                "gender FE = 1.male" ///
                "cohort FE = 1.cohorts" "rural dummy = 1.urban" "province FE = *.province", labels("Y")) ///
		mgroups("\shortstack{Age 24-26}" "\shortstack{Age 27-29}" "\shortstack{Age 30-32}" "\shortstack{Age 33-35}", ///
		pattern(1 0 1 0 1 0 1 0) prefix(\multicolumn{@span}{c}{) suffix(}) span erepeat(\cmidrule(lr){@span})) ///
        nomtitles nonotes

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
        ylabel(-20(5)5)
    //     xlabel(25(3)30) ///

    gen type = "data"
    save "./data/temp/data_result.dta", replace

frame change default



* ----------------------------------------------------------------------
* robustness for parallel trends assumption
matrix result = J(30,3,0)
preserve

    * define exposure cohorts 
    gen cohorts = . 
    replace cohorts = 0 if inrange(birth_y,1985,1990)
    replace cohorts = 1 if inrange(birth_y,1971-5,1971)
    keep if !mi(cohorts)

    eststo clear
	
    local enum = 1
	forvalues ageRestriction = 24(3)34 {
        
    poisson netincome_w_y i.cohorts##i.male i.year ///
        if inrange(age, `ageRestriction', `ageRestriction'+2) ///
        [iweight=weight], vce(cluster birth_y)
    
    * store regression
    eststo age`ageRestriction', title("`ageRestriction'-`= `ageRestriction' + 2'")
    
    * run nlcom for percentage effect
    quietly nlcom (pct_effect: (exp(_b[1.cohorts#1.male]) - 1)*100)
    
    local peff : display %6.2f r(b)[1,1]
    local pse  : display %6.2f sqrt(r(V)[1,1])
    
    * add coefficient normally
    estadd scalar pct_eff = r(b)[1,1]
    
    * add SE as a string with parentheses
    estadd local pct_se_str = "(" + "`pse'" + ")"

    matrix result[`enum',1] = r(b)[1,1]
    matrix result[`enum',2] = r(V)[1,1]
    
    matrix result[`enum',3] = `ageRestriction'
    local enum = `enum' + 1
	
    ** with additional controls
	
	poisson netincome_w_y i.cohorts##i.male i.year i.urban i.province ///
        if inrange(age, `ageRestriction', `ageRestriction'+2) ///
        [iweight=weight], vce(cluster birth_y)
    
    * store regression
    eststo age2`ageRestriction', title("`ageRestriction'-`= `ageRestriction' + 2'")
    
    * run nlcom for percentage effect
    quietly nlcom (pct_effect: (exp(_b[1.cohorts#1.male]) - 1)*100)
    
    local peff : display %6.2f r(b)[1,1]
    local pse  : display %6.2f sqrt(r(V)[1,1])
    
    * add coefficient normally
    estadd scalar pct_eff = r(b)[1,1]
    
    * add SE as a string with parentheses
    estadd local pct_se_str = "(" + "`pse'" + ")"
	
	
    }


    esttab age* using "./results/empirical/D1_1_table_poisson_placebo.tex", replace ///
        keep(1.cohorts#1.male) varlabels(1.cohorts#1.male "Conscription * Male") ///
        b(2) se(2) star(* 0.10 ** 0.05 *** 0.01) ///
        stats(pct_eff pct_se_str N, ///
            fmt(2 s 0) ///
            labels("Implied effect (\%)" "SE (percentage effect)" "Observations")) ///
        indicate("year fixed effect = *.year" ///
                "gender FE = 1.male" ///
                "cohort FE = 1.cohorts" "rural dummy = 1.urban" "province FE = *.province", labels("Y")) ///
		mgroups("\shortstack{Age 24-26}" "\shortstack{Age 27-29}" "\shortstack{Age 30-32}" "\shortstack{Age 33-35}", ///
		pattern(1 0 1 0 1 0 1 0) prefix(\multicolumn{@span}{c}{) suffix(}) span erepeat(\cmidrule(lr){@span})) ///
        nomtitles nonotes

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
        ylabel(-20(5)5)
    //     xlabel(25(3)30) ///

frame change default




* ----------------------------------------------------------------------
** calculate estimates implied by the model

import delimited ".\data\simulation\simNew.csv", clear
save ".\data\simulation\simNew.dta", replace

import delimited ".\data\simulation\NoConscription.csv", clear

append using ".\data\simulation\simNew.dta", generate(main)
erase ".\data\simulation\simNew.dta"

rename (v1-v12) (age education x3 x4 choice income educated x5 type Emax choice_next homeSinceSchool)

* main = 0 is no conscription

* ----------------------------------------------------------------------
matrix result = J(30,3,0)
preserve
	
	// drop if !mi(x5)
	replace income = 0 if mi(income)
	
    local enum = 1
    forvalues ageRestriction = 24(3)40 {
	
        // ppmlhdfe income main if inrange(age, `ageRestriction', `ageRestriction'+2) , vce(robust)
        poisson income i.main if inrange(age, `ageRestriction', `ageRestriction'+2), vce(robust)
		// matlist e(b)'

        nlcom (percentage_change: (exp(_b[1.main])-1))
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

gen type = "model"
save "./data/temp/model_result.dta",replace

// graph twoway bar coef age, legend(label(1 "coefficient")) barw(1.5) bfcolor(navy%20) bcolor(navy) lwidth(thick) ///
//     || rcap ub lb age,  legend(label(2 "95% CI")) color(black%80) lwidth(medthick) ///
//     ytitle("Coefficient") xtitle("age") yline(0, lcolor(khaki) lwidth(thick) lpattern(dash) ) ///
//     legend(order(1 2) pos(11) ring(0) col(2)) ///
//     graphregion(color(white)) name("inc", replace) ///
// 	ylabel(-0.2(0.05)0.05)
// //     xlabel(22(2)34) ///

frame change default



* ----------------------------------------------------------------------
** combine empirical and model estimates

capture frame drop graph
frame create graph
frame change graph 

use "./data/temp/model_result.dta", clear

append using "./data/temp/data_result.dta"

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

frame change default

// 	graph export "./Data analysis/Results/D1_1_Exemption_Reform_earnings.pdf", replace

// erase "./data/temp/data_result.dta"
// earase "./data/temp/model_result.dta"




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




