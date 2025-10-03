cd("G:\\My Drive\\Projects\\Labor\\Github\\Compulsory-Service-Structural-Model")
using DelimitedFiles

#= code for reading in server =#

wageMomentStdBoot= readdlm("data/moments/wageMomentStdBootByCollarOnly.csv",',')      ;
# choiceMomentStdBoot = readdlm("/home/sabouri/Dropbox/Labor/Codes/Moments/choiceMomentStdBoot.csv",',') ;
# educatedShareStdBoot = readdlm("/home/sabouri/Dropbox/Labor/Codes/Moments/educatedShareStdBoot.csv",',') ;

choiceMomentStdBoot = readdlm("data/moments/choiceMomentSTDLFS.csv") ;
educatedShareStdBoot = readdlm("data/moments/educatedShareSTDLFS.csv") ;

transMomentStdBoot = readdlm("data/moments/transMomentStdBoot.csv")