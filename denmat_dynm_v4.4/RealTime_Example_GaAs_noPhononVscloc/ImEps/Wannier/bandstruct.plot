#!/usr/bin/gnuplot -persist
set xtics ( "Gamma" 0,  "X" 57,  "W" 86,  "L" 115,  "Gamma" 185,  "K" 259 )
unset key
nRows = real(system("awk '$1==\"kpoint\" {nRows++} END {print nRows}' bandstruct.kpoints"))
nCols = real(system("wc -c < bandstruct.eigenvals")) / (8*nRows)
formatString = system(sprintf("echo '' | awk 'END { str=\"\"; for(i=0; i<%d; i++) str = str \"%%\" \"lf\"; print str}'", nCols))
set term png
set output "bandstruct.png"
VBM=0. #0.229499
ha2ev=1. #27.2114
set xzeroaxis               #Add dotted line at zero energy
set xrange [179:191]
set ylabel "E - VBM [Ha]"   #Add y-axis label
set yrange [0.306:0.323]
plot for [i=1:nCols] "bandstruct.eigenvals" binary format=formatString u 0:((column(i)-VBM)*ha2ev) w lp lc rgb "black", \
     for [i=1:14] "wannier.eigenvals" u (1*column(0)):((column(i)-VBM)*ha2ev) w lp lc rgb "red", \
     0.4035 t ""
