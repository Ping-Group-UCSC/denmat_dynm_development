#!/usr/bin/gnuplot -persist
set xtics ( "G" 0,  "X" 142,  "W" 213,  "L" 284,  "G" 458,  "K" 642 )
unset key
set terminal png
set output "phbs_loto.png"
nRows = real(system("awk '$1==\"kpoint\" {nRows++} END {print nRows}' bandstruct.kpoints"))
nCols = 6
dim=300
set arrow from 142,0. to 142,dim nohead lw 0.5 lt 2 lc rgb 'black'
set arrow from 213,0. to 213,dim nohead lw 0.5 lt 2 lc rgb 'black'
set arrow from 284,0. to 284,dim nohead lw 0.5 lt 2 lc rgb 'black'
set arrow from 458,0. to 458,dim nohead lw 0.5 lt 2 lc rgb 'black'
set ylabel "Phonon energy (1/cm)" font ",15"
set yrange [0:300]
ha2mev = 27211.386
ha2cmm1 = 219474.6313632043
ha2thz = 6579.68974479
plot for [i=1:nCols] "phfrq_loto.dat" u 0:(column(i)*ha2cmm1) w l lw 2 lc rgb "red"
