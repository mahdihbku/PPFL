set terminal png font "Helvetica,14"
set output "vrf2.png"
set datafile sep ','
set key bottom right Right offset 0, 1
set xlabel "Number of concurrent verifications"
set ylabel 'Computation time (ms)'
unset log x
# set xtics 0,2000,100000
# set xtics add ("100" 100)
# set log x
plot [:30][0:] 'vrf_verif_time.csv' using 2:3 title 'Verification time (Verify)' with linespoints pt 6 ps 2 lc 3

