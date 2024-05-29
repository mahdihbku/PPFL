# set terminal postscript eps enhanced "Helvetica" 24
# set output 'vrf_keygen_prove_time.eps'
set terminal png font "Helvetica,14"
set output "vrf1.png"
set datafile sep ','
set key center right Right
set xlabel "Input size (Bytes)"
set ylabel 'Computation time (ms)'
# set xtics 0,2000,100000
set xtics add ("100" 100)
set xtics add ("1K" 1000)
set xtics add ("10K" 10000)
set xtics add ("100K" 100000)
set xtics add ("1M" 1000000)
set log x
plot [100:][0:] 'vrf_keygen_prove_time.csv' using 1:2 title 'Key generation time (Gen)' with linespoints pt 2 ps 2 lc 1, 'vrf_keygen_prove_time.csv' using 1:3 title 'Proof generation time (Eval)' with linespoints pt 4 ps 2 lc 2, 'vrf_keygen_prove_time.csv' using 1:4 title 'Verification time (Verify)' with linespoints pt 6 ps 2 lc 3
