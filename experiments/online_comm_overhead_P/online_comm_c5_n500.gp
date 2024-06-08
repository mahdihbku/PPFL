set terminal png font "Helvetica,14" 
set output "online_comm_c5_n500.png"
set datafile sep ','
set key bottom right Right offset 0, 1
set xlabel "P" offset 0, -1
set ylabel "Overhead communication cost"
set ytics add ("100 KB" 0.1)
set ytics add ("1 MB" 1)
set ytics add ("3 MB" 3)
set ytics add ("10 MB" 10)
set ytics add ("100 MB" 100)
set ytics add ("1 GB" 1000)
set ytics add ("10 GB" 10000)
set ytics add ("" 20000)
set ytics add ("" 50000)
set ytics add ("" 60000)
set ytics add ("" 80000)
set ytics add ("100 GB" 100000)
set ytics add ("200 GB" 200000)
set xtics add ("10^4" 10000)
set xtics add ("|\nBasic" 64811)
set xtics add ("10^5" 100000)
set xtics add ("10^6" 1000000)
set xtics add ("10^7" 10000000)
# set xtics add ("|\nResNet-18" 11511784)
set xtics add ("|\nResNet-50" 25557032)
set xtics add ("10^8" 100000000)
set xtics add ("|\nVGG16" 138357544)
set log y
set log x
# set xtics rotate by 90 right

# x=p, for committee_size=5 and p=100
plot [9999:138357545][:220001] '../online_comm/online_comm.csv' every ::234::242 using 3:($4-$7) title 'Client' with linespoints pt 2 ps 2 lc 7, \
             '../online_comm/online_comm.csv' every ::234::242 using 3:($5-$8) title 'Server' with linespoints pt 4 ps 2 lc 2, \
             '../online_comm/online_comm.csv' every ::234::242 using 3:6 title 'Committee member' with linespoints pt 6 ps 2 lc 3
