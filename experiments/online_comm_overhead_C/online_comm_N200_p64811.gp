set terminal png font "Helvetica,14" 
set output "online_comm_N200_p64811.png"
set datafile sep ','
set key center right Right offset 0, -4
set xlabel "C"
set ylabel "Overhead communication cost"
set xtics ("3" 3,"5" 5, "10" 10,"15" 15,"20" 20)
set ytics add ("500 KB" 0.5)
set ytics add ("1 MB" 1)
# set ytics add ("3 MB" 3)
set ytics add ("5 MB" 5)
set ytics add ("10 MB" 10)
set ytics add ("70 MB" 70)
set ytics add ("200 MB" 200)
set log y

# x=C, for N=200 and P=64811
plot [3:][0.5:70] '../online_comm/online_comm.csv' every 9::91::135 using 2:($4-$7) title 'Client' with linespoints lc 7 pt 2 ps 2, \
             '../online_comm/online_comm.csv' every 9::91::135 using 2:($5-$8) title 'Server' with linespoints pt 4 ps 2 lc 2, \
             '../online_comm/online_comm.csv' every 9::91::135 using 2:6 title 'Committee member' with linespoints pt 6 ps 2 lc 3
