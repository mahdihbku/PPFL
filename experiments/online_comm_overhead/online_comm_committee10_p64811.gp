set terminal png font "Helvetica,14" 
set output "online_comm_committee10_p64811.png"
set datafile sep ','
set key center right Right offset 0, -3
set xlabel "Number of clients"
set ylabel "Overhead communication cost"
set xtics 0,100,500
set ytics add ("500 KB" 0.5)
set ytics add ("1 MB" 1)
# set ytics add ("3 MB" 3)
set ytics add ("5 MB" 5)
set ytics add ("10 MB" 10)
set ytics add ("100 MB" 100)
set ytics add ("200 MB" 200)
set log y

# x=number of clients, for committee_size=10 and p=64811
plot [0:][0.5:201] '../online_comm/online_comm.csv' every 45::20 using 1:($4-$7) title 'Client' with linespoints lc 7 pt 2 ps 2, \
             '../online_comm/online_comm.csv' every 45::20 using 1:($5-$8) title 'Server' with linespoints pt 4 ps 2 lc 2, \
             '../online_comm/online_comm.csv' every 45::20 using 1:6 title 'Committee member' with linespoints pt 6 ps 2 lc 3
