set terminal png font "Helvetica,14" 
set output "online_comm_committee15_p64811.png"
set datafile sep ','
set key top right Right offset 0, -3.5
set xlabel "Number of clients"
set ylabel "Communication cost"
set xtics 0,100,500
set ytics add ("500 KB" 0.5)
set ytics add ("1 MB" 1)
set ytics add ("3 MB" 3)
set ytics add ("5 MB" 5)
set ytics add ("10 MB" 10)
set ytics add ("100 MB" 100)
set ytics add ("1 GB" 1000)
set ytics add ("2 GB" 2000)
set log y

# x=number of clients, for committee_size=10 and p=64811
plot [0:][0.5:2001] 'online_comm.csv' every 45::29 using 1:4 title 'Client communication' with linespoints lc 7 pt 2 ps 2, \
             'online_comm.csv' every 45::29 using 1:7 title 'Client communication baseline' with linespoints lt 0 lc 7 pt 2 ps 2, \
             'online_comm.csv' every 45::29 using 1:5 title 'Server communication' with linespoints pt 4 ps 2 lc 2, \
             'online_comm.csv' every 45::29 using 1:8 title 'Server communication baseline' with linespoints lt 0 pt 4 ps 2 lc 2, \
             'online_comm.csv' every 45::29 using 1:6 title 'Committee member communication' with linespoints pt 6 ps 2 lc 3
