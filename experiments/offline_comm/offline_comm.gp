set terminal png font "Helvetica,14"
set output "off_comm.png"
set datafile sep ','
set key bottom right Right
set xlabel "Number of clients"
set ylabel 'Communication cost'
set xtics 0,100,500
set ytics add ("0.5 KB" 0.5)
set ytics add ("1 KB" 1)
set ytics add ("10 KB" 10)
set ytics add ("100 KB" 100)
set ytics add ("1 MB" 1000)
set ytics add ("10 MB" 10000)
set log y
plot [0:][0.5:13000] 'offline_comm.csv' using 1:2 title 'Client communication' with linespoints pt 2 ps 2 lc 1, 'offline_comm.csv' using 1:3 title 'Server communication' with linespoints pt 4 ps 2 lc 2
