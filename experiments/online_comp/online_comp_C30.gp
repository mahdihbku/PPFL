set terminal png font "Helvetica,14" 
set output "online_comp_C30.png"
set datafile sep ','
set key bottom right Right offset 0, 1
set xlabel "Number of clients"
set ylabel "Computation cost (ms)"
set xtics 0,200,1000
set ytics add ("80" 80)
set log y

# x=number of clients, for committee_size=10
plot [0:][:81] 'online_sim.csv' every 5::4 using 1:3 title 'Client' with linespoints lc 7 pt 2 ps 2, \
             'online_sim.csv' every 5::4 using 1:5 title 'Server' with linespoints pt 4 ps 2 lc 2, \
             'online_sim.csv' every 5::4 using 1:4 title 'Committee member' with linespoints pt 6 ps 2 lc 3
