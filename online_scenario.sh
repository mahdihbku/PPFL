#!/bin/bash

# List of integer parameters for client.py instances
# Ns=(100 200 300 400 500)
Ns=(3 5)

# List of integer parameters for committee.py instances
# Cs=(3 5 10 15 20)
Cs=(2 3)

# Clear output files
rm experiments/online_comp/server.csv
rm experiments/online_comp/client.csv
rm experiments/online_comp/committee.csv

# Kill any existing instance of the server
lsof -i :10000 | grep LISTEN | awk '{print $2}' | xargs kill -9

for N in "${Ns[@]}"
do
    for C in "${Cs[@]}"
    do
        sed -i '' "s/^num_participants = .*/num_participants = $N/" "params.py"
        sed -i '' "s/^committee_size = .*/committee_size = $C/" "params.py"

        echo "Running script for N=$N and C=$C"
        # Launch server.py
        echo "Launching server.py"
        python PPserver.py &
        server_pid=$!

        sleep 2

        for ((i=1; i<=N; i++))
        do
            echo "Launching python client.py $i"
            python PPclient.py "$i" > /dev/null &
        done

        sleep 2

        # Launch committee.py instances with different parameters
        for ((i=1; i<=C; i++))
        do
            port=$((12343 + i))
            echo "Launching committee.py $port"
            python PPcommitteeMember.py "$port" > /dev/null &
        done

        echo "All scripts launched."
        wait $server_pid
    done
done

# To clear things:
# kill a process using the port $port_number
# lsof -i :$port_number | grep LISTEN | awk '{print $2}' | xargs kill -9
