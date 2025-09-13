NUM=2

pwd

bash scripts_test_SimplerEnv/simpler_0.sh $NUM 0 & 
pid1=$!
bash scripts_test_SimplerEnv/simpler_1.sh $NUM 1 &
pid2=$!

wait $pid1 $pid2