bash /mnt/petrelfs/lihao3/EmbodiedMLLM/CogACTx_DIT_Atten_HisF_MultiF_R_Silence/scripts_test_SimplerEnv/RUN_all_test_cogact_5.sh &
pid1=$!
bash /mnt/petrelfs/lihao3/EmbodiedMLLM/CogACTx_DIT_Atten_HisF_MultiF_R_Silence/scripts_test_SimplerEnv/RUN_all_test_cogact_7.sh &
pid2=$!
wait $pid1 $pid2