bash /mnt/petrelfs/lihao3/EmbodiedMLLM/CogACTx_TMP_scale_law/scripts_test_SimplerEnv/RUN_all_test_cogact_oxe_10.sh &
pid1=$!
bash /mnt/petrelfs/lihao3/EmbodiedMLLM/CogACTx_TMP_scale_law/scripts_test_SimplerEnv/RUN_all_test_cogact_oxe_11.sh &
pid2=$!
wait $pid1 $pid2