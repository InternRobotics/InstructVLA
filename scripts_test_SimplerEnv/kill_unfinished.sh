pkill -f 'bash .*scripts_self.*\.sh'

pgrep -f 'python .*run_libero_eval.py' | xargs -r kill -9

pgrep -f 'python .*simpler_env/main_inference_server.py' | xargs -r kill -9

pgrep -f 'python .*simpler_env/main_inference_client.py' | xargs -r kill -9

pgrep -f 'python .*simpler_env/main_inference.py' | xargs -r kill -9