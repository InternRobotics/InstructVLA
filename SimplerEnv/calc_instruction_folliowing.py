import os
import os.path as osp
import argparse

def calc_results(root_dir):
    free_tasks = []
    alt_tasks = []

    for env_dir in os.listdir(root_dir):
        env_dir_path = os.path.join(root_dir, env_dir)
        if os.path.isdir(env_dir_path):
            for control_dir in os.listdir(env_dir_path):
                control_dir_path = os.path.join(env_dir_path, control_dir)
                
                for task_dir in os.listdir(control_dir_path):
                    task_dir_path = os.path.join(control_dir_path, task_dir)
                    task_name = os.path.basename(task_dir_path)

                    if task_name.startswith("Free") or task_name.startswith("Alt"):
                        success_count = 0
                        failure_count = 0

                        if os.path.isdir(task_dir_path):
                            for variance_dir in os.listdir(task_dir_path):
                                task_dir_path_full = os.path.join(task_dir_path, variance_dir)
                                if 'rob_' in variance_dir:
                                    actions_folder = os.path.join(task_dir_path_full, 'actions')
                                    if os.path.isdir(actions_folder): 
                                        for file_name in os.listdir(actions_folder):
                                            if file_name.endswith(".png"):
                                                if "success_obj" in file_name:
                                                    success_count += 1
                                                elif "failure_obj" in file_name:
                                                    failure_count += 1


                        total_files = success_count + failure_count
                        success_probability = success_count / total_files if total_files > 0 else 0

                        task_info = (task_name, success_probability)
                        if task_name.startswith("Free"):
                            free_tasks.append(task_info)
                        elif task_name.startswith("Alt"):
                            alt_tasks.append(task_info)
    
    # get fixed task order
    free_tasks.sort(key=lambda x: x[0])
    alt_tasks.sort(key=lambda x: x[0])

    return free_tasks, alt_tasks

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate success rates from results directory.")
    parser.add_argument(
        "--results-dir",
        type=str,
        required=True,
        help="Path to the results directory (full_path)."
    )
    args = parser.parse_args()

    free_success_rates_with_names, alt_success_rates_with_names = calc_results(args.results_dir)

    free_avg_success_probability = (
        sum(success for _, success in free_success_rates_with_names) / len(free_success_rates_with_names)
        if free_success_rates_with_names else 0
    )
    alt_avg_success_probability = (
        sum(success for _, success in alt_success_rates_with_names) / len(alt_success_rates_with_names)
        if alt_success_rates_with_names else 0
    )

    # format as csv
    print("Category,Task,Success_Rate")
    
    for task_name, success_rate in free_success_rates_with_names:
        print(f"Free,{task_name.split('_')[0]},{success_rate:.4f}")
    
    for task_name, success_rate in alt_success_rates_with_names:
        print(f"Alt,{task_name.split('_')[0]},{success_rate:.4f}")

    print()
    print("Category,Average_Success_Rate")
    print(f"Free,{free_avg_success_probability:.4f}")
    print(f"Alt,{alt_avg_success_probability:.4f}")