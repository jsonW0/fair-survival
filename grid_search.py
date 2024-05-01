import subprocess
import time


subprocess.call(["bash", "scripts/real_world.sh"])
weibulls = [("1.5","1"),("4","2")]
count = 0
start = time.perf_counter()
total = 3*2*3*2*2*2*3
for model in ["coxph","uniform","randomforest"]:
    for group_repr in [("0.1","0.9"),("0.5","0.5")]:
        for censorship_repr in [("0.1","0.9"),("0.5","0.5"),("0.9","0.1")]:
            for mean in [("0","0"),("0","5")]:
                for weibull_i in range(len(weibulls)):
                    for weibull_j in range(len(weibulls)):
                        scale = (weibulls[weibull_i][0],weibulls[weibull_j][0])
                        shape = (weibulls[weibull_i][1],weibulls[weibull_j][1])
                        for censorship_times in [("0","1","0","1"),("0","0.1","0.9","1"),("0.9","1","0","0.1")]:
                            now = time.perf_counter()
                            print(f"{count}/{total} ({count/total*100:.2f}%) {now-start:.2f}s/{(now-start)/(count/total+1e-6):.2f}s)")
                            experiment_name = f"synthetic_{count}"
                            subprocess.call(["python", "run_survival.py",
                                            "--model", model,
                                            "--dataset", "synthetic",
                                            "--experiment_name", experiment_name,
                                            "--seed", "1",
                                            "--num_trials", "5",
                                            "--N", "1000",
                                            "--G", "2",
                                            "--D", "1",
                                            "--repr", group_repr[0], group_repr[1],
                                            "--censorship_repr", censorship_repr[0], censorship_repr[1],
                                            "--mean", mean[0], mean[1],
                                            "--std", "1", "1",
                                            "--scale", scale[0], scale[1],
                                            "--shape", shape[0], shape[1],
                                            "--censorship_mean", "0", "0",
                                            "--censorship_temp", "1", "1",
                                            "--censorship_times", censorship_times[0], censorship_times[1], censorship_times[2], censorship_times[3],
                                            ]
                                        )
                            count+=1
subprocess.call(["python", "summarize_results.py"])