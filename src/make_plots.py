import wandb
from matplotlib import pyplot as plt
import numpy as np

api = wandb.Api()

runs = api.runs("shishckova/tabular_final")
summary_list = {}
for run in list(runs): 
    summary_list[run.name] = run.history()["r2"].values.tolist()

x = 0
for suffix in ["", "_1_trash", "_5_trash", "_20_trash"]:
    suffix2_list = (
        [""] if suffix in ["_1_trash", "_5_trash"] else ["", "_rotated"]
    )
    for suffix2 in suffix2_list:
        for ds in ["wine_quality", "fifa"]:

            scores = {}
            for model in ["boosting",  "resnet", "mlp", "tabnet",]:
                DATASET = f"{ds}{suffix}{suffix2}"
                plot_name = f"{DATASET}_{model}"
                if plot_name not in summary_list:
                    continue
                scores[model] = summary_list[plot_name][-1]
            plt.style.use('ggplot')
            plt.bar(range(len(scores)), list(scores.values()), color=['r', 'y', 'g', 'b'])
            plt.xticks(range(len(scores)), list(scores.keys()))
            if max(scores.values()) > 0.3:
                step = 0.05
            elif max(scores.values()) > 0.25:
                step = 0.02
            else:
                step = 0.01
            plt.yticks(np.arange(0, 1, step))
            plt.ylim([-0, 1.1 * max(scores.values())])

            title = f"{ds}"
            trash = {
                "_1_trash": "1 random feature",
                "_5_trash": "5 random features",
                "_20_trash": "20 random features",
            }
            rotated = {
                "_rotated": "rotated"
            }
            if suffix in trash:
                title += ", " + trash[suffix]
            if suffix2 in rotated:
                title += ", " + rotated[suffix2]


            plt.title(title)
            print(title, scores)
            plt.savefig(f"plots/{x:02d}_" + title.replace(" ", "_") + ".png")
            plt.clf()
            x += 1
