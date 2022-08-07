import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

def draw_fig(test_case_list, time_list, op_kind_list, filename):
    d = {'testcase': test_case_list, 'time': time_list, 'op_kind': op_kind_list}
    df = pd.DataFrame(data=d)

    sns.set(style="ticks")

    g = sns.catplot(x="op_kind", y="time", col="testcase", kind="bar", col_wrap=4, data=df)

    for ax in g.axes.ravel():
        for c in ax.containers:
            labels = [f'{(v.get_height()):.3f}ms' for v in c]
            ax.bar_label(c, labels=labels, label_type='edge')
        ax.margins(y=0.2)

    plt.savefig("../../generate/"+filename+".png")
