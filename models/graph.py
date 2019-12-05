import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

INFILE = './my_metrics/test-accuracy.csv'
TITLE = 'Testing Accuracy for Baseline'
X_LABEL = 'round'
Y_LABEL = 'accuracy'

if __name__ == '__main__':

    sns.set(style='darkgrid')

    data = pd.read_csv(INFILE, header=0)

    # Plot the responses for different events and regions
    plt.figure()
    ax = sns.lineplot(x=X_LABEL, y=Y_LABEL, data=data).set_title(TITLE)
    plt.show()
