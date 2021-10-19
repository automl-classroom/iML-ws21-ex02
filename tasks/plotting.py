from utils.styled_plot import plt


def plot_bar(x, y, title=None, ylabel=None):
    """
    Displays a bar plot on given x and y.
    
    Inputs:
        x (list or np.ndarray): Labels. Each entry is associated to one bar.
        y (list or np.ndarray): Values for x. Same shape as x.
        title (str): Title.
        ylabel (str): Label for y-axis.
        
    Returns:
        plt (matplotlib.pyplot or utils.styled_plot.plt)
    """
    
    plt.figure()
    
    return plt


if __name__ == "__main__":
    plt = plot_bar(
        x=["A", "B", "C"],
        y=[0.2, 0.5, 0.4],
        title="Test",
        ylabel="Test"
    )
    

    

    