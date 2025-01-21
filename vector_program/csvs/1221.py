import pandas as pd
import statsmodels.api as sm
import statsmodels.robust.norms as norms
import matplotlib.pyplot as plt
import numpy as np

# Load residuals standard deviation CSV
residuals_df = pd.read_csv('residuals_std_results_by_group.csv')

# Load original dataset
df = pd.read_csv('BIGBERT3rd.csv')

# Specify explanatory and dependent variables
explanatory_variable = 'week1_diary_Human'
dependent_variables = [
    'week1_PANAS_P', 'week1_PANAS_N',
    'week1_competence', 'week1_warmth',
    'week1_willingness', 'week1_understanding'
]

# Define Y-axis limits for each dependent variable
y_limits = {
    'week1_PANAS_P': (0.9, 6.1),
    'week1_PANAS_N': (0.9, 6.1),
    'week1_competence': (0.9, 5.1),
    'week1_warmth': (0.9, 5.1),
    'week1_willingness': (0.9, 5.1),
    'week1_understanding': (0.9, 5.1)
}

# Define colors for each week1_group
week1_group_colors = {
    week1_group: color for week1_group, color in zip(df['week1_group'].unique(), ['blue', 'green', 'orange', 'purple', 'red'])
}

def run_robust_regression_and_plot_all():
    """Perform robust regression and create subplots for all dependent variables."""
    plt.figure(figsize=(22, 12))  # Set figure size for all subplots

    for i, dependent_var in enumerate(dependent_variables, start=1):
        plt.subplot(2, 3, i)  # Create a 2x3 grid of subplots

        for week1_group_value in df['week1_group'].unique():
            # Filter data for the specific week1_group
            week1_group_df = df[df['week1_group'] == week1_group_value]

            # Select dependent variable
            y = week1_group_df[dependent_var]

            # Retrieve residuals standard deviation
            residuals_std = residuals_df.loc[
                (residuals_df['dependent_var'] == dependent_var) &
                (residuals_df['week1_group'] == week1_group_value),
                'residuals_std'
            ].values

            if len(residuals_std) == 0:
                print(f"No residuals_std found for dependent variable {dependent_var} in week1_group {week1_group_value}.")
                continue

            residuals_std = residuals_std[0]

            # Set Huber threshold
            delta = 1.345 * residuals_std
            print(f'Using Huber threshold (delta) for {dependent_var} (week1_group: {week1_group_value}): {delta}')

            # Add constant to explanatory variable
            X = sm.add_constant(week1_group_df[explanatory_variable])

            # Perform robust regression using Huber’s T norm
            model = sm.RLM(y, X, M=norms.HuberT(t=delta)).fit()

            # Predict values for regression line
            predictions = model.predict(X)

            print(model.summary())

            # Scatter plot and regression line
            plt.scatter(week1_group_df[explanatory_variable], y, color=week1_group_colors[week1_group_value], label=f"{week1_group_value}")
            # plt.plot(week1_group_df[explanatory_variable], predictions, color=week1_group_colors[week1_group_value])

        # Set titles and labels for the subplot
        plt.title(dependent_var, fontsize=14)
        # plt.xlabel("Human-Diary", fontsize=12)
        # plt.ylabel(dependent_var, fontsize=12)

        # Set Y-axis range and ticks
        y_min, y_max = y_limits[dependent_var]
        plt.ylim(y_min, y_max)
        # plt.yticks(np.arange(y_min, y_max + 1.0, 1.0))  # Common tick interval of 1.0
        plt.yticks(range(int(np.ceil(y_min)), int(np.floor(y_max)) + 1, 1))

        plt.xlim(0.0, 1.0)  # X-axis range
        plt.grid(False)

    # Add a global legend and adjust layout
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    # plt.legend(by_label.values(), by_label.keys(), loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=5)
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust layout to fit titles and legends
    # plt.suptitle("Robust Regression by Dependent Variables", fontsize=20)  # Add a global title
    plt.savefig("combined_regression_plots.svg", format="svg")  # Sweek1 as SVG
    plt.show()  # Display the plot

# Execute the combined plotting function
run_robust_regression_and_plot_all()
