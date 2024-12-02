import pandas as pd
import statsmodels.api as sm
import statsmodels.robust.norms as norms
import matplotlib.pyplot as plt
from scipy.stats import shapiro
import numpy as np

# Load residuals standard deviation CSV
residuals_df = pd.read_csv('all_residuals_std_results_by_group.csv')

# Load original dataset
df = pd.read_csv('BIGBERT.csv')

# Remove 'groupb'
df = df[df['group'] != 'groupb']

# Specify explanatory and dependent variables
explanatory_variable = 'ave_cos_BERT_diary_Human'
dependent_variables = [
    'ave_PANAS_P', 'ave_PANAS_N',
    'ave_competence', 'ave_warmth',
    'ave_willingness', 'ave_understanding'
]

# Add a dummy variable for control
df['group_c'] = pd.get_dummies(df['group'], drop_first=True).astype(int)
control_variables = ['group_c']

# Define colors for each group
group_colors = {
    group: color for group, color in zip(df['group'].unique(), ['blue', 'green', 'orange', 'purple', 'red'])
}

def run_robust_regression_and_plot(dependent_var):
    """Perform robust regression for all groups and plot them on a single graph."""
    plt.figure(figsize=(10, 8))  # Create a new figure for the combined plot

    for group_value in df['group'].unique():
        # Filter data for the specific group
        group_df = df[df['group'] == group_value]

        # Select dependent variable
        y = group_df[dependent_var]

        # Retrieve residuals standard deviation
        residuals_std = residuals_df.loc[
            (residuals_df['dependent_var'] == dependent_var) &
            (residuals_df['group'] == group_value),
            'residuals_std'
        ].values

        if len(residuals_std) == 0:
            print(f"No residuals_std found for dependent variable {dependent_var} in group {group_value}.")
            continue

        residuals_std = residuals_std[0]

        # Set Huber threshold
        delta = 1.345 * residuals_std
        print(f'Using Huber threshold (delta) for {dependent_var} (Group: {group_value}): {delta}')

        # Add constant to explanatory variable
        X = sm.add_constant(group_df[explanatory_variable])

        # Perform robust regression using Huber’s T norm
        model = sm.RLM(y, X, M=norms.HuberT(t=delta)).fit()

        # Display results
        print(f'Robust Regression results for {dependent_var} (Group: {group_value}):')
        print(model.summary())

        # Predict values for regression line
        predictions = model.predict(X)

        # Scatter plot and regression line
        plt.scatter(group_df[explanatory_variable], y, color=group_colors[group_value])
        plt.plot(group_df[explanatory_variable], predictions, color=group_colors[group_value])

    # 軸のフォントサイズの設定
    font_size = 20  # 任意のフォントサイズ
    plt.xticks(fontsize=font_size)
    plt.yticks(fontsize=font_size)

    # Add labels, legend, and title
    #plt.title(f'Regression: {dependent_var} ~ {explanatory_variable}')
    #plt.xlabel('Human-Diary')
    #plt.ylabel(dependent_var)
    # 軸と範囲の設定
    plt.xlim(0.3, 0.8)  # X軸の範囲
    plt.yticks(np.arange(1.0, 7.0, 1.0))
    plt.ylim(0.9, 6.1)  # Y軸の範囲
    plt.grid(False)

    # Save combined plot as SVG
    plt.savefig(f"SVGs/{dependent_var}_combined_regression_plot.svg", format="svg")
    plt.close()  # Close the plot to free memory

# Perform robust regression and create combined plots for each dependent variable
for dependent_var in dependent_variables:
    run_robust_regression_and_plot(dependent_var)
