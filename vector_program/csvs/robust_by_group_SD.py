import pandas as pd
import statsmodels.api as sm
import statsmodels.robust.norms as norms
import matplotlib.pyplot as plt
from scipy.stats import shapiro
import numpy as np

# Function to remove outliers using 3SD
def remove_outliers(df, columns):
    """Remove outliers beyond 3 standard deviations for the specified columns"""
    for col in columns:
        mean = df[col].mean()
        std = df[col].std()
        lower_bound = mean - 3 * std
        upper_bound = mean + 3 * std
        df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
    return df

# Load the residuals standard deviation CSV file
residuals_df = pd.read_csv('all_residuals_std_results_by_group_after_3SD.csv')

# Load the original dataset
df = pd.read_csv('BIGBERT.csv')

# Remove group 'groupb'
df = df[df['group'] != 'groupb']

# Specify explanatory and dependent variables
explanatory_variable = 'ave_cos_BERT_diary_Human'  # Explanatory variable
dependent_variables = [
    'ave_PANAS_P', 'ave_PANAS_N',
    'ave_competence', 'ave_warmth',
    'ave_willingness', 'ave_understanding'
]  # Multiple dependent variables

# Apply 3SD outlier removal to all variables
all_variables = [explanatory_variable] + dependent_variables
df = remove_outliers(df, all_variables)

# Add a dummy variable for control
df['group_c'] = pd.get_dummies(df['group'], drop_first=True).astype(int)
control_variables = ['group_c']  # List of control variables

def run_robust_regression_by_group(dependent_var, group_value, control_vars=[]):
    """Perform robust regression for a given dependent variable and group"""

    # Filter data for the specific group
    group_df = df[df['group'] == group_value]

    # Select the dependent variable
    y = group_df[dependent_var]

    # Retrieve the residuals standard deviation
    residuals_std = residuals_df.loc[
        (residuals_df['dependent_var'] == dependent_var) &
        (residuals_df['group'] == group_value), 
        'residuals_std'
    ].values

    if len(residuals_std) == 0:
        print(f"No residuals_std found for dependent variable {dependent_var} in group {group_value}.")
        return

    residuals_std = residuals_std[0]  # Extract the single value

    # Set the Huber threshold
    delta = 1.345 * residuals_std
    print(f'Using Huber threshold (delta) for dependent variable {dependent_var} (Group: {group_value}): {delta}')

    # Add a constant term to the explanatory variable
    X = sm.add_constant(group_df[explanatory_variable])

    # Perform robust regression using Huber's T norm
    model = sm.RLM(y, X, M=norms.HuberT(t=delta)).fit()

    # Display the robust regression results
    print(f'Robust Regression results for {dependent_var} (Group: {group_value}):')
    print(model.summary())

    # Calculate residuals
    residuals = model.resid

    # Perform Shapiro-Wilk test
    shapiro_test_stat, shapiro_p_value = shapiro(residuals)

    # Output the test results
    print(f'Shapiro-Wilk Test for {dependent_var} (Group: {group_value}):')
    print(f'Statistic: {shapiro_test_stat}, p-value: {shapiro_p_value}\n')

    # Create a plot
    plt.figure(figsize=(8, 6))
    plt.scatter(group_df[explanatory_variable], y)
    plt.plot(group_df[explanatory_variable], model.predict(X), color='red')

    # Set font sizes for axes
    font_size = 20  # Arbitrary font size
    plt.xticks(fontsize=font_size)
    plt.yticks(fontsize=font_size)

    # Set axis limits
    plt.xlim(0.3, 0.8)  # X-axis range
    plt.yticks(np.arange(1.0, 6.0, 1.0))
    plt.ylim(0.9, 5.1)  # Y-axis range
    plt.grid(False)

    # Save the plot as an SVG file
    #plt.savefig(f"SVGs/{dependent_var}_regression_plot_group_{group_value}.svg", format="svg")
    plt.close()  # Close the plot to free memory

# Perform robust regression for each group and dependent variable
for group_value in df['group'].unique():
    for dependent_var in dependent_variables:
        run_robust_regression_by_group(dependent_var, group_value, control_vars=control_variables)
