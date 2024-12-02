import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import shapiro

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

# Load the CSV file
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

# List to store results
results_list = []

# Loop through each unique value in the 'group' column
for group_value in df['group'].unique():
    # Extract data for the specific group
    group_df = df[df['group'] == group_value]

    # Add a constant term to the explanatory variable for regression analysis
    X = sm.add_constant(group_df[explanatory_variable])

    # Perform regression analysis for each dependent variable
    for dependent_var in dependent_variables:
        # Select the dependent variable
        y = group_df[dependent_var]

        # Perform OLS regression
        model = sm.OLS(y, X).fit()

        # Display regression results
        print(f'Regression results for {dependent_var} (Group: {group_value}):')
        print(model.summary())

        # Predict values for the regression line
        predictions = model.predict(X)

        # Calculate residuals
        residuals = model.resid
        residuals_std = np.std(residuals)  # Standard deviation of residuals

        # Perform Shapiro-Wilk test
        shapiro_test_stat, shapiro_p_value = shapiro(residuals)

        # Output test results
        print(f'Shapiro-Wilk Test for {dependent_var} (Group: {group_value}):')
        print(f'Statistic: {shapiro_test_stat}, p-value: {shapiro_p_value}\n')

        # Add standard deviation of residuals to the results list
        results_list.append({
            'group': group_value,
            'dependent_var': dependent_var,
            f'residuals_std_{group_value}': residuals_std
        })

        # Create a plot
        plt.figure(figsize=(8, 6))
        plt.scatter(group_df[explanatory_variable], y)
        plt.plot(group_df[explanatory_variable], predictions, color='red')

        # Set titles and labels
        plt.title(f'Regression: {dependent_var} ~ {explanatory_variable} (Group: {group_value})')
        plt.xlabel('Human-Diary')
        plt.ylabel(dependent_var)

        # Set axes limits
        plt.xlim(0.3, 0.8)  # X-axis range
        plt.yticks(np.arange(1.0, 6.0, 1.0))
        plt.ylim(0.9, 5.1)  # Y-axis range
        plt.grid(False)

        # Save the plot as an SVG file
        # plt.savefig(f"SVGs/{dependent_var}_regression_plot_group_{group_value}.svg", format="svg")
        plt.close()  # Close the plot to free memory

# Convert the results list to a DataFrame
results_df = pd.DataFrame(results_list)

# Save the standard deviation of residuals to a CSV file
results_df.to_csv('all_residuals_std_results_by_group_after_3SD.csv', index=False)
