import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.stats import skew

def get_data(path: str) -> pd.DataFrame:
  data = pd.read_csv(path)

  return data

def plot_wages(wages: pd.Series, exp_wages: pd.Series) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
   # plot for log wage
    axes[0].hist(wages, bins=20, density=True, alpha=0.2, color='blue')
    sns.kdeplot(wages, bw_adjust=0.5, ax=axes[0])
    axes[0].set_title('Weekly Wage (log)')
    axes[0].grid(True)
    axes[0].set_xlabel('Log, $')

    # plot for exp wage
    axes[1].hist(exp_wages, bins=20, density=True, alpha=0.2, color='blue')
    sns.kdeplot(exp_wages, bw_adjust=0.5, ax=axes[1])
    axes[1].set_title('Weekly wage (exponential)')
    axes[1].grid(True)
    axes[1].set_xlabel('Exponential, $')
    
    plt.savefig("wages_density.png")

    plt.clf()

def plot_educ_means(data: list[float], reg: bool = False):
    "If reg attr is True, will also plot linear regression in addition to means"
    x = np.arange(len(data))
    
    plt.plot(x, data, color="black", marker="o", label="Means")
    
    coef, intercept = np.polyfit(x, data, 1)
    regression_line = coef * x + intercept

    print(f'Slope - {coef}, Intercept - {intercept}')

    if reg:
        plt.plot(x, regression_line, color="red", alpha=0.7, label=f"Regression line (Coef - {round(coef, 3)})")
    
    plt.grid(True)
    plt.title("Conditional sample means with linear regression")
    plt.legend()

    plt.ylabel("Log weekly earnings, $")
    plt.xlabel("Years of education")

    fign = "educ_means_lr.png" if reg else "educ_means.png"

    plt.savefig(fign)

def get_statistics(data: pd.Series):
    mean_wage = data.mean()
    median_wage = data.median()
    skewness_wage = skew(data)

    return mean_wage, median_wage, skewness_wage

def get_mean_for_education(data: pd.DataFrame, education: int) -> float:
    """
    Returns mean value of lwklywge column for given education year
    """
    filtered_data = data[data["educ"] == education]

    return filtered_data["lwklywge"].mean()

def main():
    data = get_data(Path(__file__).parent / "Assig1.csv")

    print(data.head())

    exp_wages = np.exp(data["lwklywge"])
    data["wage"] = exp_wages

    #Answers to 1.a and 1.b (wage column is exp., lwklywge is log)
    plot_wages(data["lwklywge"], data["wage"])

    wage_attrs = ["wage", "lwklywge"]
    for wage_attr in wage_attrs:
            mean_wage, median_wage, skewness_wage = get_statistics(data=data[wage_attr])

            print(f"Sample Mean of Wage ({wage_attr}): {mean_wage:.2f}")
            print(f"Sample Median of Wage ({wage_attr}): {median_wage:.2f}")
            print(f"Coefficient of Skewness of Wage ({wage_attr}): {skewness_wage:.2f}")

    # Answer to 1.c
    data_16 = data[data["educ"] == 16]
    mean_wage_16 = data_16["wage"].mean()
    print(f"Sample Mean of Wage for people with 16 years of education: {mean_wage_16: .2f}")

    # Answer to 1.d
    unique_education_y = data["educ"].unique()
    
    education_means = []
    for education in sorted(unique_education_y):
       education_means.append(get_mean_for_education(data=data, education=education))

    plot_educ_means(data=education_means, reg=True)



if __name__ == "__main__":
  main()
