########################################################################################################################
# Author:      Antoni Adamczyk
# MatNr:       12306508
# Description: ... short description of the file ...
# Comments:    ... comments for the tutors ...
#              ... can be multiline ...
########################################################################################################################

import numpy as np
import pandas as pd
from pathlib import Path
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns

#------------------------------------------------- Provided Functions -------------------------------------------------#

#------------------------------------------------------- Tasks --------------------------------------------------------#
# IMPORTANT: For the whole assignment, do NOT change any function names or signatures and only write code in
# the highlighted TODO-blocks. Otherwise, we can not test your submissions automatically and there is a chance you
# will receive zero points.



def task_hypothesis_threshold(
    n: int,
    mu_0: float,
    sigma:float,
    alpha: float
) -> float:
    """
    Args:
        n (int): The number of fields.
        mu_0 (float): The yield level in tons per hectare.
        sigma (float): The population standard deviation.
        alpha (float): The significance level.

    Returns:
        threshold (float): The calculate threshold the farmers' should set.
    """
    threshold = 0.0
    # TODO START

    z_score = stats.norm.ppf(1 - alpha)
    threshold = mu_0 + z_score * (sigma / np.sqrt(n))

    # TODO END
    print(f"b)  Threshold = {threshold:.3f}")
    return threshold

def task_hypothesis_threshold_plot(
    n: int, 
    mu_0: float, 
    sigma: float, 
    alpha: float,
    threshold: float
) -> None:
    """
    Args:
        n (int): The number of fields.
        mu_0 (float): The yield level in tons per hectare.
        sigma (float): The population standard deviation.
        alpha (float): The significance level.
        threshold (float): The calculate threshold for X_bar_n from task_hypothesis_threshold.

    Returns:
        None
    """
    # TODO START
    x = np.linspace(mu_0 - 4 * sigma, mu_0 + 4 * sigma, 1000)
    y = stats.norm.pdf(x, mu_0, sigma)

    x_increase_area = np.linspace(threshold, mu_0 + 3 * sigma, 1000)
    y_increase_area = stats.norm.pdf(x_increase_area, mu_0, sigma)

    x_no_increase_area = np.linspace(mu_0 - 3 * sigma, threshold, 1000)
    y_no_increase_area = stats.norm.pdf(x_no_increase_area, mu_0, sigma)

    plt.fill_between(x_increase_area, y_increase_area, color='lightgreen', alpha=0.5, label='Increase')

    plt.fill_between(x_no_increase_area, y_no_increase_area, color='lightcoral', alpha=0.5, label='No increase')

    plt.plot(x, y)
    plt.axvline(threshold, color='green', linestyle='dashed', label='Yield Threshold')
    plt.axvline(mu_0, color='red', linestyle='dashed', label='Yield Level')
    plt.title(f'Critical region for increased yield level (mu_0={mu_0}, sigma={sigma:.3f}, alpha={alpha})', fontsize=9)
    plt.xlabel('Crop Yield (in tons/hectare)')
    plt.ylabel('P(Crop Yield)')
    plt.legend()
    plt.grid(True)
    plt.show()
    # TODO END
    pass

# Power analysis function
def task_power(
    n: int, 
    mu_0: float, 
    sigma: float, 
    threshold: float,
    mu0_multiplier: float
) -> float:
    """
    Args:
        n (int): The number of fields.
        mu_0 (float): The yield level in tons per hectare.
        sigma (float): The population standard deviation.
        threshold (float): The calculate threshold for X_bar_n from task_hypothesis_threshold.
        mu0_multiplier (float): The average yield increase (1.15 for task c).

    Returns:
        power (float): The calculated power of the test.
    """
    # Your code here
    power = 0.0
    # TODO START
    mu_1 = mu0_multiplier*mu_0

    effect_size = (mu_1 - threshold) / (sigma / np.sqrt(n))
    power = 1 - stats.norm.cdf(-effect_size)


    # TODO END
    print(f"c1) Power = {power:.3f}")
    return power

def task_nr_of_fields(
        mu_0: float,
        sigma: float,
        alpha: float,
        target_power=0.90
) -> int:
    """
    Args:
        mu_0 (float): The yield level in tons per hectare.
        sigma (float): The population standard deviation.
        alpha (float): The significance level.
        target_power (float): The target power that has to be achieved.

    Returns:
        number_of_fields (int): The calculated needed number of fields to archive the specified power.
    """
    number_of_fields = 0
    # TODO START

    mu_1 = mu_0*1.15
    power = 0.0
    z_score = stats.norm.ppf(1 - alpha)
    

    while power < target_power:
        number_of_fields = number_of_fields+1
        threshold = mu_0 + z_score * (sigma / np.sqrt(number_of_fields))
        effect_size = (mu_1 - threshold) / (sigma / np.sqrt(number_of_fields))
        power = 1 - stats.norm.cdf(-effect_size)




    # TODO END
    print(f"c2) Fields needed for power {target_power} = {number_of_fields}")
    return number_of_fields

def task_yield_for_power(
    n: int,
    mu_0: float,
    sigma: float,
    threshold: float,
    target_power: float
) -> float:
    """
    Args:
        n (int): The number of fields.
        mu_0 (float): The yield level in tons per hectare
        sigma (float): The population standard deviation.
        threshold (float): The calculate threshold for X_bar_n from task_hypothesis_threshold
        target_power (float): The target power that has to be achieved.
    
    Returns:
        target_yield (float): The calculated yield where the test achieves the power of 0.99.
    """
    range = [0.9, 1.5]
    range_interval = np.linspace(range[0], range[1], 1000)
    x_axis = mu_0 * range_interval  # Example of specifying the x-axis for the plot (you can do otherwise if you want)
    target_yield = 0.0
    # TODO START
    
    powers = [(1 - stats.norm.cdf(-(mu_0*i - threshold) / (sigma / np.sqrt(n)))) for i in range_interval]
    for i in range_interval:
        if (1 - stats.norm.cdf(-(mu_0*i - threshold) / (sigma / np.sqrt(n)))) >= 0.99:
            target_yield = mu_0*i
            break


    plt.plot(x_axis, powers)
    plt.xlim([min(x_axis), max(x_axis)])
    plt.axvline(target_yield, color='red', linestyle='dashed', label='Target yield')
    
    plt.fill_between(x_axis, powers, where=(x_axis <= target_yield), color='red', alpha=0.3, label='Power < 0.99')
    plt.fill_between(x_axis, powers, where=(x_axis > target_yield), color='green', alpha=0.3, label='Power ≥ 0.99')

    plt.title(f'Power with change in yield (mu_0: {mu_0}, threshold: {threshold:.2f})')
    plt.xlabel('Crop yield (in tons/hectare)')
    plt.ylabel('P(Crop yield)')
    plt.legend()
    plt.grid(True)
    plt.show()

    # TODO END 
    print(f"d)  Yield for power of 0.99 = {target_yield:.3f}")
    return target_yield


#------------------------------------------------ Permutation Testing -------------------------------------------------#

# TODO: START -> implement your own functions for generating the plots for the permutation test tasks
def task_ecdf_plot(data, name1: str, name2: str):
    ages1 = data[data['studies'] == name1]['age'].tolist()
    ages1_sorted = np.sort(ages1)

    ages2 = data[data['studies'] == name2]['age'].tolist()
    ages2_sorted = np.sort(ages2)

    ecdf_values1 = np.arange(1, len(ages1_sorted) + 1) / len(ages1_sorted)
    ecdf_values2 = np.arange(1, len(ages2_sorted) + 1) / len(ages2_sorted)

    plt.step(ages1_sorted, ecdf_values1, label = name1, color = "darkblue")
    plt.step(ages2_sorted, ecdf_values2, label = name2, color = "green")
    plt.xlabel('Age')
    plt.ylabel('ECDF')
    plt.title(f'ECDF for {name1} and {name2}')
    plt.legend()
    plt.grid(True)
    plt.show()
# TODO END



def task_permutation_test(
    x: list[int], 
    y: list[int], 
    n_permutations: int, 
    statistic: callable
):
    """
    Perform a permutation test for the given measurements lists and the given statistic function.
    
    Args:
        x (list[int]): The first given measurements list.
        y (list[int]): The second given measurements list.
        n_permutations (int): The number of permutations to perform.
        statistic (function): The function to calculate the test statistic.
        
    Returns:
        p_value (float): The calculated p-value.
        permuted_diffs (list[float]): The list of calculated absolute differences for each permutation.
        
    """
    p_value = 0.0
    permuted_diffs = []
    # TODO START

    observed_diff = np.abs(statistic(x)-statistic(y))

    all_list = x + y
    count = 0

    for i in range(n_permutations):
        np.random.shuffle(all_list)
        permuted_x = all_list[:len(x)]
        permuted_y = all_list[len(x):]

        permuted_diff = np.abs(statistic(permuted_x)-statistic(permuted_y))
        permuted_diffs.append(permuted_diff)
        if permuted_diff >= observed_diff:
            count += 1

    p_value = (count + 1) / (n_permutations + 1)

    data = np.sort(permuted_diffs)
   

    sns.kdeplot(x=data, label="cos")
    plt.axvline(observed_diff, color='red', linestyle='dashed', label="Observed difference")
    plt.xlabel('Absolute means\' differences')
    plt.ylabel('KDE')
    plt.title("KDE plot for permuted mean differences")
    plt.legend()
    plt.grid(True)
    plt.show()


    # TODO END
    print(f"b)  p-value = {p_value:.3f}")
    return p_value, permuted_diffs


def task_bonferroni_correction(
    m: int, 
    alpha: float
):
    """
    Perform bonferroni correction for m generated datasets and counts how many times the null hypothesis is rejected
    based on the calculated significance level.

    Args:
        m (int): The number of hypothesis tests to conduct.
        alpha (float): The original significance level of the hypothesis.

    Returns:
        reject_count (int): The number of rejected hypothesis tests.
        significance_level (float): The significance level used for the bonferroni correction.

    """
    reject_count, significance_level = 0, 0.0
    # TODO START
    significance_level = alpha/m

    for i in range(m):
        informatics_sample = np.random.uniform(low=22, high=35, size=1000)
        ice_sample = np.random.uniform(low=21, high=35, size=1000)

        informatics_sample = informatics_sample.tolist()
        ice_sample = ice_sample.tolist()

        p_value, diffs = task_permutation_test(informatics_sample, ice_sample, n_permutations, np.mean)

        if p_value < significance_level:
            reject_count += 1


    # TODO END
    print(f"c)  Rejected {reject_count} times out of {m} with significance level {significance_level}")
    return reject_count, significance_level

#------------------------------------------------------ Testing -------------------------------------------------------#
# Here, you can test your implementations. We already provided some example function calls.
# Make sure that you additionally experiment with different function parameters/dataset variations/... to ensure the
# functionality of your implementation.

# --------- General ---------#
YOUR_MTNR = 12306508 # TODO Replace with your matriculation number
np.random.seed(YOUR_MTNR)

# --------- Hypothesis Testing ---------#
print("Hypothesis Testing".center(80, "-"))
# TODO Specify the correct values, described in the task
n = 8
mu_0 = 40.0
alpha = 0.05
sigma = np.random.normal(9,1)
mu0_multiplier = 1.15

# b)
# Example function calls for task b
th = task_hypothesis_threshold(n, mu_0, sigma, alpha)
task_hypothesis_threshold_plot(n, mu_0, sigma, alpha, th)

# c)
# TODO Call task_power function with correct parameters
task_power(n, mu_0, sigma, th, mu0_multiplier)
# TODO Call task_nr_of_fields function with correct parameters
task_nr_of_fields(mu_0, sigma, alpha, target_power = 0.9)

# d)
# TODO Call task_yield_for_power function with correct parameters
task_yield_for_power(n, mu_0, sigma, th, target_power=0.99)

# --------- Permutation Test ---------#
print("Permutation Test".center(80, "-"))

FILE_PATH = Path.cwd() / Path("studies.csv")  # Replace with your Path if necessary
studies = pd.read_csv(FILE_PATH)

# a)
#create your own functions for your plots in the desired order
task_ecdf_plot(studies, "informatics", "ice")
# b)
n_permutations = 10000
informatics = studies[studies['studies'] == 'informatics']['age'].tolist() #TODO: Replace with the correct values (should be a list of ages)
ice = studies[studies['studies'] == 'ice']['age'].tolist() #TODO: Replace with the correct values (should be a list of ages)
task_permutation_test(informatics, ice, n_permutations, np.mean)

# c)
m = 100
alpha = 0.1
task_bonferroni_correction(m, alpha)




