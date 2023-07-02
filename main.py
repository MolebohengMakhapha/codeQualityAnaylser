# Code Quality Analyzer
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# Function to analyze code complexity
def analyze_code_complexity(code):
    complexity_score = 0

    for line in code.split("\n"):
        if line.strip().startswith("#"):
            complexity_score += 1

    return complexity_score


# Function to analyze code quality
def analyze_code_quality(code):
    quality_score = 0

    # Analyze code complexity
    complexity_score = analyze_code_complexity(code)
    quality_score += complexity_score

    # Check for common coding mistakes
    if "goto" in code:
        quality_score -= 10
    if "magic_number" in code:
        quality_score -= 5

    return quality_score


# Main program logic
if __name__ == "__main__":
    # Prompt the user to enter code for analysis
    code = input("Enter your code for analysis: ")

    # Analyze the code quality
    score = analyze_code_quality(code)

    # Create a list of quality scores
    scores = [score, 20, 10, 15, 5]

    # Sort the scores in descending order
    sorted_scores = sorted(scores, reverse=True)

    # Display the sorted scores
    print("Sorted Scores:", sorted_scores)

    # Perform data processing and tidying
    data = {
        'Code Sample': ['A', 'B', 'C', 'D', 'E'],
        'Quality Score': sorted_scores
    }
    df = pd.DataFrame(data)

    # Calculate statistics
    mean_score = df['Quality Score'].mean()
    max_score = df['Quality Score'].max()
    min_score = df['Quality Score'].min()

    # Display the statistics
    print("Mean Score:", mean_score)
    print("Max Score:", max_score)
    print("Min Score:", min_score)

    # Extract specific data from the DataFrame
    sample_names = df['Code Sample'].tolist()
    quality_scores = df['Quality Score'].tolist()
    sample_dict = df.to_dict()

    # Manipulate the extracted data
    sample_names_upper = [name.upper() for name in sample_names]
    score_dict = {name: score for name, score in zip(sample_names, quality_scores)}

    # Display the manipulated data
    print("Sample Names (Upper):", sample_names_upper)
    print("Score Dictionary:", score_dict)
    print("DataFrame as Dictionary:", sample_dict)

    # Simulating time series data
    start_date = '2023-01-01'
    end_date = '2023-12-31'
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    scores = np.random.randint(1, 100, size=len(dates))

    # Create a DataFrame for time series data
    df = pd.DataFrame({'Date': dates, 'Quality Score': scores})

    # Set the 'Date' column as the index
    df.set_index('Date', inplace=True)

    # Display the first few rows of the DataFrame
    print(df.head())

    # Perform basic operations on time series data
    print("Minimum Score:", df['Quality Score'].min())
    print("Maximum Score:", df['Quality Score'].max())
    print("Mean Score:", df['Quality Score'].mean())

    # Plot the time series data
    plt.plot(df.index, df['Quality Score'])
    plt.xlabel('Date')
    plt.ylabel('Quality Score')
    plt.title('Code Quality over Time')
    plt.xticks(rotation=45)
    plt.show()

    # Perform set operations
    sample_set = set(sample_names)
    unique_samples = {'F', 'G', 'H'}

    # Union of two sets
    union_set = sample_set.union(unique_samples)
    print("Union Set:", union_set)

    # Intersection of two sets
    intersection_set = sample_set.intersection(unique_samples)
    print("Intersection Set:", intersection_set)

    # Difference between two sets
    difference_set = sample_set.difference(unique_samples)
    print("Difference Set:", difference_set)
