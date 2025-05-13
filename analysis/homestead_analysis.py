import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

if __name__ == "__main__":
    raw = pd.read_csv("/Users/aryanahri/epf_datasets/data/raw_data/us_homestead.csv")
    cleaned = pd.read_csv("/Users/aryanahri/epf_datasets/data/cleaned/cleaned_us_homestead.csv")
    
    raw['date'] = pd.to_datetime(raw.iloc[:, 1])
    raw_sorted = raw.sort_values(by='date', ascending=True)
    
    print(f"Original dataset shape: {raw.shape}")
    print(f"Sorted dataset shape: {raw_sorted.shape}")
    
    print("\nFirst few rows of sorted dataset:")
    print(raw_sorted.head())
    
    Y1 = np.array(raw_sorted['Consumption'])
    Y2 = np.array(cleaned['Consumption'])
    
    # X = np.arange(len(Y1))
    
    # plt.plot(X, Y1, label="raw")
    # plt.plot(X, Y2, label="cleaned")
    # plt.legend()
    # plt.show()
    
    x_start = 0
    x_window = 336
    x_y_gap = 1
    y_window = 24
    
    X = np.arange(len(Y2))
    model_in = Y2[
        x_start: 
        x_start + x_window]
    label = Y2[
        x_start + x_window + x_y_gap : 
        x_start + x_window + y_window + x_y_gap]
    
    test = Y2[17816 + 24 : 17816 + 360 + 24]
    # test_out = Y2[17816 + 336 : 17816 + 336 + 24]
    
    # plt.plot(X[x_start: x_start + x_window], model_in)
    # plt.show()
    # plt.plot(X[x_start + x_window + x_y_gap: x_start + x_window + y_window + x_y_gap],
    #          label)
    # plt.show()
    
    concatenated = np.concatenate([model_in, label])
    
    X = np.arange(
        x_start,
        x_start + x_window + y_window
    )
    
    plt.figure(figsize=(10, 6))
    plt.plot(X[:x_window], concatenated[:x_window], 'b-', label='Input Window')
    plt.plot(X[:x_window], test[:x_window], 'g-', label='Test Input Window')
    
    plt.plot(X[x_window:], concatenated[x_window:], 'r-', label='Target Window')
    plt.plot(X[x_window:], test[x_window:], 'o-', label='Test Target Window')
    
    plt.axvline(x=x_start + x_window, color='k', linestyle='--', label='Prediction Boundary')
    # plt.plot(X, Y2[:x_window + y_window], label="unsliced")
    plt.title('Concatenated Input and Target Windows')
    plt.legend()
    plt.show()