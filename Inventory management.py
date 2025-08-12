# Databricks notebook source
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Generate sample data
np.random.seed(0)
products = ['Product A', 'Product B', 'Product C']
demand = np.random.randint(100, 500, size=(len(products), 12))  # 12 months of demand data
costs = np.random.uniform(10, 50, size=len(products))  # costs per unit
holding_costs = costs * 0.2  # 20% of the unit cost
ordering_costs = np.random.uniform(50, 100, size=len(products))  # ordering costs per order

# Create a DataFrame to store the data
df = pd.DataFrame({
    'Product': products,
    'Demand': demand.sum(axis=1),
    'Cost': costs,
    'Holding Cost': holding_costs,
    'Ordering Cost': ordering_costs
})

# Calculate the Economic Order Quantity (EOQ)
df['EOQ'] = np.sqrt((2 * df['Demand'] * df['Ordering Cost']) / df['Holding Cost'])

# Calculate the total cost
df['Total Cost'] = df['Demand'] * df['Cost'] + np.sqrt(2 * df['Demand'] * df['Ordering Cost'] * df['Holding Cost'])

print(df)

# Visualize the results
plt.figure(figsize=(10, 6))
plt.bar(df['Product'], df['EOQ'])
plt.xlabel('Product')
plt.ylabel('EOQ')
plt.title('Economic Order Quantity (EOQ)')
plt.show()


# COMMAND ----------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Generate sample data
np.random.seed(0)
products = ['Product A', 'Product B', 'Product C']
demand = np.random.randint(100, 500, size=(len(products), 12))  # 12 months of demand data
costs = np.random.uniform(10, 50, size=len(products))  # costs per unit
holding_costs = costs * 0.2  # 20% of the unit cost
ordering_costs = np.random.uniform(50, 100, size=len(products))  # ordering costs per order

# Create a DataFrame to store the data
df = pd.DataFrame({
    'Product': products,
    'Demand': demand.sum(axis=1),
    'Cost': costs,
    'Holding Cost': holding_costs,
    'Ordering Cost': ordering_costs
})

# Calculate the Economic Order Quantity (EOQ)
df['EOQ'] = np.sqrt((2 * df['Demand'] * df['Ordering Cost']) / df['Holding Cost'])

# Sensitivity analysis
demand_multiplier = np.linspace(0.5, 1.5, 10)
eoq_sensitivity = np.zeros((len(products), len(demand_multiplier)))
total_cost_sensitivity = np.zeros((len(products), len(demand_multiplier)))

for i, multiplier in enumerate(demand_multiplier):
    new_demand = df['Demand'] * multiplier
    eoq = np.sqrt((2 * new_demand * df['Ordering Cost']) / df['Holding Cost'])
    total_cost = new_demand * df['Cost'] + np.sqrt(2 * new_demand * df['Ordering Cost'] * df['Holding Cost'])
    eoq_sensitivity[:, i] = eoq
    total_cost_sensitivity[:, i] = total_cost

# Visualize the results
plt.figure(figsize=(10, 6))
for i, product in enumerate(products):
    plt.plot(demand_multiplier, eoq_sensitivity[i], label=product)
plt.xlabel('Demand Multiplier')
plt.ylabel('EOQ')
plt.title('EOQ Sensitivity to Demand')
plt.legend()
plt.show()

plt.figure(figsize=(10, 6))
for i, product in enumerate(products):
    plt.plot(demand_multiplier, total_cost_sensitivity[i], label=product)
plt.xlabel('Demand Multiplier')
plt.ylabel('Total Cost')
plt.title('Total Cost Sensitivity to Demand')
plt.legend()
plt.show()


# COMMAND ----------

import numpy as np
import matplotlib.pyplot as plt

# Parameters
T = 12  # planning horizon (months)
demand = np.random.randint(100, 500, size=T)  # demand for each month
holding_cost = 5  # holding cost per unit per month
ordering_cost = 100  # ordering cost per order
initial_inventory = 0  # initial inventory level

# Dynamic programming
dp = np.zeros((T + 1, max(demand) + 1))  # dp[t, i] = minimum cost for months 0 to t with inventory level i
for t in range(T):
    for i in range(max(demand) + 1):
        if t == 0:
            dp[t + 1, i] = ordering_cost + holding_cost * i
        else:
            min_cost = float('inf')
            for j in range(max(demand) + 1):
                cost = dp[t, j] + ordering_cost * (j < i) + holding_cost * max(0, i - demand[t])
                if cost < min_cost:
                    min_cost = cost
            dp[t + 1, i] = min_cost

# Optimal inventory levels
optimal_inventory = np.zeros(T + 1)
optimal_inventory[T] = np.argmin(dp[T])
for t in range(T - 1, -1, -1):
    min_cost = float('inf')
    for i in range(max(demand) + 1):
        cost = dp[t, i] + ordering_cost * (i < optimal_inventory[t + 1]) + holding_cost * max(0, optimal_inventory[t + 1] - demand[t])
        if cost < min_cost:
            min_cost = cost
            optimal_inventory[t] = i

# Visualize the results
plt.figure(figsize=(10, 6))
plt.plot(optimal_inventory[:-1], label='Optimal Inventory Level')
plt.plot(demand, label='Demand')
plt.xlabel('Month')
plt.ylabel('Units')
plt.title('Optimal Inventory Levels')
plt.legend()
plt.show()

# COMMAND ----------

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Generate sample data
np.random.seed(0)
demand = np.random.randint(100, 500, size=24)  # 2 years of demand data
months = np.arange(24)

# Create a DataFrame
df = pd.DataFrame({'Month': months, 'Demand': demand})

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df[['Month']], df['Demand'], test_size=0.2, random_state=42)

# Train a linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse:.2f}')

# Use the model to predict demand for the next 12 months
next_months = np.arange(24, 36).reshape(-1, 1)
predicted_demand = model.predict(next_months)
print(predicted_demand)


# COMMAND ----------

import numpy as np
import matplotlib.pyplot as plt

# Parameters
T = 12  # planning horizon (months)
demand_mean = 200  # mean demand
demand_std = 50  # standard deviation of demand
holding_cost = 5  # holding cost per unit per month
ordering_cost = 100  # ordering cost per order
initial_inventory = 0  # initial inventory level

# Simulate demand
np.random.seed(0)
demand = np.random.normal(demand_mean, demand_std, size=T)

# Dynamic inventory model
inventory = np.zeros(T + 1)
inventory[0] = initial_inventory
for t in range(T):
    if inventory[t] < demand[t]:
        inventory[t + 1] = demand[t] + 10  # order up to demand + 10 units
    else:
        inventory[t + 1] = inventory[t] - demand[t]

# Calculate costs
holding_costs = np.zeros(T)
ordering_costs = np.zeros(T)
for t in range(T):
    holding_costs[t] = holding_cost * max(0, inventory[t] - demand[t])
    ordering_costs[t] = ordering_cost * (inventory[t] < demand[t])

# Visualize the results
plt.figure(figsize=(10, 6))
plt.plot(inventory[:-1], label='Inventory Level')
plt.plot(demand, label='Demand')
plt.xlabel('Month')
plt.ylabel('Units')
plt.title('Inventory Level and Demand')
plt.legend()
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(holding_costs, label='Holding Costs')
plt.plot(ordering_costs, label='Ordering Costs')
plt.xlabel('Month')
plt.ylabel('Costs')
plt.title('Holding and Ordering Costs')
plt.legend()
plt.show()


# COMMAND ----------

import numpy as np
import matplotlib.pyplot as plt

# Parameters
T = 12  # planning horizon (months)
demand_mean = 200  # mean demand
demand_std = 50  # standard deviation of demand
holding_cost = 5  # holding cost per unit per month
ordering_cost = 100  # ordering cost per order
initial_inventory = 0  # initial inventory level
num_simulations = 1000  # number of simulations

# Monte Carlo simulations
np.random.seed(0)
total_costs = np.zeros(num_simulations)
for i in range(num_simulations):
    demand = np.random.normal(demand_mean, demand_std, size=T)
    inventory = np.zeros(T + 1)
    inventory[0] = initial_inventory
    total_cost = 0
    for t in range(T):
        if inventory[t] < demand[t]:
            inventory[t + 1] = demand[t] + 10  # order up to demand + 10 units
            total_cost += ordering_cost
        else:
            inventory[t + 1] = inventory[t] - demand[t]
        total_cost += holding_cost * max(0, inventory[t] - demand[t])
    total_costs[i] = total_cost

# Calculate statistics
mean_total_cost = np.mean(total_costs)
std_total_cost = np.std(total_costs)
print(f'Mean Total Cost: {mean_total_cost:.2f}')
print(f'Standard Deviation of Total Cost: {std_total_cost:.2f}')

# Visualize the results
plt.figure(figsize=(10, 6))
plt.hist(total_costs, bins=50, alpha=0.7, color='blue', edgecolor='black')
plt.xlabel('Total Cost')
plt.ylabel('Frequency')
plt.title('Histogram of Total Costs')
plt.show()


# COMMAND ----------

import pandas as pd
import requests

# Sample data
data = {'item_id': [1, 2, 3], 'quantity': [10, 20, 30]}

# Validate data
def validate_data(data):
    if not isinstance(data, dict):
        raise ValueError('Invalid data format')
    if 'item_id' not in data or 'quantity' not in data:
        raise ValueError('Missing required fields')
    return True

# API integration
def integrate_with_api(data):
    url = 'https://example.com/api/inventory'
    response = requests.post(url, json=data)
    if response.status_code == 200:
        print('Data sent successfully')
    else:
        print('Error sending data')

# Validate and send data
if validate_data(data):
    integrate_with_api(data)
