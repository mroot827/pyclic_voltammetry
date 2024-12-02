
#DEPENDENCIES:
!pip install numpy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

cv_txt = 'YOUR .TXT FILE HERE'

def pyclic_voltammetry(cv_txt):

  #loading in CV datapoints from file:
  with open(cv_txt, 'r') as file:
      lines = file.readlines()

  start_index = None
  scan_rate = None
  for i, line in enumerate(lines):
      if 'Potential/V, Current/A' in line:
          start_index = i + 1
          break
      if 'Scan Rate (V/s)' in line:
          scan_rate = float(line.split('=')[1].strip())
          break
  data_lines = lines[start_index:]
  data = []

  for line in data_lines:
      if line.strip() and ',' in line:
          parts = line.strip().split(',')
          if len(parts) == 2:
              try:
                  potential, current = float(parts[0]), float(parts[1])
                  data.append([potential, current])
              except ValueError:
                  continue

  df = pd.DataFrame(data, columns=['Potential/V', 'Current/A'])
  df['Current/A'] = df['Current/A'] * 10**5


  #****************************************************************************

  #extracting other information from file:
  min_potential = df['Potential/V'].min()
  max_potential = df['Potential/V'].max()

  range_of_potential= [max_potential,min_potential]

  #****************************************************************************
  #finding minimum and maximum current values:
  red_current = df['Current/A'].min()
  ox_current = df['Current/A'].max()

  #****************************************************************************
  #solving for peak-to-peak seperation:
  #oxidation potential:
  red_pot = df.loc[df['Current/A'] == red_current, 'Potential/V'].values[0]
  ox_pot = df.loc[df['Current/A'] == ox_current, 'Potential/V'].values[0]

  peak_separation = ox_pot - red_pot

  half_wave = (ox_pot + red_pot)/ 2

  #****************************************************************************
  initial_current = df['Current/A'].iloc[0]
  red_peak = ox_current - initial_current
  min_value = df['Current/A'].min()
  #****************************************************************************
  red_current_index= df.loc[df['Current/A'] == red_current].index[0]
  ox_current_index = df.loc[df['Current/A'] == ox_current].index[0]
  min_potential_index = df.loc[df['Potential/V'] == min_potential].index[0]

  #reduction peak height:
  start_current = df['Current/A'].iloc[0]
  ipa = (start_current - min_value)


  #****************************************************************************
  #Setting window to solve for tangent line to return wave:
  window = df.iloc[min_potential_index:ox_current_index]
  adjust_coef = 2*len(window)/3
  window = df.iloc[min_potential_index:ox_current_index-int(adjust_coef)]
  #****************************************************************************
  #finding tangent line to return wave:

  def perform_linear_regression(segment):
      pot_reg = segment['Potential/V'].values.reshape(-1, 1)
      current_reg = segment['Current/A'].values

      model = LinearRegression()
      model.fit(pot_reg, current_reg)

      return model.coef_[0], model.intercept_

  window_size = 25

  slopes = []
  intercepts = []
  errors = []

  #iterating through subwindows of dataframe to find linear sections:
  for i in range(len(window) - window_size + 1):

      segment = window.iloc[i:i + window_size]
      slope, intercept = perform_linear_regression(segment)
      slopes.append(slope)
      intercepts.append(intercept)
      X_segment = segment['Potential/V'].values.reshape(-1, 1)
      y_pred = slope * X_segment + intercept
      residuals = segment['Current/A'].values - y_pred.flatten()
      rmse = np.sqrt(np.mean(residuals**2))
      errors.append(rmse)


  results = pd.DataFrame({
      'start_index': range(len(window) - window_size + 1),
      'rmse': errors
  })

  best_start_index = results['rmse'].idxmin()

  end_index = best_start_index + window_size

  linear_segment = window.iloc[best_start_index:end_index]

  #finding equation of linear segment:
  def finding_tangent_local(segment):
      x_data = segment['Potential/V'].values
      y_data = segment['Current/A'].values
      x_data = x_data.reshape(-1, 1)
      y_data = y_data.reshape(-1, 1)

      # lin reg
      model = LinearRegression()
      model.fit(x_data, y_data)
      slope = model.coef_[0]
      intercept = model.intercept_

      return slope, intercept

  slope, intercept = finding_tangent_local(linear_segment)

  x_tangent = np.linspace(linear_segment['Potential/V'].min(), red_pot, 100) # Adjust 100 for desired smoothness
  y_tangent = slope * x_tangent + intercept
  #****************************************************************************
  #calculating peak ratio:
  ipf = slope * red_pot + intercept
  ipf = np.array(ipf)
  ipf = ipf[0]
  peak_ratio = abs(ipf/ipa)
  print(peak_ratio)

  #****************************************************************************
  parameters = pd.DataFrame({
      'Reduction Potential [V]': [red_pot],
      'Oxidation Potential [V]': [ox_pot],
      'Peak-to-Peak Seperation [V]': [peak_separation],
      'Half Wave Potential [V]': [half_wave],
      'Peak Ratio': [peak_ratio]
  })


  return parameters

minval = min_value * 10**-5
plt.plot(df['Potential/V'], df['Current/A'], color='black')
plt.xlabel('Potential, V vs Ag Wire')
plt.ylabel('Current [µA]')
plt.title(f'{scan_rate} V/s')

plt.plot([ox_pot, ox_pot], [initial_current, df['Current/A'].max()], linestyle='--', label='Oxidation Potential', color ='firebrick', linewidth= 0.8)
plt.plot([df['Potential/V'].max(), ox_pot], [initial_current, initial_current], linestyle='--', color ='firebrick', linewidth= 0.8)
plt.plot(x_tangent, y_tangent, color='firebrick',linestyle='--',linewidth= 0.8)
plt.plot([red_pot, red_pot], [min_value , ipf], linestyle='--', label='Reduction Potential', color='firebrick', linewidth=0.8)
plt.show()

print(parameters)
