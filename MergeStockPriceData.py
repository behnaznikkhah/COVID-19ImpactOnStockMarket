# MergeStockPriceData.py

# Sai Madhuri Yerramsetti
# October 7, 2020
# Student Number: 0677671


# Import packages
import pandas as pd
import os

# Function to read and merge the ctock market data 
def read_and_merge_data(file_dir):
    
    filepath_list = []

    # make a list of csv files list in the given folder
    file_path = os.path.abspath(file_dir) 
    filepath_list = [os.path.join(file_path, f) for f in os.listdir(file_path) if f.endswith('.csv')]

    print("file path list is", filepath_list)

    # Read the csv from files list, add a new column with country name and rename column name of 'Vol.' to 'Volume'
    data_Abudhabi = pd.read_csv(filepath_list[0])
    data_Abudhabi['Country'] = 'UAE'
    data_Abudhabi.rename(columns = {'Vol.':'Volume'}, inplace = True)

    # Read the csv from files list, add a new column with country name and rename column name of 'Vol.' to 'Volume'
    data_Brazil = pd.read_csv(filepath_list[1])
    data_Brazil['Country'] = 'Brazil'
    data_Brazil.rename(columns = {'Vol.':'Volume'}, inplace = True)

    # Read the csv from files list, add a new column with country name and rename column name of 'Vol.' to 'Volume'
    data_France = pd.read_csv(filepath_list[2])
    data_France['Country'] = 'France'
    data_France.rename(columns = {'Vol.':'Volume'}, inplace = True)

    # Read the csv from files list, add a new column with country name and rename column name of 'Vol.' to 'Volume'
    data_Germany = pd.read_csv(filepath_list[3])
    data_Germany['Country'] = 'Germany'
    data_Germany.rename(columns = {'Vol.':'Volume'}, inplace = True)

    # Read the csv from files list, add a new column with country name and rename column name of 'Vol.' to 'Volume'
    data_USA = pd.read_csv(filepath_list[7])
    data_USA['Country'] = 'USA'
    data_USA.rename(columns = {'Vol.':'Volume'}, inplace = True)

    # Read the csv from files list, add a new column with country name and rename column name of 'Vol.' to 'Volume'
    data_sa = pd.read_csv(filepath_list[13])
    data_sa['Country'] = 'SouthAfrica'
    data_sa.rename(columns = {'Vol.':'Volume'}, inplace = True)

    # Read the csv from files list, add a new column with country name and rename column name of 'Vol.' to 'Volume'
    data_HongKong = pd.read_csv(filepath_list[4])
    data_HongKong['Country'] = 'HongKong'
    data_HongKong.rename(columns = {'Vol.':'Volume'}, inplace = True)

    # Read the csv from files list, add a new column with country name and rename column name of 'Vol.' to 'Volume'
    data_Jakarta = pd.read_csv(filepath_list[5])
    data_Jakarta['Country'] = 'Indonesia'
    data_Jakarta.rename(columns = {'Vol.':'Volume'}, inplace = True)

    # Read the csv from files list, add a new column with country name and rename column name of 'Vol.' to 'Volume'
    data_Korea = pd.read_csv(filepath_list[6])
    data_Korea['Country'] = 'SouthKorea'
    data_Korea.rename(columns = {'Vol.':'Volume'}, inplace = True)

    # Read the csv from files list, add a new column with country name and rename column name of 'Vol.' to 'Volume'
    data_India = pd.read_csv(filepath_list[8])
    data_India['Country'] = 'India'
    data_India.rename(columns = {'Vol.':'Volume'}, inplace = True)

    # Read the csv from files list, add a new column with country name and rename column name of 'Vol.' to 'Volume'
    data_Japan = pd.read_csv(filepath_list[9])
    data_Japan['Country'] = 'Japan'
    data_Japan.rename(columns = {'Vol.':'Volume'}, inplace = True)

    # Read the csv from files list, add a new column with country name and rename column name of 'Vol.' to 'Volume'
    data_Australia = pd.read_csv(filepath_list[10])
    data_Australia['Country'] = 'Australia'
    data_Australia.rename(columns = {'Vol.':'Volume'}, inplace = True)

    # Read the csv from files list, add a new column with country name and rename column name of 'Vol.' to 'Volume'
    data_Canada = pd.read_csv(filepath_list[11])
    data_Canada['Country'] = 'Canada'
    data_Canada.rename(columns = {'Vol.':'Volume'}, inplace = True)

    # Read the csv from files list, add a new column with country name and rename column name of 'Vol.' to 'Volume'
    data_Shanghai = pd.read_csv(filepath_list[12])
    data_Shanghai['Country'] = 'China'
    data_Shanghai.rename(columns = {'Vol.':'Volume'}, inplace = True)

    # combone all the files oof all countries into single file
    data = pd.concat([data_Abudhabi, data_Brazil, data_France, data_Germany, data_USA, data_sa, data_HongKong,
                      data_Jakarta, data_Korea, data_India, data_Japan, data_Australia, data_Canada, data_Shanghai], axis=0)
    print("................Merging is done..................")
    print(data.head(5))
    print(data.tail(5))
    print("Info of the combined data: \n", data.info())

    # convert the combined data into csv file and save it in local machine
    data.to_csv(r'D:\Madhuri\Big Data Project\Data\stock_prices_merged.csv', index = False, header=True) 


if __name__ == "__main__":
    
  file_dir = 'D:/Madhuri/Big Data Project/Data/Stock Price Data'

  # Function call
  read_and_merge_data(file_dir)
