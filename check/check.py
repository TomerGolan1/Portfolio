import glob
import importlib
import os
from datetime import date
import time
import json
import zipfile
import pandas

try:  ## check if a directory exist, if not create an output directory.
    os.mkdir("output")
except : pass


## params:
tickers_list=[['GOOG', 'AAPL']]
start_date=[date(2020, 1, 1)]
end_date=[date(2020, 1, 6)]



######################################################################################################
######################################################################################################
flag=True
while flag:
    print("Please enter a file name:")
    File = input()
    if File[-3:] == ".py":
        if not (File[:-3].isdigit()):
            print("File name is invalid! The file name should be your identity number")
        else:
            File=File[:-3]
            flag=False

    else:
        if not (File.isdigit()):
            print("File name is invalid! The file name should be your identity number")
        else:
            flag=False

######################################################################################################
######################################################################################################
file_list = []
try:
    path_to_zip_file='zip_to_here/'
    with zipfile.ZipFile(path_to_zip_file + File + ".zip", 'r') as zip_ref:
        zip_ref.extractall(path_to_zip_file)
except Exception as e:  # if the code is crashing report it
    print("Error opening zip file", ": crashed-", e)
    exit(1)
try:
    for file in glob.glob(path_to_zip_file +"*.py"):
        file=file.replace('\\','/')
        file_name=file.split('/')[1]
        file_list.append(file_name)
        os.replace(file, file_name)


except Exception as e:  # if the code is crashing report it
    print("Error copying files", ": crashed-", e)
    exit(1)
## import the py file:
try:
    Module = importlib.import_module(File)
    Output = File
    Output += "-test"
    print(Output)
    test_number = 1
    number_of_tries = 5
    test_pass = True
except Exception as e:  # if the code is crashing report it
    print("Error opening the main file", ": crashed-", e)
    exit(1)
try:
    t0 = time.time()
    pb = Module.PortfolioBuilder()
    while number_of_tries > 0:
        try:
            df = pb.get_daily_data(['GOOG', 'AAPL', 'MSFT'], date(2018, 1, 1), date(2020, 5, 14))
            if df.empty:
                number_of_tries -= 1
            else:
                number_of_tries = 0
        except Exception as e:  # if the code is crashing report it
            print("Action failed-", e)
            number_of_tries -= 1
    universal = pb.find_universal_portfolio(10)
    df_json = df.to_json(orient='table',date_format='iso')
    with open('test/ans1.txt') as json_file:
        data = json.load(json_file)
    data = json.loads(data)
    df_json = json.loads(df_json)
    for i in range(len(data['data'])):
        if (data['data'][i]['Date'] != df_json['data'][i]['Date']) or (abs(data['data'][i]['GOOG'] - df_json['data'][i]['GOOG']) > 0.0001) or (abs(data['data'][i]['AAPL'] - df_json['data'][i]['AAPL']) > 0.0001) or (abs(data['data'][i]['MSFT'] - df_json['data'][i]['MSFT']) > 0.0001):
            test_pass = False
            break
    if test_pass:
        print("Test number-", str(test_number), "pass")
    else:
        print("Test number-", str(test_number), "Failed: The return of the daily data is incorrect")

except Exception as e:  # if the code is crashing report it
    print("Test Number ", test_number, ": crashed-", e)
test_number += 1
test_pass = True
try:
    universal = pb.find_universal_portfolio(10)
    with open('test/ans2.txt') as json_file:
        data = json.load(json_file)
    for i in range(len(data)):
        if abs(data[i] - universal[i]) > 0.0001:
            test_pass = False
            break
    if test_pass:
        print("Test number-", str(test_number), "pass")
    else:
        print("Test number-", str(test_number), "Failed: The return of the universal portfolio is incorrect")
except Exception as e:  # if the code is crashing report it
    print("Test Number ", test_number, ": crashed-", e)

test_number += 1
test_pass = True
try:
    exponential_grad = pb.find_exponential_gradient_portfolio()
    with open('test/ans3.txt') as json_file:
        data = json.load(json_file)
    for i in range(len(data)):
        if abs(data[i] - exponential_grad[i]) > 0.0001:
            test_pass = False
            break
    if test_pass:
        print("Test number-", str(test_number), "pass")
    else:
        print("Test number-", str(test_number), "Failed: The return of the gradient portfolio is incorrect")
except Exception as e:  # if the code is crashing report it
    print("Test Number ", test_number, ": crashed-", e)
for file in file_list:
    os.remove(file)