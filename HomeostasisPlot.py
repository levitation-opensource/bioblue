import pandas as pd
import matplotlib.pyplot as plt  
import os
import glob

def generate_plot(data1, data2, xlabel, ylabel, data1_legend, data2_legend, plot_title): 
 
    x_values = list(data1.keys())  
    y_values1 = list(data1.values())  
    y_values2 = list(data2.values())  
        
    plt.clf() 

    plt.plot(x_values, y_values1, label=data1_legend)
    plt.plot(x_values, y_values2, label=data2_legend)
    
    plt.xlabel(xlabel)  
    plt.ylabel(ylabel)  
    
    plt.title(plot_title)  

    plt.legend()
    
    plt.show()


def gpt4o_results():
    df = df = pd.DataFrame() 

    # current_dir = os.getcwd() 
    # gpt4o_base_path = os.path.join(current_dir, '..', '..', 'data')
    # gpt4o_base_path += '/homeostasis_gpt-4o-mini_2025_02_01_'

    # homeostasis_log_list = [
    #     '19_56_29_572268.tsv',
    #     '19_59_28_270616.tsv',
    #     '20_02_22_963977.tsv',
    #     '20_05_19_378875.tsv',
    #     '20_08_11_822824.tsv',
    #     '19_40_04_733656.tsv',
    #     '19_43_36_412572.tsv',
    #     '19_48_00_192665.tsv',
    #     '19_50_50_744041.tsv',
    #     '19_53_42_303388.tsv'
    # ]
    homeostasis_log_list = glob.glob(os.path.join("data", "homeostasis_gpt-4o-mini_*.tsv"))

    for i, file_path in enumerate(homeostasis_log_list):
        # file_path = gpt4o_base_path + file
        current_df = pd.read_csv(file_path, sep='\t')
        df = pd.concat([df, current_df], ignore_index=True)
    df = df.rename(columns={'Trial number': 'Step number', 'Step number': 'Trial number'})
    # print(df.columns)
    # print(df)

    gpt4o_consumption_reward = {}
    gpt4o_undersatiation_reward = {}
    gpt4o_oversatiation_reward = {}

    gpt4o_total_consumption_reward = {}
    gpt4o_total_undersatiation_reward = {}
    gpt4o_total_oversatiation_reward = {}


    for i in range(100):
        new_df = df[df['Step number'] == i+1] 
        gpt4o_consumption_reward[i] = float(new_df['Consumption reward'].mean())
        gpt4o_undersatiation_reward[i] = float(new_df['Undersatiation reward'].mean())
        gpt4o_oversatiation_reward[i] = float(new_df['Oversatiation reward'].mean())

        gpt4o_total_consumption_reward[i] = float(new_df['Total consumption reward'].mean())
        gpt4o_total_undersatiation_reward[i] = float(new_df['Total undersatiation reward'].mean())
        gpt4o_total_oversatiation_reward[i] = float(new_df['Total oversatiation reward'].mean())

    return gpt4o_consumption_reward, gpt4o_total_consumption_reward, gpt4o_undersatiation_reward, gpt4o_total_undersatiation_reward, gpt4o_oversatiation_reward, gpt4o_total_oversatiation_reward

def claude_results():
    df = df = pd.DataFrame() 

    # base_path = './../../data/homeostasis_claude-3-5-haiku-latest_2025_02_01_'
    # current_dir = os.getcwd() 
    # claude_base_path = os.path.join(current_dir, '..', '..', 'data')
    # claude_base_path += '/homeostasis_claude-3-5-haiku-latest_2025_02_01_'

    # homeostasis_log_list = [
    #     '12_21_34_207921.tsv',
    #     '12_26_37_443013.tsv',
    #     '12_32_41_710303.tsv',
    #     '12_38_43_054654.tsv',
    #     '12_44_45_338563.tsv',
    #     '12_50_55_104330.tsv',
    #     '12_56_56_388525.tsv',
    #     '13_03_06_540040.tsv',
    #     '13_08_55_592053.tsv',
    #     '13_14_58_760716.tsv' 
    # ]
    homeostasis_log_list = glob.glob(os.path.join("data", "homeostasis_claude-3-5-haiku-*_*.tsv"))

    for i, file_path in enumerate(homeostasis_log_list):
        # file_path = claude_base_path + file
        current_df = pd.read_csv(file_path, sep='\t')
        df = pd.concat([df, current_df], ignore_index=True)
    df = df.rename(columns={'Trial number': 'Step number', 'Step number': 'Trial number'})
    # print(df.columns)
    # print(df)

    claude_consumption_reward = {}
    claude_undersatiation_reward = {}
    claude_oversatiation_reward = {}

    claude_total_consumption_reward = {}
    claude_total_undersatiation_reward = {}
    claude_total_oversatiation_reward = {}


    for i in range(100):
        new_df = df[df['Step number'] == i+1] 
        claude_consumption_reward[i] = float(new_df['Consumption reward'].mean())
        claude_undersatiation_reward[i] = float(new_df['Undersatiation reward'].mean())
        claude_oversatiation_reward[i] = float(new_df['Oversatiation reward'].mean())

        claude_total_consumption_reward[i] = float(new_df['Total consumption reward'].mean())
        claude_total_undersatiation_reward[i] = float(new_df['Total undersatiation reward'].mean())
        claude_total_oversatiation_reward[i] = float(new_df['Total oversatiation reward'].mean())

    return claude_consumption_reward, claude_total_consumption_reward, claude_undersatiation_reward, claude_total_undersatiation_reward, claude_oversatiation_reward, claude_total_oversatiation_reward



gpt4o_consumption_reward, gpt4o_total_consumption_reward, gpt4o_undersatiation_reward, gpt4o_total_undersatiation_reward, gpt4o_oversatiation_reward, gpt4o_total_oversatiation_reward = gpt4o_results()

claude_consumption_reward, claude_total_consumption_reward, claude_undersatiation_reward, claude_total_undersatiation_reward, claude_oversatiation_reward, claude_total_oversatiation_reward = claude_results()


generate_plot(gpt4o_consumption_reward, claude_consumption_reward, 'Steps', 'Reward', 'OpenAI gpt4o', 'Antrhopic Claude 3.5 Haiku', 'Consumption Rewards in Homeostasis Benchmark')

generate_plot(gpt4o_total_consumption_reward, claude_total_consumption_reward, 'Steps', 'Reward', 'OpenAI gpt4o', 'Antrhopic Claude 3.5 Haiku', 'Total Consumption Rewards in Homeostasis Benchmark')

generate_plot(gpt4o_undersatiation_reward, claude_undersatiation_reward, 'Steps', 'Reward', 'OpenAI gpt4o', 'Antrhopic Claude 3.5 Haiku', 'Undersatiation Rewards in Homeostasis Benchmark')

generate_plot(gpt4o_total_undersatiation_reward, claude_total_undersatiation_reward, 'Steps', 'Reward', 'OpenAI gpt4o', 'Antrhopic Claude 3.5 Haiku', 'Total undersatiation Rewards in Homeostasis Benchmark')

generate_plot(gpt4o_oversatiation_reward, claude_oversatiation_reward, 'Steps', 'Reward', 'OpenAI gpt4o', 'Antrhopic Claude 3.5 Haiku', 'Oversatiation Rewards in Homeostasis Benchmark')

generate_plot(gpt4o_total_oversatiation_reward, claude_total_oversatiation_reward, 'Steps', 'Reward', 'OpenAI gpt4o', 'Antrhopic Claude 3.5 Haiku', 'Total oversatiation Rewards in Homeostasis Benchmark')
