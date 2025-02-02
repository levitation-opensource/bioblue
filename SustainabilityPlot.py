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
    df = pd.DataFrame() 

    # current_dir = os.getcwd() 
    # gpt4o_base_path = os.path.join(current_dir, '..', '..', 'data')
    # gpt4o_base_path += '/sustainability_gpt-4o-mini_2025_02_01_'

    # sustainability_log_list = [
    #     '20_22_13_475904.tsv',
    #     '20_25_29_071555.tsv',
    #     '20_28_01_340874.tsv',
    #     '20_31_29_847955.tsv',             
    #     '20_33_53_727289.tsv',             
    #     '20_36_23_907161.tsv',             
    #     '20_39_13_750999.tsv',
    #     '20_41_53_835899.tsv',             
    #     '20_44_29_276347.tsv',
    #     '20_47_02_537301.tsv'
    # ]
    sustainability_log_list = glob.glob(os.path.join("data", "sustainability_gpt-4o-mini_*.tsv"))

    for i, file_path in enumerate(sustainability_log_list):
        # file_path = gpt4o_base_path + file
        current_df = pd.read_csv(file_path, sep='\t')
        df = pd.concat([df, current_df], ignore_index=True)

    df = df.rename(columns={'Trial number': 'Step number', 'Step number': 'Trial number'})

    avg_consumption_reward = {}
    avg_total_consumption_reward = {}
    avg_instability_reward = {}
    avg_total_instability_reward = {}

    for i in range(100):
        new_df = df[df['Step number'] == i+1] 
        avg_consumption_reward[i] = float(new_df['Consumption reward'].mean())
        avg_total_consumption_reward[i] = float(new_df['Total consumption reward'].mean())
        avg_instability_reward[i] = float(new_df['Instability reward'].mean())
        avg_total_instability_reward[i] = float(new_df['Total instability reward'].mean())

    return avg_consumption_reward, avg_total_consumption_reward, avg_instability_reward, avg_total_instability_reward


def claude_results():
    df = pd.DataFrame() 

    # current_dir = os.getcwd() 
    # claude_base_path = os.path.join(current_dir, '..', '..', 'data')
    # claude_base_path += '/sustainability_claude-3-5-haiku-latest_2025_02_01_'

    # sustainability_log_list = [
    #     '19_44_31_302947.tsv',
    #     '19_49_28_429212.tsv',
    #     '19_55_09_324643.tsv',
    #     '20_00_52_471423.tsv',
    #     '20_06_37_162068.tsv',
    #     '20_12_20_606011.tsv',
    #     '20_17_58_611178.tsv',
    #     '20_23_47_577616.tsv',
    #     '20_29_31_712147.tsv',
    #     '20_35_11_344604.tsv'
    # ]
    sustainability_log_list = glob.glob(os.path.join("data", "sustainability_claude-3-5-haiku-*_*.tsv"))

    for i, file_path in enumerate(sustainability_log_list):
        # file_path = claude_base_path + file
        current_df = pd.read_csv(file_path, sep='\t')
        df = pd.concat([df, current_df], ignore_index=True)

    df = df.rename(columns={'Trial number': 'Step number', 'Step number': 'Trial number'})

    avg_consumption_reward = {}
    avg_total_consumption_reward = {}
    avg_instability_reward = {}
    avg_total_instability_reward = {}

    for i in range(100):
        new_df = df[df['Step number'] == i+1] 
        avg_consumption_reward[i] = float(new_df['Consumption reward'].mean())
        avg_total_consumption_reward[i] = float(new_df['Total consumption reward'].mean())
        avg_instability_reward[i] = float(new_df['Instability reward'].mean())
        avg_total_instability_reward[i] = float(new_df['Total instability reward'].mean())
    
    return avg_consumption_reward, avg_total_consumption_reward, avg_instability_reward, avg_total_instability_reward


gpt4o_consumption_reward, gpt4o_total_consumption_reward, gpt4o_instability_reward, gpt4o_total_instability_reward = gpt4o_results()

claude_consumption_reward, claude_total_consumption_reward, claude_instability_reward, claude_total_instability_reward = claude_results()


generate_plot(gpt4o_consumption_reward, claude_consumption_reward, 'Steps', 'Reward', 'OpenAI gpt4o', 'Antrhopic Claude 3.5 Haiku', 'Consumption reward for Sustainability benchmark (Avg over 10 trials)')

generate_plot(gpt4o_total_consumption_reward, claude_total_consumption_reward, 'Steps', 'Reward', 'OpenAI gpt4o', 'Antrhopic Claude 3.5 Haiku', 'Total consumption reward for Sustainability benchmark (Avg over 10 trials)')

generate_plot(gpt4o_instability_reward, claude_instability_reward, 'Steps', 'Reward', 'OpenAI gpt4o', 'Antrhopic Claude 3.5 Haiku', 'Instability reward for Sustainability benchmark (Avg over 10 trials)')

generate_plot(gpt4o_total_instability_reward, claude_total_instability_reward, 'Steps', 'Reward', 'OpenAI gpt4o', 'Antrhopic Claude 3.5 Haiku', 'Total instability reward for Sustainability benchmark (Avg over 10 trials)')
