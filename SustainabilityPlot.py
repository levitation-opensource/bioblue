import pandas as pd
import matplotlib.pyplot as plt  
import os
import glob

def generate_plot(data1, data2, xlabel, ylabel, data1_legend, data2_legend, plot_title, save_filename): 
 
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
    
    plt.savefig(save_filename + ".png", dpi=200)
    plt.show()


def gpt4o_results():
    df = pd.DataFrame() 

    sustainability_log_list = glob.glob(os.path.join("data", "sustainability_gpt-4o-mini_2025_02_02_*.tsv"))

    for i, file_path in enumerate(sustainability_log_list):
        # file_path = gpt4o_base_path + file
        current_df = pd.read_csv(file_path, sep='\t')
        df = pd.concat([df, current_df], ignore_index=True)

    df = df.rename(columns={'Trial number': 'Step number', 'Step number': 'Trial number'})

    avg_consumption_reward = {}
    avg_total_consumption_reward = {}
    avg_instability_reward = {}
    avg_total_instability_reward = {}

    for i in range(100):    # TODO there is probably a way to do this without a loop
        new_df = df[df['Step number'] == i+1] 
        avg_consumption_reward[i] = float(new_df['Consumption reward'].mean())
        avg_total_consumption_reward[i] = float(new_df['Total consumption reward'].mean())
        avg_instability_reward[i] = float(new_df['Instability reward'].mean())
        avg_total_instability_reward[i] = float(new_df['Total instability reward'].mean())

    return avg_consumption_reward, avg_total_consumption_reward, avg_instability_reward, avg_total_instability_reward


def claude_results():
    df = pd.DataFrame() 

    sustainability_log_list = glob.glob(os.path.join("data", "sustainability_claude-3-5-haiku-latest_2025_02_03_*.tsv"))

    for i, file_path in enumerate(sustainability_log_list):
        # file_path = claude_base_path + file
        current_df = pd.read_csv(file_path, sep='\t')
        df = pd.concat([df, current_df], ignore_index=True)

    df = df.rename(columns={'Trial number': 'Step number', 'Step number': 'Trial number'})

    avg_consumption_reward = {}
    avg_total_consumption_reward = {}
    avg_instability_reward = {}
    avg_total_instability_reward = {}

    for i in range(100):    # TODO there is probably a way to do this without a loop
        new_df = df[df['Step number'] == i+1] 
        avg_consumption_reward[i] = float(new_df['Consumption reward'].mean())
        avg_total_consumption_reward[i] = float(new_df['Total consumption reward'].mean())
        avg_instability_reward[i] = float(new_df['Instability reward'].mean())
        avg_total_instability_reward[i] = float(new_df['Total instability reward'].mean())
    
    return avg_consumption_reward, avg_total_consumption_reward, avg_instability_reward, avg_total_instability_reward


gpt4o_consumption_reward, gpt4o_total_consumption_reward, gpt4o_instability_reward, gpt4o_total_instability_reward = gpt4o_results()

claude_consumption_reward, claude_total_consumption_reward, claude_instability_reward, claude_total_instability_reward = claude_results()


generate_plot(gpt4o_consumption_reward, claude_consumption_reward, 'Steps', 'Reward', 'OpenAI gpt4o', 'Anthropic Claude 3.5 Haiku', 'Consumption reward for Sustainability benchmark (Avg over 10 trials)', 'sustainability consumption reward')

generate_plot(gpt4o_total_consumption_reward, claude_total_consumption_reward, 'Steps', 'Reward', 'OpenAI gpt4o', 'Anthropic Claude 3.5 Haiku', 'Total consumption reward for Sustainability benchmark (Avg over 10 trials)', 'sustainability total consumption reward')

generate_plot(gpt4o_instability_reward, claude_instability_reward, 'Steps', 'Reward', 'OpenAI gpt4o', 'Anthropic Claude 3.5 Haiku', 'Instability reward for Sustainability benchmark (Avg over 10 trials)', 'sustainability instability penalty')

generate_plot(gpt4o_total_instability_reward, claude_total_instability_reward, 'Steps', 'Reward', 'OpenAI gpt4o', 'Anthropic Claude 3.5 Haiku', 'Total instability reward for Sustainability benchmark (Avg over 10 trials)', 'sustainability total instability penalty')
