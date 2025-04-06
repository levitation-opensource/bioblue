import pandas as pd
import matplotlib.pyplot as plt  
import os
import glob

use_hint = False

hint_data_filename_sufix = "-no-hint" if not use_hint else "-with-hint"
hint_plot_filename_sufix = " no hint" if not use_hint else " with hint"
hint_plot_title_sufix = " No Hint" if not use_hint else " With Hint"

def generate_plot_four_series(data1A, data2A, data1B, data2B, xlabel, ylabel, data1_legend, data2_legend, plot_title, save_filename): 
 
    x_values = list(data1A.keys())  

    y_values1A = list(data1A.values())  
    y_values2A = list(data2A.values())  
    y_values1B = list(data1B.values())  
    y_values2B = list(data2B.values())  
        
    plt.clf()

    plt.plot(x_values, y_values1A, label=data1_legend + " of objective A")
    plt.plot(x_values, y_values2A, label=data2_legend + " of objective A")
    plt.plot(x_values, y_values1B, label=data1_legend + " of objective B")
    plt.plot(x_values, y_values2B, label=data2_legend + " of objective B")
   
    plt.xlabel(xlabel)  
    plt.ylabel(ylabel)  
    
    plt.title(plot_title)  

    plt.legend()
    
    plt.savefig(save_filename + ".png", dpi=200)
    plt.show()


def generate_plot_two_series(data1, data2, xlabel, ylabel, data1_legend, data2_legend, plot_title, save_filename): 
 
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


def results():
    df = pd.DataFrame() 

    homeostasis_log_list = glob.glob(os.path.join("data", f"balancing-unbounded-objectives{hint_data_filename_sufix}_gpt-4o-mini_*.tsv"))

    for i, file_path in enumerate(homeostasis_log_list):
        # file_path = base_path + file
        current_df = pd.read_csv(file_path, sep='\t')
        df = pd.concat([df, current_df], ignore_index=True)
    # df = df.rename(columns={'Trial number': 'Step number', 'Step number': 'Trial number'})
    # print(df.columns)
    # print(df)

    harvesting_reward_a = {}
    total_harvesting_reward_a = {}

    harvesting_reward_b = {}
    total_harvesting_reward_b = {}

    imbalance_reward = {}
    total_imbalance_reward = {}


    for i in range(100):    # TODO there is probably a way to do this without a loop
        new_df = df[df['Step number'] == i+1] 

        harvesting_reward_a[i] = float(new_df['Harvesting reward A'].mean())
        total_harvesting_reward_a[i] = float(new_df['Total harvesting reward of objective A'].mean())

        harvesting_reward_b[i] = float(new_df['Harvesting reward B'].mean())
        total_harvesting_reward_b[i] = float(new_df['Total harvesting reward of objective B'].mean())

        imbalance_reward[i] = float(new_df['Imbalance reward'].mean())
        total_imbalance_reward[i] = float(new_df['Total imbalance reward'].mean())

    return (
        harvesting_reward_a, 
        total_harvesting_reward_a, 

        harvesting_reward_b, 
        total_harvesting_reward_b, 

        imbalance_reward, 
        total_imbalance_reward, 
    )

def claude_results():
    df = pd.DataFrame() 

    homeostasis_log_list = glob.glob(os.path.join("data", f"balancing-unbounded-objectives{hint_data_filename_sufix}_claude-3-5-haiku-*_*.tsv"))

    for i, file_path in enumerate(homeostasis_log_list):
        # file_path = claude_base_path + file
        current_df = pd.read_csv(file_path, sep='\t')
        df = pd.concat([df, current_df], ignore_index=True)
    # df = df.rename(columns={'Trial number': 'Step number', 'Step number': 'Trial number'})
    # print(df.columns)
    # print(df)

    harvesting_reward_a = {}
    total_harvesting_reward_a = {}

    harvesting_reward_b = {}
    total_harvesting_reward_b = {}

    imbalance_reward = {}
    total_imbalance_reward = {}


    for i in range(100):    # TODO there is probably a way to do this without a loop
        new_df = df[df['Step number'] == i+1] 

        harvesting_reward_a[i] = float(new_df['Harvesting reward A'].mean())
        total_harvesting_reward_a[i] = float(new_df['Total harvesting reward of objective A'].mean())

        harvesting_reward_b[i] = float(new_df['Harvesting reward B'].mean())
        total_harvesting_reward_b[i] = float(new_df['Total harvesting reward of objective B'].mean())

        imbalance_reward[i] = float(new_df['Imbalance reward'].mean())
        total_imbalance_reward[i] = float(new_df['Total imbalance reward'].mean())

    return (
        harvesting_reward_a, 
        total_harvesting_reward_a, 

        harvesting_reward_b, 
        total_harvesting_reward_b, 

        imbalance_reward, 
        total_imbalance_reward, 
    )



(
    gpt4o_harvesting_reward_a, 
    gpt4o_total_harvesting_reward_a, 

    gpt4o_harvesting_reward_b, 
    gpt4o_total_harvesting_reward_b, 

    gpt4o_imbalance_reward, 
    gpt4o_total_imbalance_reward, 
) = results()

(
    claude_harvesting_reward_a, 
    claude_total_harvesting_reward_a, 

    claude_harvesting_reward_b, 
    claude_total_harvesting_reward_b, 

    claude_imbalance_reward, 
    claude_total_imbalance_reward, 
) = claude_results()


generate_plot_four_series(gpt4o_harvesting_reward_a, claude_harvesting_reward_a, gpt4o_harvesting_reward_b, claude_harvesting_reward_b, 'Steps', 'Reward', 'OpenAI gpt4o', 'Anthropic Claude 3.5 Haiku', f'Harvesting rewards in Balancing Unbounded Objectives{hint_plot_title_sufix} Bench', f'balancing unbounded objectives{hint_plot_filename_sufix} harvesting reward')

generate_plot_four_series(gpt4o_total_harvesting_reward_a, claude_total_harvesting_reward_a, gpt4o_total_harvesting_reward_b, claude_total_harvesting_reward_b, 'Steps', 'Reward', 'OpenAI gpt4o', 'Anthropic Claude 3.5 Haiku', f'Total Harvesting rewards in Balancing Unbounded Objectives{hint_plot_title_sufix} Bench', f'balancing unbounded objectives{hint_plot_filename_sufix} total harvesting reward')

generate_plot_two_series(gpt4o_imbalance_reward, claude_imbalance_reward, 'Steps', 'Reward', 'OpenAI gpt4o', f'Anthropic Claude 3.5 Haiku', f'Imbalance rewards in Balancing Unbounded Objectives{hint_plot_title_sufix} Bench', f'balancing unbounded objectives{hint_plot_filename_sufix} imbalance penalty')

generate_plot_two_series(gpt4o_total_imbalance_reward, claude_total_imbalance_reward, 'Steps', 'Reward', 'OpenAI gpt4o', 'Anthropic Claude 3.5 Haiku', f'Total imbalance rewards in Balancing Unbounded Objectives{hint_plot_title_sufix} Bench', f'balancing unbounded objectives{hint_plot_filename_sufix} total imbalance penalty')
