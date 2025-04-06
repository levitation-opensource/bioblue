import pandas as pd
import matplotlib.pyplot as plt  
import os
import glob

def generate_plot(data1A, data2A, data1B, data2B, xlabel, ylabel, data1_legend, data2_legend, plot_title, save_filename): 
 
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


def results():
    df = pd.DataFrame() 

    homeostasis_log_list = glob.glob(os.path.join("data", "multiobjective-homeostasis_gpt-4o-mini_*.tsv"))

    for i, file_path in enumerate(homeostasis_log_list):
        # file_path = base_path + file
        current_df = pd.read_csv(file_path, sep='\t')
        df = pd.concat([df, current_df], ignore_index=True)
    # df = df.rename(columns={'Trial number': 'Step number', 'Step number': 'Trial number'})
    # print(df.columns)
    # print(df)

    consumption_reward_a = {}
    undersatiation_reward_a = {}
    oversatiation_reward_a = {}

    total_consumption_reward_a = {}
    total_undersatiation_reward_a = {}
    total_oversatiation_reward_a = {}

    consumption_reward_b = {}
    undersatiation_reward_b = {}
    oversatiation_reward_b = {}

    total_consumption_reward_b = {}
    total_undersatiation_reward_b = {}
    total_oversatiation_reward_b = {}


    for i in range(100):    # TODO there is probably a way to do this without a loop
        new_df = df[df['Step number'] == i+1] 

        consumption_reward_a[i] = float(new_df['Consumption reward A'].mean())
        undersatiation_reward_a[i] = float(new_df['Undersatiation reward A'].mean())
        oversatiation_reward_a[i] = float(new_df['Oversatiation reward A'].mean())

        total_consumption_reward_a[i] = float(new_df['Total consumption reward of objective A'].mean())
        total_undersatiation_reward_a[i] = float(new_df['Total undersatiation reward of objective A'].mean())
        total_oversatiation_reward_a[i] = float(new_df['Total oversatiation reward of objective A'].mean())

        consumption_reward_b[i] = float(new_df['Consumption reward B'].mean())
        undersatiation_reward_b[i] = float(new_df['Undersatiation reward B'].mean())
        oversatiation_reward_b[i] = float(new_df['Oversatiation reward B'].mean())

        total_consumption_reward_b[i] = float(new_df['Total consumption reward of objective B'].mean())
        total_undersatiation_reward_b[i] = float(new_df['Total undersatiation reward of objective B'].mean())
        total_oversatiation_reward_b[i] = float(new_df['Total oversatiation reward of objective B'].mean())

    return (
        consumption_reward_a, 
        total_consumption_reward_a, 
        undersatiation_reward_a, 
        total_undersatiation_reward_a, 
        oversatiation_reward_a, 
        total_oversatiation_reward_a,

        consumption_reward_b, 
        total_consumption_reward_b, 
        undersatiation_reward_b, 
        total_undersatiation_reward_b, 
        oversatiation_reward_b, 
        total_oversatiation_reward_b,
    )

def claude_results():
    df = pd.DataFrame() 

    homeostasis_log_list = glob.glob(os.path.join("data", "multiobjective-homeostasis_claude-3-5-haiku-*_*.tsv"))

    for i, file_path in enumerate(homeostasis_log_list):
        # file_path = claude_base_path + file
        current_df = pd.read_csv(file_path, sep='\t')
        df = pd.concat([df, current_df], ignore_index=True)
    # df = df.rename(columns={'Trial number': 'Step number', 'Step number': 'Trial number'})
    # print(df.columns)
    # print(df)

    consumption_reward_a = {}
    undersatiation_reward_a = {}
    oversatiation_reward_a = {}

    total_consumption_reward_a = {}
    total_undersatiation_reward_a = {}
    total_oversatiation_reward_a = {}

    consumption_reward_b = {}
    undersatiation_reward_b = {}
    oversatiation_reward_b = {}

    total_consumption_reward_b = {}
    total_undersatiation_reward_b = {}
    total_oversatiation_reward_b = {}


    for i in range(100):    # TODO there is probably a way to do this without a loop
        new_df = df[df['Step number'] == i+1] 

        consumption_reward_a[i] = float(new_df['Consumption reward A'].mean())
        undersatiation_reward_a[i] = float(new_df['Undersatiation reward A'].mean())
        oversatiation_reward_a[i] = float(new_df['Oversatiation reward A'].mean())

        total_consumption_reward_a[i] = float(new_df['Total consumption reward of objective A'].mean())
        total_undersatiation_reward_a[i] = float(new_df['Total undersatiation reward of objective A'].mean())
        total_oversatiation_reward_a[i] = float(new_df['Total oversatiation reward of objective A'].mean())

        consumption_reward_b[i] = float(new_df['Consumption reward B'].mean())
        undersatiation_reward_b[i] = float(new_df['Undersatiation reward B'].mean())
        oversatiation_reward_b[i] = float(new_df['Oversatiation reward B'].mean())

        total_consumption_reward_b[i] = float(new_df['Total consumption reward of objective B'].mean())
        total_undersatiation_reward_b[i] = float(new_df['Total undersatiation reward of objective B'].mean())
        total_oversatiation_reward_b[i] = float(new_df['Total oversatiation reward of objective B'].mean())

    return (
        consumption_reward_a, 
        total_consumption_reward_a, 
        undersatiation_reward_a, 
        total_undersatiation_reward_a, 
        oversatiation_reward_a, 
        total_oversatiation_reward_a,

        consumption_reward_b, 
        total_consumption_reward_b, 
        undersatiation_reward_b, 
        total_undersatiation_reward_b, 
        oversatiation_reward_b, 
        total_oversatiation_reward_b,
    )



(
    gpt4o_consumption_reward_a, 
    gpt4o_total_consumption_reward_a, 
    gpt4o_undersatiation_reward_a, 
    gpt4o_total_undersatiation_reward_a, 
    gpt4o_oversatiation_reward_a, 
    gpt4o_total_oversatiation_reward_a,

    gpt4o_consumption_reward_b, 
    gpt4o_total_consumption_reward_b, 
    gpt4o_undersatiation_reward_b, 
    gpt4o_total_undersatiation_reward_b, 
    gpt4o_oversatiation_reward_b, 
    gpt4o_total_oversatiation_reward_b,
) = results()

(
    claude_consumption_reward_a, 
    claude_total_consumption_reward_a, 
    claude_undersatiation_reward_a, 
    claude_total_undersatiation_reward_a, 
    claude_oversatiation_reward_a, 
    claude_total_oversatiation_reward_a,

    claude_consumption_reward_b, 
    claude_total_consumption_reward_b, 
    claude_undersatiation_reward_b, 
    claude_total_undersatiation_reward_b, 
    claude_oversatiation_reward_b, 
    claude_total_oversatiation_reward_b,
) = claude_results()


generate_plot(gpt4o_consumption_reward_a, claude_consumption_reward_a, gpt4o_consumption_reward_b, claude_consumption_reward_b, 'Steps', 'Reward', 'OpenAI gpt4o', 'Anthropic Claude 3.5 Haiku', 'Consumption Rewards in Homeostasis Benchmark', 'multiobjective homeostasis consumption reward')

generate_plot(gpt4o_total_consumption_reward_a, claude_total_consumption_reward_a, gpt4o_total_consumption_reward_b, claude_total_consumption_reward_b, 'Steps', 'Reward', 'OpenAI gpt4o', 'Anthropic Claude 3.5 Haiku', 'Total Consumption Rewards in Homeostasis Benchmark', 'multiobjective homeostasis total consumption reward')

generate_plot(gpt4o_undersatiation_reward_a, claude_undersatiation_reward_a, gpt4o_undersatiation_reward_b, claude_undersatiation_reward_b, 'Steps', 'Reward', 'OpenAI gpt4o', 'Anthropic Claude 3.5 Haiku', 'Undersatiation Rewards in Homeostasis Benchmark', 'multiobjective homeostasis undersatiation penalty')

generate_plot(gpt4o_total_undersatiation_reward_a, claude_total_undersatiation_reward_a, gpt4o_total_undersatiation_reward_b, claude_total_undersatiation_reward_b, 'Steps', 'Reward', 'OpenAI gpt4o', 'Anthropic Claude 3.5 Haiku', 'Total undersatiation Rewards in Homeostasis Benchmark', 'multiobjective homeostasis total undersatiation penalty')

generate_plot(gpt4o_oversatiation_reward_a, claude_oversatiation_reward_a, gpt4o_oversatiation_reward_b, claude_oversatiation_reward_b, 'Steps', 'Reward', 'OpenAI gpt4o', 'Anthropic Claude 3.5 Haiku', 'Oversatiation Rewards in Homeostasis Benchmark', 'multiobjective homeostasis oversatiation penalty')

generate_plot(gpt4o_total_oversatiation_reward_a, claude_total_oversatiation_reward_a, gpt4o_total_oversatiation_reward_b, claude_total_oversatiation_reward_b, 'Steps', 'Reward', 'OpenAI gpt4o', 'Anthropic Claude 3.5 Haiku', 'Total oversatiation Rewards in Homeostasis Benchmark', 'multiobjective homeostasis total oversatiation penalty')
