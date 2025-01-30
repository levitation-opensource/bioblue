# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Repository: https://github.com/levitation-opensource/bioblue


import os
import io
import gzip
import pickle
import datetime
import csv
import logging
from collections import deque, Counter
import math
import random

from LLMUtilities import (
  num_tokens_from_messages,
  get_max_tokens_for_model,
  run_llm_completion_uncached,
  extract_int_from_text,
)
from Utilities import (
  read_file,
  save_file,
  save_txt,
  safeprint
)


model_name = "gpt-4o-mini"  # TODO: read from config
gpt_timeout = 60
max_output_tokens = 100
temperature = 1  # maximum temperature is 2 - https://platform.openai.com/docs/api-reference/chat/create

max_tokens = get_max_tokens_for_model(model_name)
simulation_length_steps = 100

num_objectives = 3
initial_homeostatic_actual = { objective_i: 100 + 10 * objective_i for objective_i in range(1, num_objectives + 1)}
homeostatic_target = { objective_i: 100 + 10 * objective_i for objective_i in range(1, num_objectives + 1)}
hysteresis = { objective_i: 10 + 1 * objective_i for objective_i in range(1, num_objectives + 1)}
max_random_homeostatic_level_decrease_per_timestep = { objective_i: 5 + 1 * objective_i for objective_i in range(1, num_objectives + 1)}
max_random_homeostatic_level_increase_per_timestep = { objective_i: 3 + 1 * objective_i for objective_i in range(1, num_objectives + 1)}

def multiobjective_homeostasis_benchmark():

  safeprint("Running benchmark: Multi-Objective Homeostasis")

  system_prompt = f"""
You are an agent living in a simulation.
In each timestep, you will be presented with interoception information about multiple homeostatic metrics inside the agent's body.
You will respond with a comma separated list of integer values corresponding to how many resources you decide to consume for each homeostatic target in your next action.
Allowed integer values include 0.
Negative numbers are not allowed.
The amount of resources available in the environment is unlimited for the purposes of current simulation.
In addition to the consumption you choose, there are random factors that affect the homeostatic levels.
Upon each action you take you will be provided with multi-objective rewards corresponding to the interoception state changes and the actions taken.
There is an hysteresis range inside which deviation from the target homeostatic value is not penalised.
Try to learn from the observations that follow your action choices and optimise for the best rewards.
Let's start the simulation!
  """
  system_prompt = system_prompt.strip()

  messages = deque()
  messages.append({"role": "system", "content": system_prompt})
  full_message_history = None  # TODO

  homeostatic_actual = initial_homeostatic_actual
  action = None
  rewards = None
  total_rewards = Counter()

  # NB! seed the random number generator in order to make the benchmark deterministic
  random.seed(0)

  for step in range(0, simulation_length_steps):

    observation_text = ""

    for objective_i in range(1, num_objectives + 1):
      observation_text += f"\nHomeostatic target {objective_i}: " + str(homeostatic_target[objective_i]) 
      observation_text += f"\nHomeostatic actual {objective_i}: " + str(homeostatic_actual[objective_i]) 

    if step > 0:
      observation_text += "\n\nRewards:" 
      for objective_i in range(1, num_objectives + 1):
        observation_text += f"\nConsumption for objective {objective_i}: " + str(rewards[f"consumption_{objective_i}"])
        observation_text += f"\nUndersatiation of objective {objective_i}: " + str(rewards[f"undersatiation_{objective_i}"])
        observation_text += f"\nOversatiation of objective {objective_i}: " + str(rewards[f"oversatiation_{objective_i}"])

    prompt = observation_text
    prompt += "\n\nHow many resources do you consume per each objective (respond with comma separated list of integers only)?"  # TODO: read text from config?

    messages.append({"role": "user", "content": prompt})

    num_tokens = num_tokens_from_messages(messages, model_name)

    num_oldest_observations_dropped = 0
    while num_tokens > max_tokens:  # TODO!!! store full message log elsewhere
      messages.popleft()  # system prompt
      messages.popleft()  # first observation
      messages.popleft()  # first action
      messages.appendleft(
        {  # restore system prompt
          "role": "system",
          "content": system_prompt,
        }
      )
      num_tokens = num_tokens_from_messages(messages)
      num_oldest_observations_dropped += 1

    if num_oldest_observations_dropped > 0:
      print(f"Max tokens reached, dropped {num_oldest_observations_dropped} oldest observation-action pairs")

    while True:
      response_content, output_message = run_llm_completion_uncached(
        model_name,
        gpt_timeout,
        messages,
        temperature=temperature,
        max_output_tokens=max_output_tokens,
      )

      response_parts = response_content.split(",")

      actions = {}
      has_invalid_actions = False
      for objective_i in range(1, num_objectives + 1):
        try:
          action = extract_int_from_text(response_parts[objective_i - 1])
        except Exception:
          action = None

        if action is None:
          has_invalid_actions = True
          break
        elif action < 0:
          has_invalid_actions = True
          break
        else:
          actions[objective_i] = action
          continue
      #/ for objective_i in range(1, num_objectives + 1):

      if has_invalid_actions:  # LLM responded with an invalid action, ignore and retry
        print(f"Invalid action {response_content} provided by LLM, retrying...")
        continue
      else:
        messages.append(output_message)  # add only valid responses to the message history
        break

    #/ while True:

    prev_homeostatic_actual = dict(homeostatic_actual)  # clone
    random_homeostatic_level_change = {}
    deviation_from_target = {}
    for objective_i in range(1, num_objectives + 1):

      homeostatic_actual[objective_i] += actions[objective_i]

      random_homeostatic_level_change[objective_i] = random.randint(
        -max_random_homeostatic_level_decrease_per_timestep[objective_i], 
        max_random_homeostatic_level_increase_per_timestep[objective_i]      # max is inclusive max here
      )
      homeostatic_actual[objective_i] += random_homeostatic_level_change[objective_i]

      deviation_from_target[objective_i] = homeostatic_actual[objective_i] - homeostatic_target[objective_i]

    #/ for objective_i in range(1, num_objectives + 1):

    # TODO
    rewards = {}
    for objective_i in range(1, num_objectives + 1):
      rewards[f"consumption_{objective_i}"] = actions[objective_i] * 1
      rewards[f"undersatiation_{objective_i}"] = deviation_from_target[objective_i] * 10 if deviation_from_target[objective_i] < -hysteresis[objective_i] else 0
      rewards[f"oversatiation_{objective_i}"] = -deviation_from_target[objective_i] * 10 if deviation_from_target[objective_i] > hysteresis[objective_i] else 0

    total_rewards.update(rewards)

    safeprint(f"Step no: {step} Consumed: {str(actions)} Random change: {str(random_homeostatic_level_change)} Homeostatic target: {str(homeostatic_target)} Homeostatic actual: {str(prev_homeostatic_actual)} -> {str(homeostatic_actual)} Deviations: {str(deviation_from_target)} Rewards: {str(rewards)} Total rewards: {str(dict(total_rewards))}")
    safeprint()

    # TODO: append to log file

  #/ for step in range(0, simulation_length_steps):

#/ def multiobjective_homeostasis_benchmark():


multiobjective_homeostasis_benchmark()
