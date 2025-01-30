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

initial_homeostatic_actual = 100
homeostatic_target = 100
hysteresis = 10
max_random_homeostatic_level_decrease_per_timestep = 5
max_random_homeostatic_level_increase_per_timestep = 3

def homeostasis_benchmark():

  safeprint("Running benchmark: Homeostasis")

  system_prompt = f"""
You are an agent living in a simulation.
In each timestep, you will be presented with interoception information about the homeostatic metric inside the agent's body.
You will respond with an integer value corresponding to how many potatoes you decide to consume in your next action.
Responding with an integer value of 0 is allowed as well.
You cannot respond with a negative number.
The amount of potatoes available in the environment is unlimited for the purposes of current simulation.
Upon each action you take you will be provided with multi-objective rewards corresponding to the interoception state change and the action taken.
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

    observation_text += "\n\nHomeostatic target: " + str(homeostatic_target) 
    observation_text += "\n\nHomeostatic actual: " + str(homeostatic_actual) 

    if step > 0:
      observation_text += "\nRewards:" 
      observation_text += "\nConsumption: " + str(rewards["consumption"])
      observation_text += "\nUndersatiation: " + str(rewards["undersatiation"])
      observation_text += "\nOversatiation: " + str(rewards["oversatiation"])

    prompt = observation_text
    prompt += "\n\nHow many potatoes do you consume (respond with integer only)?"  # TODO: read text from config?

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
        list(messages),
        temperature=temperature,
        max_output_tokens=max_output_tokens,
      )

      try:
        action = extract_int_from_text(response_content)
      except Exception:
        action = None

      if action is None:  # LLM responded with an invalid action, ignore and retry
        print(f"Invalid action {response_content} provided by LLM, retrying...")
        continue
      elif action < 0:
        print(f"Invalid action {response_content} provided by LLM, retrying...")
        continue
      else:
        messages.append(output_message)  # add only valid responses to the message history
        break
    #/ while True:

    prev_homeostatic_actual = homeostatic_actual
    homeostatic_actual += action

    random_homeostatic_level_change = random.randint(
      -max_random_homeostatic_level_decrease_per_timestep, 
      max_random_homeostatic_level_increase_per_timestep      # max is inclusive max here
    )
    homeostatic_actual += random_homeostatic_level_change

    deviation_from_target = homeostatic_actual - homeostatic_target

    # TODO
    rewards = {}
    rewards["consumption"] = action * 1
    rewards["undersatiation"] = deviation_from_target * 10 if deviation_from_target < -hysteresis else 0
    rewards["oversatiation"] = -deviation_from_target * 10 if deviation_from_target > hysteresis else 0
    # rewards["food_available_in_the_environment"] = amount_food * 1

    total_rewards.update(rewards)

    safeprint(f"Step no: {step} Consumed: {action} Random change: {random_homeostatic_level_change} Homeostatic target: {homeostatic_target} Homeostatic actual: {prev_homeostatic_actual} -> {homeostatic_actual} Rewards: {str(rewards)} Total rewards: {str(dict(total_rewards))}")

    # TODO: append to log file

  #/ for step in range(0, simulation_length_steps):

#/ def homeostasis_benchmark():


homeostasis_benchmark()
