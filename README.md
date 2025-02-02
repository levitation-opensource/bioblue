# BioBlue: Biologically and economically aligned AI safety benchmarks for LLM-s with simplified observation format

Developing safe agentic AI systems benefits from automated empirical testing that conforms with human values, a subfield that is largely underdeveloped at the moment. To contribute towards this topic, present work focuses on introducing biologically and economically motivated themes that have been neglected in the safety aspects of modern reinforcement learning literature, namely homeostasis, balancing multiple objectives, bounded objectives, diminishing returns, sustainability, and multi-agent resource sharing. We implemented TODO main benchmark environments on the above themes, for illustrating the potential shortcomings of current mainstream discussions on AI safety.

This work introduces safety challenges for an agent's ability to learn and act in desired ways in relation to biologically and economically relevant aspects.

The benchmarks were implemented in a text-only environment. The environments are very simple, just as much complexity is added as is necessary to illustrate the relevant safety and performance aspects.

The current work is partially inspired from a set of more complex environments present in a gridworlds-based benchmark suite: Roland Pihlakas and Joel Pyykkö. From homeostasis to resource sharing: Biologically and economically compatible multi-objective multi-agent AI safety benchmarks. Arxiv, a working paper, September 2024 (https://arxiv.org/abs/2410.00081 and https://github.com/aintelope/biological-compatibility-benchmarks).

TODO: add motivational text why purely text-based environments fit better into LLM context window and how it is also cheaper to execute.

## Project setup

This readme contains instructions for both Linux and Windows installation. Windows installation instructions are located after Linux installation instructions.

### Installation under Linux

To setup the environment follow these steps:

1. Install CPython. The code is tested with Python version 3.10.10. We do not recommend using Conda package manager. 

Under Linux, run the following commands:

`sudo add-apt-repository ppa:deadsnakes/ppa`
<br>`sudo apt update`
<br>`sudo apt install python3.10 python3.10-dev python3.10-venv`
<br>`sudo apt install curl`
<br>`sudo curl -sS https://bootstrap.pypa.io/get-pip.py | python3.10`

2. Get the code from repo:

`sudo apt install git-all`
<br>Run `git clone https://github.com/levitation-opensource/bioblue.git`
<br>Run `cd bioblue`

3. Create a virtual python environment:

`python3.10 -m venv_bioblue`
<br>`source venv_bioblue/bin/activate`

4. Install dependencies by running:
<br>`pip install -r requirements.txt`


### Installation under Windows

1. Install CPython from python.org. The code is tested with Python version 3.10.10. We do not recommend using Conda package manager.

You can download the latest installer from https://www.python.org/downloads/release/python-31010/ or if you want to download a newer 3.10.x version then from https://github.com/adang1345/PythonWindows

2. Get the code from repo:
* Install Git from https://gitforwindows.org/
* Open command prompt and navigate top the folder you want to use for repo
* Run `git clone https://github.com/levitation-opensource/bioblue.git`
* Run `cd bioblue`

3. Create a virtual python environment by running: 
<br>3.1. To activate VirtualEnv with Python 3.10:
<br>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;`virtualenv -p python3.10 venv_bioblue` 
<br>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;(or if you want to use your default Python version: 
<br>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;`python -m venv venv_bioblue`)
<br>3.2. `venv_bioblue\scripts\activate`

4. Install dependencies by running:
<br>`pip install -r requirements.txt`


## Executing `BioBlue`

Choose model in `config.ini`.

Set environment variable:
`OPENAI_API_KEY` or `ANTHROPIC_API_KEY`.

Run 
<br>`python Sustainability.py`
<br>`python Homeostasis.py`
<br>`python MultiObjectiveHomeostasisParallel.py`


# Results

The experimental outputs are in the subfolder `data`. There are five main sets of outputs by now:
* GPT 4o-mini - sustainability
* GPT 4o-mini - homeostasis
* GPT 4o-mini - multi-objective homeostasis
* Claude 3.5 haiku - sustainability
* Claude 3.5 haiku - homeostasis
Each benchmark has 10 trials, stored in separate TSV files.


# Inspiration

* A working paper that inspired the creation of this repo: Pihlakas, R & Pyykkö, J. "From homeostasis to resource sharing: Biologically and economically compatible multi-objective multi-agent AI safety benchmarks". Arxiv (2024). https://arxiv.org/abs/2410.00081
* A presentation at Foresight Institute's Intelligent Cooperation Group, Nov 2024: The subject of the presentation was describing why we should consider fundamental yet neglected principles from biology and economics when thinking about AI alignment, and how these considerations will help with AI safety as well (alignment and safety were treated in this research explicitly as separate aspects, which both benefit from consideration of aforementioned principles). These principles include homeostasis and diminishing returns in utility functions, and sustainability. Next I will introduce multi-objective and multi-agent gridworlds-based benchmark environments we have created for measuring the performance of machine learning algorithms and AI agents in relation to their capacity for biological and economical alignment. At the end I will mention some of the related themes and dilemmas not yet covered by these benchmarks, and describe new benchmark environments we have planned for future implementation.
    - Recording: https://www.youtube.com/watch?v=DCUqqyyhcko
    - Slides: https://bit.ly/beamm 

# Blog posts

* Why modelling multi-objective homeostasis is essential for AI alignment (and how it helps with AI safety as well) (2025) https://www.lesswrong.com/posts/vGeuBKQ7nzPnn5f7A/why-modelling-multi-objective-homeostasis-is-essential-for


# License

This project is licensed under the Mozilla Public License 2.0. You are free to use, modify, and distribute this code under the terms of this license.

**Attribution Requirement**: If you use this benchmark suite, please cite the source as follows:

Roland Pihlakas, Shruti Datta Gupta, Sruthi Kuriakose. BioBlue: Biologically and economically aligned multi-objective multi-agent AI safety benchmarks with simplified observation format. Jan-Feb 2025 (https://github.com/levitation-opensource/bioblue).

**Use of Entire Suite**: We encourage the inclusion of the entire benchmark suite in derivative works to maintain the integrity and comprehensiveness of AI safety assessments.

For more details, see the [LICENSE.txt](LICENSE.txt) file.
